#!/usr/bin/env python3
"""UDP and TCP DNS service constrained by the shared domain and address policy.

Only allowlisted Internet-class A and AAAA questions are resolved. Refused
requests receive REFUSED, resolution failures receive SERVFAIL, and successful
responses contain validated public addresses. Runtime records and heartbeat
files are written to the caller-selected state directory."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

from policy import DomainPolicy, public_addresses, write_pid_identity

QTYPE_A = 1
QTYPE_AAAA = 28
QCLASS_IN = 1

def parse_question(packet: bytes) -> tuple[int, int, str, int, int, bytes]:
    """Parse one uncompressed DNS question and reject unsupported framing."""
    if len(packet) < 17:
        raise ValueError("short DNS packet")
    txid, flags, qdcount, _an, _ns, _ar = struct.unpack("!HHHHHH", packet[:12])
    if flags & 0x8000 or ((flags >> 11) & 0xF) != 0 or qdcount != 1:
        raise ValueError("unsupported DNS query")
    offset = 12
    labels: list[str] = []
    total = 0
    while True:
        if offset >= len(packet):
            raise ValueError("truncated qname")
        length = packet[offset]
        offset += 1
        if length == 0:
            break
        if length & 0xC0 or length > 63 or offset + length > len(packet):
            raise ValueError("compressed or invalid qname")
        label = packet[offset : offset + length].decode("ascii")
        labels.append(label)
        total += length + 1
        if total > 253:
            raise ValueError("qname too long")
        offset += length
    if offset + 4 > len(packet):
        raise ValueError("truncated DNS question")
    qtype, qclass = struct.unpack("!HH", packet[offset : offset + 4])
    question = packet[12 : offset + 4]
    return txid, flags, ".".join(labels), qtype, qclass, question

def make_response(packet: bytes, policy: DomainPolicy) -> tuple[bytes, dict[str, Any]]:
    """Apply policy, resolve public addresses, and encode a DNS response and event."""
    txid, request_flags, host, qtype, qclass, question = parse_question(packet)
    allowed, reason, canonical = policy.decision(host)
    host = canonical
    addresses: list[str] = []
    rcode = 0
    if not allowed or qclass != QCLASS_IN or qtype not in {QTYPE_A, QTYPE_AAAA}:
        rcode = 5  # REFUSED
    else:
        try:
            targets = public_addresses(host, 443)
            wanted = socket.AF_INET if qtype == QTYPE_A else socket.AF_INET6
            addresses = sorted({sockaddr[0] for family, sockaddr in targets if family == wanted})
            if not addresses:
                rcode = 0
        except PermissionError as exc:
            reason = f"unsafe_dns:{exc}"
            rcode = 5
        except OSError as exc:
            reason = f"resolve_error:{exc}"
            rcode = 2  # SERVFAIL

    response_flags = 0x8000 | (request_flags & 0x0100) | 0x0080 | rcode
    header = struct.pack("!HHHHHH", txid, response_flags, 1, len(addresses), 0, 0)
    answers = bytearray()
    for address in addresses:
        packed = ipaddress.ip_address(address).packed
        answers.extend(b"\xc0\x0c")
        answers.extend(struct.pack("!HHIH", qtype, QCLASS_IN, 60, len(packed)))
        answers.extend(packed)
    event = {
        "decision": "allow" if rcode == 0 and allowed else "deny",
        "host": host,
        "qtype": qtype,
        "rcode": rcode,
        "reason": reason,
        "answers": addresses,
    }
    return header + question + bytes(answers), event

class State:
    """Serialize DNS events and expose runtime service counters."""
    def __init__(self, state_dir: Path, policy: DomainPolicy):
        self.state_dir = state_dir
        self.policy = policy
        self.started = time.time()
        self.queries = 0
        self.denied = 0
        self.errors = 0

        self.upstream_failures = 0
        self.lock = asyncio.Lock()
        state_dir.mkdir(parents=True, exist_ok=True)
        self.records = state_dir / "dns.jsonl"
        self.heartbeat_path = state_dir / "dns-heartbeat.json"

    async def record(self, peer: Any, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), "client": str(peer[0]) if peer else "unknown", **event}
        async with self.lock:
            with self.records.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "ts": time.time(),
            "started": self.started,
            "policy_sha256": self.policy.digest,
            "queries": self.queries,
            "denied": self.denied,
            "errors": self.errors,
            "upstream_failures": self.upstream_failures,
        }

    async def heartbeat(self) -> None:
        while True:
            tmp = self.heartbeat_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.heartbeat_path)
            await asyncio.sleep(2)

class UDPProtocol(asyncio.DatagramProtocol):
    """Handle DNS datagrams through the shared response builder."""
    def __init__(self, state: State):
        self.state = state
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: Any) -> None:
        asyncio.create_task(self._answer(data, addr))

    async def _answer(self, data: bytes, addr: Any) -> None:
        self.state.queries += 1
        try:
            response, event = await asyncio.to_thread(make_response, data, self.state.policy)
            if event["decision"] == "deny":
                self.state.denied += 1
            if self.transport is not None:
                self.transport.sendto(response, addr)
            await self.state.record(addr, event)
        except (OSError, ConnectionError, TimeoutError) as exc:
            self.state.upstream_failures += 1
            await self.state.record(addr, {"decision": "upstream_error", "reason": f"{type(exc).__name__}:{exc}"})
        except Exception as exc:
            self.state.errors += 1
            await self.state.record(addr, {"decision": "error", "reason": f"{type(exc).__name__}:{exc}"})

async def handle_tcp(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: State
) -> None:
    """Handle a length-prefixed DNS question over TCP."""
    peer = writer.get_extra_info("peername")
    try:
        length_bytes = await asyncio.wait_for(reader.readexactly(2), timeout=10)
        length = struct.unpack("!H", length_bytes)[0]
        if length <= 0 or length > 4096:
            raise ValueError("invalid TCP DNS length")
        packet = await asyncio.wait_for(reader.readexactly(length), timeout=10)
        state.queries += 1
        response, event = await asyncio.to_thread(make_response, packet, state.policy)
        if event["decision"] == "deny":
            state.denied += 1
        writer.write(struct.pack("!H", len(response)) + response)
        await writer.drain()
        await state.record(peer, event)
    except (OSError, ConnectionError, TimeoutError) as exc:
        state.upstream_failures += 1
        await state.record(peer, {"decision": "upstream_error", "reason": f"{type(exc).__name__}:{exc}"})
    except Exception as exc:
        state.errors += 1
        await state.record(peer, {"decision": "error", "reason": f"{type(exc).__name__}:{exc}"})
    finally:
        writer.close()
        await asyncio.gather(writer.wait_closed(), return_exceptions=True)

async def amain(args: argparse.Namespace) -> None:
    policy = DomainPolicy.load(Path(args.allowlist), Path(args.denylist))
    state = State(Path(args.state_dir), policy)
    loop = asyncio.get_running_loop()
    udp_transport, _ = await loop.create_datagram_endpoint(
        lambda: UDPProtocol(state), local_addr=(args.host, args.port)
    )
    tcp_server = await asyncio.start_server(
        lambda reader, writer: handle_tcp(reader, writer, state), args.host, args.port
    )
    write_pid_identity(Path(args.state_dir) / "dns.pid")
    heartbeat_task = asyncio.create_task(state.heartbeat())
    print(
        f"[dns-proxy] listening {args.host}:{args.port} policy={policy.digest}",
        flush=True,
    )
    try:
        async with tcp_server:
            await tcp_server.serve_forever()
    finally:
        udp_transport.close()
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=53)
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--denylist", required=True)
    parser.add_argument("--state-dir", required=True)
    asyncio.run(amain(parser.parse_args()))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

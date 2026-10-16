#!/usr/bin/env python3
"""HTTP and CONNECT proxy constrained by the shared domain and address policy.

The proxy validates request framing and destination ports, limits client
connections, and either connects to validated public addresses or forwards to
an explicitly configured parent proxy. A parent proxy receives the destination
hostname and performs its own connection handling. Transport failures and
policy refusals are recorded separately from internal proxy errors."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from policy import (
    ClientLimiter,
    DomainPolicy,
    normalize_host,
    public_addresses,
    write_pid_identity,
)

MAX_HEADER_BYTES = 64 * 1024

class State:
    """Keep runtime connection counters and serialize event records."""
    def __init__(self, state_dir: Path, policy: DomainPolicy):
        self.state_dir = state_dir
        self.policy = policy
        self.started = time.time()
        self.accepted = 0
        self.denied = 0
        self.throttled = 0

        # Internal failures are separate from ordinary upstream transport failures.
        self.errors = 0
        self.upstream_failures = 0
        self.active = 0
        self.peak_active = 0
        self._record_lock = asyncio.Lock()
        self.parent = ""
        self.allow_clients: frozenset[str] = frozenset()
        state_dir.mkdir(parents=True, exist_ok=True)
        self.record_path = state_dir / "egress.jsonl"
        self.heartbeat_path = state_dir / "egress-heartbeat.json"

    async def record(self, event: dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        line = json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n"
        async with self._record_lock:
            with self.record_path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "ts": time.time(),
            "started": self.started,
            "policy_sha256": self.policy.digest,
            "accepted": self.accepted,
            "denied": self.denied,
            "throttled": self.throttled,
            "errors": self.errors,
            "upstream_failures": self.upstream_failures,
            "active": self.active,
            "peak_active": self.peak_active,
            "parent": self.parent,
            "allow_clients": sorted(self.allow_clients),
        }

    async def heartbeat(self) -> None:
        while True:
            tmp = self.heartbeat_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8"
            )
            tmp.replace(self.heartbeat_path)
            await asyncio.sleep(2)

def _parse_authority(authority: str) -> tuple[str, int]:
    """Parse a CONNECT destination with an explicit port."""
    authority = authority.strip()
    if authority.startswith("["):
        end = authority.find("]")
        if end < 0 or end + 1 >= len(authority) or authority[end + 1] != ":":
            raise ValueError("CONNECT authority must include a port")
        return authority[1:end], int(authority[end + 2 :])
    if ":" not in authority:
        raise ValueError("CONNECT authority must include a port")
    host, port = authority.rsplit(":", 1)
    return host, int(port)

def _parse_headers(blob: bytes) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Validate an HTTP request line and unfolded headers within the size limit."""
    if len(blob) > MAX_HEADER_BYTES:
        raise ValueError("request headers too large")
    text = blob.decode("iso-8859-1")
    lines = text.split("\r\n")
    try:
        method, target, version = lines[0].split(" ", 2)
    except ValueError as exc:
        raise ValueError("invalid HTTP request line") from exc
    if version not in {"HTTP/1.0", "HTTP/1.1"} or not method.isalpha():
        raise ValueError("unsupported HTTP request")
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        if line[0].isspace() or ":" not in line:
            raise ValueError("folded or invalid HTTP header")
        name, value = line.split(":", 1)
        if not name or not all(ch.isalnum() or ch in "-_" for ch in name):
            raise ValueError("invalid HTTP header name")
        headers.append((name, value.strip()))
    return method.upper(), target, version, headers

def _header(headers: list[tuple[str, str]], name: str) -> str | None:
    """Read a header while rejecting ambiguous duplicate values."""
    values = [value for key, value in headers if key.lower() == name.lower()]
    if len(values) > 1:
        raise ValueError(f"duplicate {name} header")
    return values[0] if values else None

async def _open_public(
    host: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:
    """Connect directly to a numerically pinned, policy-validated public address."""
    targets = await asyncio.to_thread(public_addresses, host, port)
    failures: list[str] = []
    for family, sockaddr in targets:
        address = sockaddr[0]
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host=address,
                    port=port,
                    family=family,
                    flags=socket.AI_NUMERICHOST,
                ),
                timeout=20,
            )
            return reader, writer, address
        except Exception as exc:
            failures.append(f"{address}:{type(exc).__name__}")
    raise OSError("all validated addresses failed: " + ",".join(failures))

async def _open_via_parent(
    parent: tuple[str, int], host: str, port: int, method: str
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:

    """Open a parent connection and establish a CONNECT tunnel when required."""
    parent_host, parent_port = parent
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host=parent_host, port=parent_port), timeout=20
    )
    if method == "CONNECT":
        writer.write(
            f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n".encode("iso-8859-1")
        )
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        status_line = head.split(b"\r\n", 1)[0].decode("iso-8859-1", "replace")
        parts = status_line.split(" ", 2)
        if len(parts) < 2 or not parts[1].isdigit() or not 200 <= int(parts[1]) < 300:
            writer.close()
            await asyncio.gather(writer.wait_closed(), return_exceptions=True)
            raise PermissionError(f"parent proxy refused CONNECT {host}:{port}: {status_line[:120]}")
    return reader, writer, f"parent:{parent_host}:{parent_port}"

class _RelayActivity:
    """Share an idle deadline between both directions of a relayed connection."""
    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.last_activity = self._loop.time()

    def mark(self) -> None:
        self.last_activity = self._loop.time()

    def remaining(self, idle_timeout: float) -> float:
        return self.last_activity + idle_timeout - self._loop.time()

def _half_close(destination: asyncio.StreamWriter) -> bool:
    """Signal end of one stream direction when the transport supports it."""
    try:
        destination.write_eof()
    except (AttributeError, OSError, RuntimeError):
        return False
    return True

async def _pipe(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter,
    activity: _RelayActivity,
) -> str:
    """Copy bytes in one direction while updating shared relay activity."""
    try:
        while True:
            data = await source.read(256 * 1024)
            if not data:
                return "eof" if _half_close(destination) else "error"
            destination.write(data)
            await destination.drain()
            activity.mark()
    except (ConnectionError, OSError):
        return "error"

async def _relay(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
    idle_timeout: float,
) -> None:
    """Relay both stream directions under a shared idle timeout."""
    if not idle_timeout > 0:
        raise ValueError("relay idle timeout must be positive")
    activity = _RelayActivity()
    tasks = {
        asyncio.create_task(
            _pipe(client_reader, upstream_writer, activity),
            name="client-to-upstream",
        ): "client_to_upstream",
        asyncio.create_task(
            _pipe(upstream_reader, client_writer, activity),
            name="upstream-to-client",
        ): "upstream_to_client",
    }
    try:
        while tasks:
            remaining = activity.remaining(idle_timeout)
            if remaining <= 0:
                break
            done, _ = await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:

                if activity.remaining(idle_timeout) <= 0:
                    break
                continue
            terminate = False
            for task in done:
                direction = tasks.pop(task)
                outcome = task.result()
                if outcome == "error" or direction == "upstream_to_client":
                    terminate = True
            if terminate:
                break
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for writer in (upstream_writer, client_writer):
            try:
                writer.close()
            except (ConnectionError, OSError, RuntimeError):
                pass
        await asyncio.gather(
            upstream_writer.wait_closed(),
            client_writer.wait_closed(),
            return_exceptions=True,
        )

async def _respond(
    writer: asyncio.StreamWriter, status: int, reason: str, body: dict[str, Any]
) -> None:
    """Send a bounded HTTP error response and close the stream."""
    payload = (json.dumps(body, sort_keys=True) + "\n").encode()
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
        + payload
    )
    await writer.drain()
    writer.close()
    await writer.wait_closed()

class Proxy:
    """Validate destinations and relay allowed client requests."""
    def __init__(
        self,
        state: State,
        limiter: ClientLimiter,
        idle_timeout: float,
        upstream_proxy: tuple[str, int] | None = None,
        allow_clients: frozenset[str] = frozenset(),
    ):
        self.state = state
        self.limiter = limiter
        self.idle_timeout = idle_timeout
        self.upstream_proxy = upstream_proxy
        self.allow_clients = allow_clients

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername") or ("unknown", 0)
        client = str(peer[0])
        acquired = False
        host = ""
        try:
            if self.allow_clients and client not in self.allow_clients:
                raise PermissionError(f"client_not_allowed:{client}")
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
            method, target, version, headers = _parse_headers(raw)

            if method == "GET" and target == "/__guard__/health":
                await _respond(writer, 200, "OK", self.state.snapshot())
                return

            if method == "CONNECT":
                host, port = _parse_authority(target)
                if port != 443:
                    raise PermissionError("CONNECT is restricted to port 443")
                origin_path = None
            else:
                parsed = urlsplit(target)
                if parsed.scheme.lower() != "http" or not parsed.hostname:
                    raise PermissionError(
                        "plain proxy requests must use an absolute http URL"
                    )
                host = parsed.hostname
                port = parsed.port or 80
                if port != 80:
                    raise PermissionError("plain HTTP is restricted to port 80")
                origin_path = parsed.path or "/"
                if parsed.query:
                    origin_path += "?" + parsed.query
                host_header = _header(headers, "host")
                if host_header:
                    header_host = host_header.rsplit(":", 1)[0].strip("[]")
                    if normalize_host(header_host) != normalize_host(host):
                        raise PermissionError("absolute URL and Host header disagree")

            allowed, reason, canonical = self.state.policy.decision(host)
            host = canonical
            if not allowed:
                raise PermissionError(reason)
            allowed, reason = self.limiter.acquire(client, host)
            waitable_limits = {
                "new_connection_rate_limit",
                "active_connection_limit",
            }
            if not allowed and reason in waitable_limits:

                deadline = time.monotonic() + 60
                self.state.throttled += 1
                await self.state.record(
                    {
                        "decision": "throttle",
                        "client": client,
                        "host": host,
                        "reason": reason,
                    }
                )
                while not allowed and reason in waitable_limits:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    delay = (
                        self.limiter.rate_retry_after(client)
                        if reason == "new_connection_rate_limit"
                        else 0.1
                    )
                    delay = min(delay, remaining)
                    await asyncio.sleep(max(0.01, delay))
                    allowed, reason = self.limiter.acquire(client, host)
            if not allowed:
                raise PermissionError(reason)
            acquired = True
            self.state.active += 1
            self.state.peak_active = max(self.state.peak_active, self.state.active)

            if self.upstream_proxy is None:
                upstream_reader, upstream_writer, address = await _open_public(host, port)
                validated = address
            else:

                targets = await asyncio.to_thread(public_addresses, host, port)
                validated = targets[0][1][0] if targets else ""
                upstream_reader, upstream_writer, address = await _open_via_parent(
                    self.upstream_proxy, host, port, method
                )
            self.state.accepted += 1
            await self.state.record(
                {
                    "decision": "allow",
                    "client": client,
                    "method": method,
                    "host": host,
                    "port": port,
                    "resolved": address,
                    "validated": validated,
                }
            )

            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                kept = []
                for name, value in headers:
                    if name.lower() in {
                        "proxy-authorization",
                        "proxy-connection",
                        "connection",
                        "keep-alive",
                    }:
                        continue
                    kept.append((name, value))
                kept.append(("Connection", "close"))

                request_target = origin_path if self.upstream_proxy is None else f"http://{host}{origin_path}"
                request = (
                    f"{method} {request_target} {version}\r\n"
                    + "".join(f"{name}: {value}\r\n" for name, value in kept)
                    + "\r\n"
                )
                upstream_writer.write(request.encode("iso-8859-1"))
                await upstream_writer.drain()

            await _relay(
                reader, writer, upstream_reader, upstream_writer, self.idle_timeout
            )
        except PermissionError as exc:
            self.state.denied += 1
            await self.state.record(
                {
                    "decision": "deny",
                    "client": client,
                    "host": host,
                    "reason": str(exc)[:300],
                }
            )
            if not writer.is_closing():
                await _respond(
                    writer,
                    403,
                    "Forbidden",
                    {"error": "egress denied", "reason": str(exc)},
                )
        except (
            ValueError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
        ) as exc:
            self.state.denied += 1
            await self.state.record(
                {
                    "decision": "deny",
                    "client": client,
                    "reason": f"bad_request:{exc}"[:300],
                }
            )
            if not writer.is_closing():
                await _respond(
                    writer, 400, "Bad Request", {"error": "invalid proxy request"}
                )
        except (OSError, ConnectionError, TimeoutError) as exc:

            self.state.upstream_failures += 1
            await self.state.record(
                {
                    "decision": "upstream_error",
                    "client": client,
                    "host": host,
                    "reason": f"{type(exc).__name__}:{exc}"[:300],
                }
            )
            if not writer.is_closing():
                await _respond(
                    writer, 502, "Bad Gateway", {"error": "upstream unavailable"}
                )
        except Exception as exc:
            self.state.errors += 1
            await self.state.record(
                {
                    "decision": "error",
                    "client": client,
                    "host": host,
                    "reason": f"{type(exc).__name__}:{exc}"[:300],
                }
            )
            if not writer.is_closing():
                await _respond(
                    writer, 502, "Bad Gateway", {"error": "upstream unavailable"}
                )
        finally:
            if acquired:
                self.limiter.release(client)
                self.state.active = max(0, self.state.active - 1)

async def amain(args: argparse.Namespace) -> None:
    policy = DomainPolicy.load(Path(args.allowlist), Path(args.denylist))
    state = State(Path(args.state_dir), policy)
    limiter = ClientLimiter(args.max_active, args.max_new_10s, args.max_domains)
    upstream_proxy: tuple[str, int] | None = None
    if args.upstream_proxy:
        upstream_proxy = _parse_authority(args.upstream_proxy)
    allow_clients = frozenset(c.strip() for c in args.allow_clients.split(",") if c.strip())
    state.parent = args.upstream_proxy or ""
    state.allow_clients = allow_clients
    proxy = Proxy(state, limiter, args.idle_timeout, upstream_proxy, allow_clients)
    server = await asyncio.start_server(
        proxy.handle,
        host=args.host,
        port=args.port,
        limit=MAX_HEADER_BYTES + 1,

        reuse_address=True,
    )
    write_pid_identity(Path(args.state_dir) / "egress.pid")
    heartbeat_task = asyncio.create_task(state.heartbeat())
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(
        f"[egress-proxy] listening {sockets} policy={policy.digest} "
        f"parent={args.upstream_proxy or '-'} allow_clients={args.allow_clients or 'any'}",
        flush=True,
    )
    try:
        async with server:
            await server.serve_forever()
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--allowlist", required=True)
    parser.add_argument("--denylist", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--max-active", type=int, default=64)
    parser.add_argument("--max-new-10s", type=int, default=120)
    parser.add_argument("--max-domains", type=int, default=64)
    parser.add_argument("--idle-timeout", type=float, default=900)
    parser.add_argument(
        "--upstream-proxy",
        default="",
        help="HOST:PORT of a parent instance of this proxy; the outbound leg of every locally validated "
        "connection goes there instead of directly to the validated address",
    )
    parser.add_argument(
        "--allow-clients",
        default="",
        help="comma-separated client IPs allowed to use this instance; anything else is refused before "
        "its request is read (empty = no client restriction)",
    )
    args = parser.parse_args()
    asyncio.run(amain(args))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

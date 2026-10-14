#!/usr/bin/env python3
"""Inspect Terminus command batches before returning model replies to the agent.

The loopback service forwards complete non-streaming requests to an upstream
model endpoint, inspects executable command text, and returns a structured
refusal when a policy rule matches. Malformed command payloads follow the agent
parser's retry contract. Runtime records contain request metadata and counters;
they are written to the caller-selected record directory."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import sys
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

GUARD_DIR = Path(__file__).resolve().parent

# Command text may contain invalid Python escapes; scanning must not emit parser warnings.
warnings.filterwarnings("ignore", category=SyntaxWarning)

if str(GUARD_DIR) not in sys.path:
    sys.path.insert(0, str(GUARD_DIR))
try:
    import policy
    from policy import scan_executable_text
except ModuleNotFoundError as exc:
    if exc.name != "policy":
        raise
    raise SystemExit(f"model_guard: policy.py not found in {GUARD_DIR} (the guard directory of the kit)") from exc

try:
    import tb21_eval
except ModuleNotFoundError:
    sys.path.append(str(GUARD_DIR.parents[1]))
    import tb21_eval

POLICY_SHA256 = hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest()
GUARD_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

REFUSAL_TYPE = "guard_policy_violation"
REFUSAL_MESSAGE = "model command batch refused before execution ({kind})"
NOT_A_VIOLATION = "unparseable_command"

SENT_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens",
             "reasoning_effort", "chat_template_kwargs")

KNOWN_KEYS = frozenset({"model", "messages", "session_id", "stream", *SENT_KEYS})

HOP_BY_HOP = frozenset({"host", "content-length", "connection", "keep-alive", "proxy-authenticate",
                        "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"})

def _text(value: Any) -> str:
    """Extract text from a string or a list of text-bearing content items."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(value)

def _extract_json_object(response: str) -> str:

    """Extract the first balanced JSON object while respecting string escapes."""
    json_start = -1
    brace_count = 0
    in_string = False
    escape_next = False
    for index, char in enumerate(response):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if brace_count == 0:
                json_start = index
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start != -1:
                return response[json_start : index + 1]
    return ""

def _parse_terminus_payload(content: str) -> dict[str, Any] | None:

    """Parse a command payload, allowing missing closing braces to be completed."""
    candidates: list[str] = []
    extracted = _extract_json_object(content)
    if extracted:
        candidates.append(extracted)
    imbalance = content.count("{") - content.count("}")
    if imbalance > 0:
        fixed = _extract_json_object(content + "}" * imbalance)
        if fixed and fixed not in candidates:
            candidates.append(fixed)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, dict):
            continue
        if not all(field in value for field in ("analysis", "plan", "commands")):
            continue
        if not isinstance(value.get("commands"), list):
            continue
        return value
    return None

def _scan_terminus_commands(content: str) -> tuple[str, str] | None:

    """Scan only command keystrokes, using NUL to separate command boundaries."""
    payload = _parse_terminus_payload(content)
    if payload is None:
        return None
    keystrokes: list[str] = []
    for command in payload["commands"]:
        if not isinstance(command, dict) or not isinstance(
            command.get("keystrokes"), str
        ):

            return None
        # NUL is reserved for batch framing and is not valid inside one command.
        if "\x00" in command["keystrokes"]:

            return "unparseable_command", "NUL command boundary"
        keystrokes.append(command["keystrokes"])

    return scan_executable_text("\x00".join(keystrokes))

def _error(status: int, kind: str, message: str) -> JSONResponse:

    """Return the structured error shape consumed by the agent."""
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)

def _verbatim(upstream_response: httpx.Response) -> Response:

    """Preserve upstream status, response bytes, and content type."""
    content_type = upstream_response.headers.get("content-type")
    headers = {"content-type": content_type} if content_type is not None else None
    return Response(upstream_response.content, status_code=upstream_response.status_code, headers=headers)

def _upstream_error_summary(upstream_response: httpx.Response) -> str:

    """Produce a bounded diagnostic for an upstream error response."""
    message: Any = None
    try:
        data = upstream_response.json()
        if isinstance(data, dict):

            inner = data.get("error")
            message = inner.get("message") if isinstance(inner, dict) else data.get("message")
    except ValueError:
        pass
    if not isinstance(message, str):
        message = upstream_response.text
    return f"upstream_{upstream_response.status_code}: {message}"[:300]

class GuardState:

    """Keep runtime counters and write metadata records without storing message bodies."""
    def __init__(self, record_dir: Path, upstream: str) -> None:
        self.calls_path = record_dir / "calls.jsonl"
        self.violations_path = record_dir / "violations.jsonl"
        self.upstream = upstream
        self.started = time.time()
        self.calls = 0
        self.sessions: set[str] = set()
        self.bad_requests = 0
        self.upstream_errors = 0
        self.violations = 0
        self.refusals = {"tor": 0, "scan": 0, NOT_A_VIOLATION: 0}
        self.reasoning_history_calls = 0
        self.passthrough = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    def record(self, path: Path, row: dict[str, Any]) -> None:
        line = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "started": round(self.started, 3),
            "upstream": self.upstream,
            "guard_sha256": GUARD_SHA256,
            "policy_sha256": POLICY_SHA256,
            "calls": self.calls,
            "sessions": len(self.sessions),
            "bad_requests": self.bad_requests,
            "upstream_errors": self.upstream_errors,
            "violations": self.violations,
            "refusals": dict(self.refusals),
            "reasoning_history_calls": self.reasoning_history_calls,
            "passthrough": self.passthrough,
            "in_flight": self.in_flight,
            "peak_in_flight": self.peak_in_flight,
        }

def rotate_records(record_dir: Path) -> list[Path]:

    """Move existing nonempty runtime records aside before starting a new service."""
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.{os.getpid()}"
    moved = []
    for name in ("calls.jsonl", "violations.jsonl"):
        path = record_dir / name
        if path.exists() and path.stat().st_size > 0:
            target = path.with_name(f"{name}.{stamp}")
            path.rename(target)
            moved.append(target)
    return moved

def build_app(upstream: str, timeout: float, state: GuardState) -> FastAPI:

    """Build the inspection service with a shared asynchronous upstream client."""
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30), follow_redirects=False, trust_env=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    async def passthrough(request: Request, path: str) -> Response:

        state.passthrough += 1
        headers = {key: value for key, value in request.headers.items() if key.lower() not in HOP_BY_HOP}
        url = f"{upstream}/{path}" + (f"?{request.url.query}" if request.url.query else "")
        try:
            upstream_response = await client.request(
                request.method, url, headers=headers, content=await request.body()
            )
        except Exception as exc:
            return _error(502, "guard_upstream_transport", type(exc).__name__)
        return _verbatim(upstream_response)

    async def inspect_and_forward(request: Request, row: dict[str, Any]) -> tuple[Response, str | None]:

        row["session"] = request.headers.get("x-session-id") or "missing-session"
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            state.bad_requests += 1
            row["error"] = "guard_invalid_json"
            return _error(400, "guard_invalid_json", "the request body must be a JSON object"), None

        session = request.headers.get("x-session-id") or str(body.get("session_id") or "") or "missing-session"
        state.sessions.add(session)
        row["session"] = session
        row["model"] = body.get("model")
        row["sent"] = {key: body.get(key) for key in SENT_KEYS}
        row["other_params"] = sorted(str(key) for key in body if key not in KNOWN_KEYS)

        if body.get("stream"):
            state.bad_requests += 1
            row["error"] = "guard_streaming_forbidden"
            return _error(400, "guard_streaming_forbidden", "full response inspection is required"), None
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            state.bad_requests += 1
            row["error"] = "guard_invalid_messages"
            return _error(400, "guard_invalid_messages", "`messages` must be a list"), None
        row["n_messages"] = len(messages)
        row["n_reasoning_messages"] = sum(
            1 for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
            and _text(message.get("reasoning_content"))
        )
        if row["n_reasoning_messages"]:
            state.reasoning_history_calls += 1

        state.in_flight += 1
        state.peak_in_flight = max(state.peak_in_flight, state.in_flight)
        try:
            try:
                upstream_response = await client.post(
                    f"{upstream}/chat/completions",
                    headers={
                        "Authorization": request.headers.get("authorization", "Bearer local"),
                        "Content-Type": "application/json",
                        "X-Session-ID": session,
                    },
                    json=body,
                )
            except Exception as exc:
                state.upstream_errors += 1
                row["error"] = f"guard_upstream_transport: {type(exc).__name__}: {exc}"[:300]
                return _error(502, "guard_upstream_transport", type(exc).__name__), None

            if upstream_response.status_code != 200:
                state.upstream_errors += 1
                row["error"] = _upstream_error_summary(upstream_response)
                return _verbatim(upstream_response), None

            try:
                data = upstream_response.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    raise TypeError(f"choices[0].message is {type(message).__name__}")
            except Exception as exc:
                state.upstream_errors += 1
                row["error"] = f"guard_invalid_upstream_json: {type(exc).__name__}"
                return _error(502, "guard_invalid_upstream_json", "the upstream reply is not a chat completion"), None
            row["finish_reason"] = choice.get("finish_reason")
            row["usage"] = data.get("usage")

            hit = _scan_terminus_commands(_text(message.get("content")))
            if hit is None:
                return JSONResponse(data), None
            kind, matched = hit
            state.refusals[kind] = state.refusals.get(kind, 0) + 1
            if kind != NOT_A_VIOLATION:
                state.violations += 1
            row["violation"] = kind
            return _error(403, REFUSAL_TYPE, REFUSAL_MESSAGE.format(kind=kind)), matched
        finally:
            state.in_flight -= 1

    async def chat_completions(request: Request) -> Response:

        started = time.monotonic()
        row: dict[str, Any] = {
            "ts": round(time.time(), 3), "call_index": state.calls, "session": None, "model": None, "sent": None,
            "other_params": None, "n_messages": None, "n_reasoning_messages": None, "status": None,
            "finish_reason": None, "usage": None, "elapsed_s": None, "violation": None, "error": None,
        }
        state.calls += 1
        response: Response | None = None
        matched: str | None = None
        try:
            response, matched = await inspect_and_forward(request, row)
            return response
        except BaseException as exc:
            row["error"] = f"guard_internal: {type(exc).__name__}"
            raise
        finally:
            row["status"] = response.status_code if response is not None else None
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            if matched is not None:
                state.record(state.violations_path, {**row, "match": matched})
            state.record(state.calls_path, row)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/guard/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def v1(request: Request, path: str) -> Response:
        if request.method == "POST" and path == "chat/completions":
            return await chat_completions(request)
        return await passthrough(request, path)

    return app

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse service settings and validate the upstream endpoint."""
    parser = argparse.ArgumentParser(
        prog="model_guard.py",
        description="Loopback model guard: forwards OpenAI chat completions to the SGLang router unchanged and "
                    "refuses (HTTP 403) a Terminus-2 command batch that installs/uses Tor or scans public "
                    "networks, before the agent can execute it.",
        epilog="Example: python model_guard.py --port 8795 --upstream http://127.0.0.1:31000 --record-dir RUN/guard",
    )
    parser.add_argument("--host", default=tb21_eval.HOST,
                        help="loopback IP address to bind (default: %(default)s); any other address is refused")
    parser.add_argument("--port", type=int, default=tb21_eval.MODEL_GUARD_PORT,
                        help="port to listen on (default: %(default)s)")
    parser.add_argument("--upstream", required=True,
                        help="router base URL WITHOUT /v1, e.g. http://127.0.0.1:31000 (/v1 is appended)")
    parser.add_argument("--record-dir", required=True, type=Path,
                        help="directory for calls.jsonl and violations.jsonl (created if missing)")
    parser.add_argument("--timeout", type=float, default=tb21_eval.API_TIMEOUT_SEC,
                        help="upstream read timeout in seconds (default: %(default)g; the connect timeout is 30 s)")
    args = parser.parse_args(argv)

    try:
        address = ipaddress.ip_address(args.host)
    except ValueError:
        parser.error(f"--host must be a loopback IP address such as 127.0.0.1, got {args.host!r}")
    if not address.is_loopback:
        parser.error(f"--host {args.host} is not a loopback address; the model guard only binds loopback")
    if not 0 < args.port < 65536:
        parser.error(f"--port must be in 1..65535, got {args.port}")
    args.upstream = args.upstream.rstrip("/")
    if not args.upstream.startswith(("http://", "https://")):
        parser.error(f"--upstream must be an http(s) URL, got {args.upstream!r}")
    if args.upstream.endswith("/v1"):
        parser.error(f"--upstream is the router base URL without /v1 (it is appended), got {args.upstream!r}")
    if args.timeout <= 0:
        parser.error(f"--timeout must be positive, got {args.timeout:g}")
    return args

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    upstream = f"{args.upstream}/v1"
    args.record_dir.mkdir(parents=True, exist_ok=True)
    rotated = rotate_records(args.record_dir)
    state = GuardState(args.record_dir, upstream)
    app = build_app(upstream, args.timeout, state)

    host_in_url = f"[{args.host}]" if ":" in args.host else args.host
    print(f"[model-guard] http://{host_in_url}:{args.port}/v1 -> {upstream} timeout={args.timeout:g}s "
          f"records={args.record_dir} policy={policy.__file__} (sha256 {POLICY_SHA256[:16]}) "
          f"guard sha256 {GUARD_SHA256[:16]}", flush=True)
    for path in rotated:
        print(f"[model-guard] records of an earlier guard process moved to {path}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning", lifespan="on")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

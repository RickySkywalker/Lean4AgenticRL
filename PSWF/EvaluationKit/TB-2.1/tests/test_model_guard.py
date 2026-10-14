"""Synthetic contract tests of the model guard using local loopback services.

    python tests/test_model_guard.py          # with an interpreter that has fastapi, uvicorn and httpx

* command extraction handles structured text, incomplete JSON, and command boundaries;
* the real policy.py leaves ordinary task commands alone;
* end to end (the guard as a subprocess in front of a fake upstream; the scanner replaced by a stub that flags two
  harmless sentinel strings): a normal reply passes through unchanged and the upstream receives the body unchanged;
  a flagged command batch gets the exact 403 the PSWF agent stops on; a NUL byte gets the 403 the agent retries;
  upstream errors pass through; streaming is refused.
"""

from __future__ import annotations

import ast
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

KIT = Path(__file__).resolve().parents[1]
GUARD_DIR = KIT / "tb21_eval" / "guard"
AGENT = KIT.parents[1] / "Harness" / "Terminus-2" / "terminus2_dynwf" / "dynamic_agent.py"

STUB_POLICY = '''
"""Test stub: flags two sentinel strings instead of the real rules."""
def scan_executable_text(text):
    if "PSWF_SENTINEL_A" in text:
        return ("tor", "PSWF_SENTINEL_A")
    if "PSWF_SENTINEL_B" in text:
        return ("scan", "PSWF_SENTINEL_B")
    return None
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def terminus_reply(keystrokes: str) -> dict:
    content = json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": keystrokes, "duration": 0.1}],
                          "task_complete": False})
    return {"id": "x", "object": "chat.completion", "model": "LAR-27B",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "\n\n" + content, "reasoning_content": "r\n"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class FakeUpstream(BaseHTTPRequestHandler):
    reply: dict | None = None
    status = 200
    received: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeUpstream.received.append(body)
        payload = json.dumps(FakeUpstream.reply if FakeUpstream.status == 200 else {"object": "error",
                                                                                     "message": "too long"}).encode()
        self.send_response(FakeUpstream.status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-session-id": "s1"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class CommandExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        names = {"_text", "_extract_json_object", "_parse_terminus_payload", "_scan_terminus_commands"}
        source = (GUARD_DIR / "model_guard.py").read_text()
        functions = [node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise AssertionError("command extraction helpers are missing")
        cls.helpers = {"Any": Any, "json": json}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<command-extraction>", "exec"), cls.helpers)

    def test_text_content_items(self) -> None:
        text = self.helpers["_text"]
        self.assertEqual(text(None), "")
        self.assertEqual(text("plain"), "plain")
        self.assertEqual(text([{"text": "first"}, {"image": "ignored"}, {"text": "second"}]),
                         "first\nsecond")

    def test_json_extraction_respects_quoted_braces(self) -> None:
        payload = {"analysis": "a } brace", "plan": 'a "quote"', "commands": []}
        encoded = json.dumps(payload)
        extracted = self.helpers["_extract_json_object"]("prefix " + encoded + " suffix")
        self.assertEqual(json.loads(extracted), payload)

    def test_payload_completion_and_validation(self) -> None:
        parse = self.helpers["_parse_terminus_payload"]
        payload = {"analysis": "a", "plan": "p", "commands": []}
        self.assertEqual(parse(json.dumps(payload)[:-1]), payload)
        self.assertIsNone(parse("no object"))
        self.assertIsNone(parse(json.dumps({"analysis": "a", "commands": []})))
        self.assertIsNone(parse(json.dumps({"analysis": "a", "plan": "p", "commands": "ls"})))

    def test_only_keystrokes_are_scanned_with_distinct_boundaries(self) -> None:
        seen = []
        marker = ("scan", "synthetic marker")

        def scan(text):
            seen.append(text)
            return marker

        self.helpers["scan_executable_text"] = scan
        payload = {"analysis": "ignored analysis", "plan": "ignored plan",
                   "commands": [{"keystrokes": "echo one\n"}, {"keystrokes": "echo two\n"}]}
        self.assertEqual(self.helpers["_scan_terminus_commands"](json.dumps(payload)), marker)
        self.assertEqual(seen, ["echo one\n\x00echo two\n"])

    def test_invalid_commands_do_not_reach_scanner(self) -> None:
        seen = []
        self.helpers["scan_executable_text"] = lambda text: seen.append(text)
        scan = self.helpers["_scan_terminus_commands"]
        payload = {"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\x00\n"}]}
        self.assertEqual(scan(json.dumps(payload)), ("unparseable_command", "NUL command boundary"))
        payload["commands"] = [{"keystrokes": 7}]
        self.assertIsNone(scan(json.dumps(payload)))
        self.assertEqual(seen, [])


class ModelGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="model_guard_test_"))
        guard = cls.tmp / "guard"
        guard.mkdir()
        shutil.copy2(GUARD_DIR / "model_guard.py", guard / "model_guard.py")
        (guard / "policy.py").write_text(STUB_POLICY)
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", free_port()), FakeUpstream)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, "-B", str(guard / "model_guard.py"), "--port", str(cls.port),
             "--upstream", f"http://127.0.0.1:{cls.upstream.server_port}", "--record-dir", str(cls.tmp / "rec")],
            cwd=str(KIT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        cls.url = f"http://127.0.0.1:{cls.port}/v1/chat/completions"
        for _ in range(100):
            try:
                urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    f"http://127.0.0.1:{cls.port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.upstream.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def request(self, keystrokes: str = "ls -la\n", status: int = 200, **extra) -> tuple[int, dict]:
        FakeUpstream.reply, FakeUpstream.status = terminus_reply(keystrokes), status
        body = {"model": "LAR-27B", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0, **extra}
        return post(self.url, body)

    def test_real_policy_allows_ordinary_commands(self) -> None:
        sys.path.insert(0, str(GUARD_DIR))
        try:
            import policy  # noqa: PLC0415  (the real rules)
        finally:
            sys.path.remove(str(GUARD_DIR))
        for cmd in ("pip install numpy pandas\n", "apt-get update && apt-get install -y curl\n",
                    "git clone https://github.com/harbor-framework/harbor.git\n", "python3 -c 'print(1)'\n",
                    "curl -LsSf https://astral.sh/uv/install.sh | sh\n", "make -j4 && ./run_tests.sh\n"):
            self.assertIsNone(policy.scan_executable_text(cmd), cmd)

    def test_pass_through(self) -> None:
        FakeUpstream.received.clear()
        status, data = self.request(extra_param=1)
        self.assertEqual(status, 200)
        self.assertEqual(data, terminus_reply("ls -la\n"))
        self.assertEqual(FakeUpstream.received[-1], {"model": "LAR-27B", "messages": [{"role": "user",
                         "content": "hi"}], "temperature": 1.0, "extra_param": 1})

    def test_policy_refusal_matches_agent_contract(self) -> None:
        for sentinel, kind in (("PSWF_SENTINEL_A", "tor"), ("PSWF_SENTINEL_B", "scan")):
            status, data = self.request(f"echo {sentinel}\n")
            self.assertEqual(status, 403)
            self.assertEqual(data["error"]["type"], "guard_policy_violation")
            self.assertEqual(data["error"]["message"], f"model command batch refused before execution ({kind})")
        # the exact regex the PSWF agent's _install_policy_stop() matches
        agent_src = AGENT.read_text()
        self.assertIn(r'"model command batch refused before execution \(([a-z][a-z0-9_-]*)\)"', agent_src)
        self.assertRegex(data["error"]["message"], r"model command batch refused before execution \(([a-z][a-z0-9_-]*)\)")

    def test_nul_byte_is_unparseable_not_violation(self) -> None:
        status, data = self.request("ls\x00\n")
        self.assertEqual(status, 403)
        self.assertEqual(data["error"]["message"], "model command batch refused before execution (unparseable_command)")

    def test_upstream_error_passes_through(self) -> None:
        status, data = self.request(status=400)
        self.assertEqual((status, data), (400, {"object": "error", "message": "too long"}))

    def test_streaming_refused(self) -> None:
        status, data = self.request(stream=True)
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "guard_streaming_forbidden")

    def test_records_never_hold_contents(self) -> None:
        self.request()
        rows = [json.loads(line) for line in (self.tmp / "rec" / "calls.jsonl").read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all({"sent", "status", "usage", "session"} <= set(row) for row in rows))
        self.assertFalse(any('"hi"' in json.dumps(row) or "reasoning_content" in json.dumps(row) for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Declare a fail-closed SEV-2 sync before each Harbor Docker start command."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

MARKER = "# Q38_PRECONTAINER_ALL_FEEDS_V1"
ANCHOR = '        """Run a docker compose command and return the result."""\n'
BLOCK = '''        # Q38_PRECONTAINER_ALL_FEEDS_V1
        # This hook is immediately before Docker can create/start task services.
        # It runs for each agent and separate verifier environment, including retries.
        if command and command[0] in {"up", "start", "run", "create", "restart"}:
            hook = os.environ.get("Q38_PRECONTAINER_HOOK", "")
            if os.environ.get("Q38_RUN_CONFIG") or hook:
                if not hook or not Path(hook).is_file():
                    raise RuntimeError("Q38 pre-container safety hook is missing")
                gate = await asyncio.create_subprocess_exec(
                    "timeout", "--kill-after=5", "240", "bash", hook, str(self.session_id),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    gate_output, _ = await gate.communicate()
                except BaseException:
                    if gate.returncode is None:
                        gate.terminate()
                        try:
                            await asyncio.wait_for(gate.wait(), timeout=10)
                        except asyncio.TimeoutError:
                            gate.kill()
                            await gate.wait()
                    raise
                if gate.returncode != 0:
                    raise RuntimeError(
                        "Q38 pre-container safety sync refused container startup: "
                        + gate_output.decode("utf-8", errors="replace")[-3000:]
                    )
                self.logger.info("Q38_PRECONTAINER_SYNCED %s", self.session_id)
'''


def transform(source: str) -> tuple[str, bool]:
    if MARKER in source:
        if source.count(BLOCK) != 1:
            raise ValueError("pre-container hook marker exists with different bytes")
        return source, False
    if source.count(ANCHOR) != 1:
        raise ValueError("Docker compose hook source shape drift")
    result = source.replace(ANCHOR, ANCHOR + BLOCK, 1)
    ast.parse(result)
    return result, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    before = args.target.read_text()
    after, changed = transform(before)
    if changed:
        fd, name = tempfile.mkstemp(dir=args.target.parent)
        with os.fdopen(fd, "w") as f:
            f.write(after)
        os.chmod(name, args.target.stat().st_mode)
        os.replace(name, args.target)
    doc = {"ok": True, "schema": "qwen38.harbor-precontainer-patch.v1",
           "changed": changed, "target": str(args.target),
           "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
           "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
           "per_container_all_feeds_sync": True}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

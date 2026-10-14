#!/usr/bin/env python3
"""Preserve container cleanup while omitting global image-store removal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

OLD = '["down", "--rmi", "local", "--volumes", "--remove-orphans"]'
NEW = '["down", "--volumes", "--remove-orphans"]'


def patch(target: Path) -> dict[str, object]:
    before = target.read_text(encoding="utf-8")
    old_count = before.count(OLD)
    new_count = before.count(NEW)
    if old_count == 1 and new_count == 0:
        after = before.replace(OLD, NEW)
        changed = True
    elif old_count == 0 and new_count == 1:
        after = before
        changed = False
    else:
        raise RuntimeError(
            f"unexpected Harbor cleanup source shape: old={old_count} new={new_count}"
        )
    if changed:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            handle.write(after)
            temporary = Path(handle.name)
        os.chmod(temporary, target.stat().st_mode)
        temporary.replace(target)
    return {
        "schema": "qwen38.harbor-cleanup-patch.v1",
        "target": str(target),
        "changed": changed,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "removed_args": ["--rmi", "local"],
        "preserved_args": ["down", "--volumes", "--remove-orphans"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = patch(args.target.resolve())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


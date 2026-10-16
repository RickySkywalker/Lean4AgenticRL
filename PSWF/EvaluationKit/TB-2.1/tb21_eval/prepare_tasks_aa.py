#!/usr/bin/env python3
"""Build an immutable Terminal-Bench 2.1 tree with a configurable agent timeout floor.

The source must be the ``tasks`` directory from the pinned official checkout.
The source is never edited.  A derived copy is made and the *agent* timeout in
each task is raised to 20,000 seconds when it is lower.  No verifier timeout or
other task input is changed.

The JSON manifest is the audit boundary for the derived dataset: it records the
Git commit, every source/derived file digest, per-task tree digests, and every
timeout decision.  The command fails closed on a wrong/dirty checkout, a task
count other than 89, a missing timeout, or any unexpected copied-file change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "qwen38.tb21.aa-task-tree.v1"
EXPECTED_REPOSITORY = "harbor-framework/terminal-bench-2-1"
EXPECTED_COMMIT = "7131e4375048a0e408a8fb404b5f499d726b695b"
EXPECTED_TASKS = 89
AA_AGENT_TIMEOUT_FLOOR_SEC = 20000.0

AGENT_SECTION_RE = re.compile(
    r"(?ms)^\[agent\][ \t]*(?:\r?\n|$)(?P<body>.*?)(?=^\[|\Z)"
)
NUMBER = r"[+]?(?:\d(?:_?\d)*(?:\.\d(?:_?\d)*)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?"
TIMEOUT_LINE_RE = re.compile(
    rf"(?m)^(?P<prefix>[ \t]*timeout_sec[ \t]*=[ \t]*)(?P<value>{NUMBER})(?P<suffix>[ \t]*(?:\#.*)?$)"
)


class PreparationError(RuntimeError):
    """A fail-closed task-tree validation error."""


@dataclass(frozen=True)
class RepositoryState:
    root: Path
    commit: str
    dirty_entries: tuple[str, ...]


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise PreparationError(f"git {' '.join(args)} failed: {detail.strip()}") from exc
    return proc.stdout.strip()


def inspect_repository(tasks_dir: Path) -> RepositoryState:
    """Resolve and audit the Git checkout containing ``tasks_dir``."""

    try:
        root = Path(_git(tasks_dir, "rev-parse", "--show-toplevel")).resolve()
    except PreparationError as exc:
        raise PreparationError(f"source is not inside the official Git checkout: {tasks_dir}") from exc

    expected_tasks_dir = (root / "tasks").resolve()
    if tasks_dir.resolve() != expected_tasks_dir:
        raise PreparationError(
            f"source must be the checkout's tasks directory ({expected_tasks_dir}), got {tasks_dir.resolve()}"
        )

    commit = _git(root, "rev-parse", "HEAD")
    rel_tasks = tasks_dir.resolve().relative_to(root).as_posix()
    dirty_text = _git(root, "status", "--porcelain", "--untracked-files=all", "--", rel_tasks)
    dirty = tuple(line for line in dirty_text.splitlines() if line.strip())
    return RepositoryState(root=root, commit=commit, dirty_entries=dirty)


def validate_repository(state: RepositoryState, expected_commit: str = EXPECTED_COMMIT) -> None:
    if state.commit != expected_commit:
        raise PreparationError(
            f"Terminal-Bench 2.1 commit mismatch: expected {expected_commit}, got {state.commit}"
        )
    if state.dirty_entries:
        preview = "; ".join(state.dirty_entries[:5])
        raise PreparationError(f"official tasks tree is dirty ({preview})")


# Bound the configurable timeout floor to a positive interval that includes the
# benchmark's task budgets. The manifest records this floor with the derived
# tree digest, so different timeout settings produce distinct artifacts.
AA_AGENT_TIMEOUT_FLOOR_MIN_SEC = 600.0
AA_AGENT_TIMEOUT_FLOOR_MAX_SEC = 86400.0


def _validate_floor(floor_sec: float) -> None:
    if not AA_AGENT_TIMEOUT_FLOOR_MIN_SEC <= floor_sec <= AA_AGENT_TIMEOUT_FLOOR_MAX_SEC:
        raise PreparationError(
            f"agent timeout floor must be within "
            f"[{AA_AGENT_TIMEOUT_FLOOR_MIN_SEC:g}, {AA_AGENT_TIMEOUT_FLOOR_MAX_SEC:g}] s, got {floor_sec:g}"
        )


def _number(value: str) -> float:
    return float(value.replace("_", ""))


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, ".15g")


def patch_task_toml(text: str, floor_sec: float = AA_AGENT_TIMEOUT_FLOOR_SEC) -> tuple[str, float, float]:
    """Raise only ``[agent].timeout_sec`` and return text, old, and new values."""

    _validate_floor(floor_sec)
    try:
        parsed = tomllib.loads(text)
        parsed_timeout = parsed["agent"]["timeout_sec"]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise PreparationError("task.toml lacks a valid [agent].timeout_sec") from exc
    if isinstance(parsed_timeout, bool) or not isinstance(parsed_timeout, (int, float)):
        raise PreparationError("[agent].timeout_sec must be numeric")

    sections = list(AGENT_SECTION_RE.finditer(text))
    if len(sections) != 1:
        raise PreparationError(f"expected exactly one [agent] section, found {len(sections)}")
    section = sections[0]
    timeout_lines = list(TIMEOUT_LINE_RE.finditer(section.group("body")))
    if len(timeout_lines) != 1:
        raise PreparationError(
            f"expected exactly one [agent].timeout_sec assignment, found {len(timeout_lines)}"
        )
    timeout_match = timeout_lines[0]
    old = _number(timeout_match.group("value"))
    if old != float(parsed_timeout):
        raise PreparationError("textual and parsed [agent].timeout_sec values disagree")
    if old <= 0:
        raise PreparationError("[agent].timeout_sec must be positive")
    new = max(old, floor_sec)
    if new == old:
        return text, old, new

    value_start = section.start("body") + timeout_match.start("value")
    value_end = section.start("body") + timeout_match.end("value")
    patched = text[:value_start] + _format_number(new) + text[value_end:]

    # Parse the result and prove that the intended semantic value landed.
    try:
        check = tomllib.loads(patched)["agent"]["timeout_sec"]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:  # pragma: no cover - defensive
        raise PreparationError("patched task.toml is invalid") from exc
    if float(check) != new:  # pragma: no cover - defensive
        raise PreparationError("patched timeout did not round-trip")
    return patched, old, new


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _path_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            records.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "size": len(target.encode("utf-8", errors="surrogateescape")),
                    "sha256": _sha256_bytes(target.encode("utf-8", errors="surrogateescape")),
                    "target": target,
                }
            )
        elif path.is_file():
            data = path.read_bytes()
            records.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": len(data),
                    "sha256": _sha256_bytes(data),
                }
            )
    return records


def _records_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = {
            key: record[key]
            for key in ("path", "kind", "size", "sha256", "target")
            if key in record
        }
        digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _task_digest(records: list[dict[str, Any]], task_name: str) -> str:
    prefix = f"{task_name}/"
    selected = []
    for record in records:
        if record["path"].startswith(prefix):
            relative_record = dict(record)
            relative_record["path"] = relative_record["path"][len(prefix) :]
            selected.append(relative_record)
    return _records_digest(selected)


def task_tomls(tasks_dir: Path, expected_tasks: int = EXPECTED_TASKS) -> list[Path]:
    if not (tasks_dir / "dataset.toml").is_file():
        raise PreparationError(f"{tasks_dir} has no dataset.toml")
    paths = sorted(
        (path for path in tasks_dir.glob("*/task.toml") if path.is_file()),
        key=lambda path: path.parent.name,
    )
    if len(paths) != expected_tasks:
        raise PreparationError(f"expected exactly {expected_tasks} tasks, found {len(paths)}")
    names = [path.parent.name for path in paths]
    if len(set(names)) != len(names):  # pragma: no cover - filesystem names are intrinsically unique
        raise PreparationError("duplicate task names")
    return paths


def _inside(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def prepare_task_tree(
    src: Path,
    dst: Path,
    manifest_path: Path,
    *,
    floor_sec: float = AA_AGENT_TIMEOUT_FLOOR_SEC,
    expected_commit: str = EXPECTED_COMMIT,
    expected_tasks: int = EXPECTED_TASKS,
    replace_existing: bool = False,
    repository_state: RepositoryState | None = None,
) -> dict[str, Any]:
    """Validate, copy, patch, verify, install, and describe the derived task tree."""

    src = src.resolve()
    dst = dst.resolve()
    manifest_path = manifest_path.resolve()
    _validate_floor(floor_sec)
    if src == dst or _inside(dst, src) or _inside(src, dst):
        raise PreparationError("source and destination must be disjoint")
    if _inside(manifest_path, src) or _inside(manifest_path, dst):
        raise PreparationError("manifest must be outside both source and derived task trees")

    state = repository_state or inspect_repository(src)
    validate_repository(state, expected_commit)
    source_tomls = task_tomls(src, expected_tasks)
    source_records = _path_records(src)
    source_records_after_validation = _path_records(src)
    if source_records != source_records_after_validation:  # pragma: no cover - concurrent mutation guard
        raise PreparationError("source tree changed while it was being hashed")

    dst.parent.mkdir(parents=True, exist_ok=True)
    marker = dst.with_name(f".{dst.name}.q38-aa-derived.json")
    if dst.exists():
        marker_ok = False
        if marker.is_file():
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
                marker_ok = (
                    marker_data.get("schema_version") == SCHEMA_VERSION
                    and marker_data.get("destination") == str(dst)
                )
            except (OSError, json.JSONDecodeError):
                marker_ok = False
        if not (replace_existing or marker_ok):
            raise PreparationError(
                f"refusing to replace unmarked destination {dst}; pass --replace-existing after auditing it"
            )

    staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}.staging-", dir=dst.parent))
    # mkdtemp creates the directory; copytree requires the leaf not to exist.
    staging.rmdir()
    try:
        shutil.copytree(src, staging, symlinks=True)
        timeout_rows: list[dict[str, Any]] = []
        expected_changes: set[str] = set()
        for source_toml in source_tomls:
            relative = source_toml.relative_to(src)
            derived_toml = staging / relative
            original_text = derived_toml.read_text(encoding="utf-8")
            patched_text, old, new = patch_task_toml(original_text, floor_sec)
            if patched_text != original_text:
                derived_toml.write_text(patched_text, encoding="utf-8")
                expected_changes.add(relative.as_posix())
            timeout_rows.append(
                {
                    "task": source_toml.parent.name,
                    "source_agent_timeout_sec": old,
                    "aa_agent_timeout_sec": new,
                    "action": "raised-to-floor" if old != new else "kept",
                }
            )

        source_records_final = _path_records(src)
        if source_records_final != source_records:
            raise PreparationError("source tree changed during preparation")
        derived_records = _path_records(staging)
        source_by_path = {record["path"]: record for record in source_records}
        derived_by_path = {record["path"]: record for record in derived_records}
        if source_by_path.keys() != derived_by_path.keys():
            raise PreparationError("derived copy added or lost files")
        actual_changes = {
            path
            for path in source_by_path
            if source_by_path[path] != derived_by_path[path]
        }
        if actual_changes != expected_changes:
            raise PreparationError(
                "unexpected derived-tree changes: "
                f"expected={sorted(expected_changes)} actual={sorted(actual_changes)}"
            )

        for row in timeout_rows:
            task_name = row["task"]
            task_path = f"{task_name}/task.toml"
            row["source_task_toml_sha256"] = source_by_path[task_path]["sha256"]
            row["derived_task_toml_sha256"] = derived_by_path[task_path]["sha256"]
            row["source_task_tree_sha256"] = _task_digest(source_records, task_name)
            row["derived_task_tree_sha256"] = _task_digest(derived_records, task_name)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "repository": EXPECTED_REPOSITORY,
            "repository_root": str(state.root),
            "expected_commit": expected_commit,
            "actual_commit": state.commit,
            "source": str(src),
            "destination": str(dst),
            "task_count": len(timeout_rows),
            "agent_timeout_floor_sec": floor_sec,
            "raised_count": sum(row["action"] == "raised-to-floor" for row in timeout_rows),
            "kept_count": sum(row["action"] == "kept" for row in timeout_rows),
            "source_tree_sha256": _records_digest(source_records),
            "derived_tree_sha256": _records_digest(derived_records),
            "tasks": timeout_rows,
            "files": [
                {
                    "path": path,
                    "source": source_by_path[path],
                    "derived": derived_by_path[path],
                    "changed": path in actual_changes,
                }
                for path in sorted(source_by_path)
            ],
        }

        backup: Path | None = None
        if dst.exists():
            backup = dst.with_name(f".{dst.name}.previous-{os.getpid()}")
            if backup.exists():  # pragma: no cover - extremely unlikely stale PID collision
                raise PreparationError(f"backup path already exists: {backup}")
            dst.replace(backup)
        try:
            staging.replace(dst)
        except Exception:
            if backup is not None and backup.exists() and not dst.exists():
                backup.replace(dst)
            raise
        if backup is not None:
            shutil.rmtree(backup)

        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(
            marker,
            {
                "schema_version": SCHEMA_VERSION,
                "destination": str(dst),
                "manifest": str(manifest_path),
                "derived_tree_sha256": manifest["derived_tree_sha256"],
            },
        )
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="pinned checkout's tasks/ directory")
    parser.add_argument("--dst", required=True, type=Path, help="derived AA task-tree destination")
    parser.add_argument("--manifest", required=True, type=Path, help="output JSON audit manifest")
    parser.add_argument("--floor-sec", type=float, default=AA_AGENT_TIMEOUT_FLOOR_SEC)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace an existing unmarked destination (marked outputs are safely refreshable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_task_tree(
            args.src,
            args.dst,
            args.manifest,
            floor_sec=args.floor_sec,
            replace_existing=args.replace_existing,
        )
    except (PreparationError, OSError) as exc:
        print(f"prepare_tasks_aa: error: {exc}", file=sys.stderr)
        return 2
    print(
        "[prepare-aa] "
        f"commit={manifest['actual_commit']} tasks={manifest['task_count']} "
        f"raised={manifest['raised_count']} kept={manifest['kept_count']} "
        f"sha256={manifest['derived_tree_sha256']} -> {args.dst}"
    )
    print(f"[prepare-aa] manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

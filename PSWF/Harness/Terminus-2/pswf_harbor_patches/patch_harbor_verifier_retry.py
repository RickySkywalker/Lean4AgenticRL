#!/usr/bin/env python3
"""Add fail-closed verifier dependency retries to the pinned Harbor build.

The agent has already reached its terminal state when ``Verifier.verify`` is
called.  This patch therefore retries only the official test script, inside the
same verifier environment.  A retry is allowed only when the just-completed
attempt has no valid, non-empty CTRF report *and* its pre-pytest stdout contains
an explicit dependency setup failure.  The existing trial-level
``asyncio.wait_for`` remains outside ``Verifier.verify``, so all attempts share
one verifier timeout budget. A deterministic, capped backoff gives transient
package-CDN failures time to clear while successful downloads remain cached in
the unchanged verifier environment.

Every completed attempt atomically preserves stable stdout, CTRF, and reward
bytes in separate verifier artifacts.  The normal stdout artifact is a
deterministic merge and a JSON lineage binds every path, digest, parsed value,
and retry decision. Reward and live CTRF outputs are cleared between attempts,
while snapshots remain immutable, so later results cannot inherit or erase
earlier evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

MAX_ATTEMPTS = 8
RETRY_BACKOFF_SECONDS = (1, 2, 4, 8, 16, 30, 30)
MAX_STDOUT_SCAN_BYTES = 2 * 1024 * 1024
MAX_ATTEMPT_STDOUT_BYTES = 16 * 1024 * 1024
MAX_CANONICAL_STDOUT_BYTES = MAX_ATTEMPTS * MAX_ATTEMPT_STDOUT_BYTES + 64 * 1024
CTRF_SNAPSHOT_WAIT_SECONDS = 3.0
CTRF_SNAPSHOT_POLL_SECONDS = 0.1
CTRF_SNAPSHOT_STABLE_SECONDS = 0.5
MAX_CTRF_BYTES = 16 * 1024 * 1024
MAX_REWARD_BYTES = 1024 * 1024
PERMISSION_HANDOFF_TIMEOUT_SECONDS = 30.0
LINEAGE_NAME = "verifier-retry.json"
LINEAGE_SCHEMA = "qwen38.harbor-verifier-dependency-retry.v3.1"
if len(RETRY_BACKOFF_SECONDS) != MAX_ATTEMPTS - 1:
    raise RuntimeError("retry backoff schedule must cover every possible retry")
PYTEST_START_MARKERS = (
    "test session starts",
    "=================================== FAILURES ===================================",
    "==================================== ERRORS ====================================",
)
DEPENDENCY_TOOL_CONTEXT_PATTERN = (
    r"(?:^|[^a-z0-9])(?:uvx?|pip3?|apt(?:-get)?)(?:[^a-z0-9]|$)"
)
TRANSIENT_RETRY_REASON_CODES = (
    "dependency_http_5xx",
    "dependency_http_429",
    "dependency_dns_temporary",
    "dependency_connection_transient",
    "dependency_tls_handshake_transient",
)
DEPENDENCY_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "dependency_fetch_error",
        r"(?:failed|unable|could not) to (?:fetch|download)|"
        r"request failed after \d+ retr(?:y|ies)",
    ),
    (
        "dependency_http_5xx",
        r"http(?: status)?(?: server)? error[^\n]{0,100}\b5\d\d\b|"
        r"\b5\d\d\s+(?:bad gateway|service unavailable|gateway timeout|"
        r"internal server error)\b",
    ),
    (
        "dependency_http_429",
        r"http(?: status)?(?: client)? error[^\n]{0,100}\b429\b|"
        r"\b429\s+(?:too many requests|rate limit(?:ed)?)\b",
    ),
    (
        "dependency_dns_temporary",
        r"temporary failure (?:in name resolution|resolving)|"
        r"(?:getaddrinfo[^\n]{0,80})?\beai_again\b",
    ),
    (
        "dependency_dns_error",
        r"temporary failure (?:in name resolution|resolving)|"
        r"could not resolve (?:host|hostname)|name or service not known|"
        r"nodename nor servname provided|getaddrinfo failed",
    ),
    (
        "dependency_tls_error",
        r"certificate verify failed|ssl:\s*certificate_verify_failed|"
        r"tls handshake (?:failed|error)|unable to get local issuer certificate|"
        r"certificate has expired",
    ),
    (
        "dependency_connection_transient",
        r"connection (?:timed out|timeout|reset(?: by peer)?)|"
        r"connect(?:ion)? timeout|read timed out|operation timed out",
    ),
    (
        "dependency_tls_handshake_transient",
        r"(?:tls|ssl) handshake (?:failed|error|timed out|timeout)|"
        r"connection reset[^\n]{0,80}(?:tls|ssl) handshake",
    ),
    (
        "dependency_apt_error",
        r"(?:^|\n)\s*e:\s*(?:failed to fetch|unable to locate package|"
        r"the repository .+ does not have a release file)|"
        r"some index files failed to download",
    ),
    (
        "dependency_pip_uv_error",
        r"(?:pip|uv|uvx)[^\n]{0,120}(?:subprocess-exited-with-error|"
        r"no matching distribution|could not find a version|"
        r"failed to (?:fetch|download|build|resolve)|resolution failed|"
        r"no solution found when resolving dependencies|"
        r"could not install packages due to an oserror)|"
        r"error:\s*(?:failed to (?:fetch|download|build|resolve)|"
        r"no solution found when resolving dependencies|"
        r"could not find a version that satisfies|"
        r"no matching distribution found|"
        r"could not install packages due to an oserror|"
        r"failed to build installable wheels)",
    ),
)

POLICY: dict[str, Any] = {
    "max_attempts": MAX_ATTEMPTS,
    "retry_backoff_seconds": list(RETRY_BACKOFF_SECONDS),
    "max_cumulative_backoff_seconds": sum(RETRY_BACKOFF_SECONDS),
    "stdout_scan_max_bytes": MAX_STDOUT_SCAN_BYTES,
    "attempt_stdout_max_bytes": MAX_ATTEMPT_STDOUT_BYTES,
    "canonical_stdout_max_bytes": MAX_CANONICAL_STDOUT_BYTES,
    "pytest_marker_scan_max_bytes": MAX_ATTEMPT_STDOUT_BYTES,
    "ctrf_snapshot_wait_seconds": CTRF_SNAPSHOT_WAIT_SECONDS,
    "ctrf_snapshot_poll_seconds": CTRF_SNAPSHOT_POLL_SECONDS,
    "ctrf_snapshot_stable_seconds": CTRF_SNAPSHOT_STABLE_SECONDS,
    "max_ctrf_bytes": MAX_CTRF_BYTES,
    "max_reward_bytes": MAX_REWARD_BYTES,
    "permission_handoff": "environment.prepare_logs_for_host",
    "permission_handoff_timeout_seconds": PERMISSION_HANDOFF_TIMEOUT_SECONDS,
    "permission_handoff_per_attempt": True,
    "retry_gate": [
        "invalid_or_empty_ctrf",
        "pre_pytest_apt_pip_uv_context",
        "explicit_transient_transport_signal",
        "pytest_not_started",
    ],
    "dependency_reason_codes": [code for code, _ in DEPENDENCY_FAILURE_PATTERNS],
    "transient_retry_reason_codes": list(TRANSIENT_RETRY_REASON_CODES),
    "pytest_start_markers": list(PYTEST_START_MARKERS),
    "same_verifier_environment": True,
    "timeout_budget": "one existing outer verifier timeout spans all attempts",
    "preserve_attempt_ctrf_bytes": True,
    "preserve_attempt_reward_bytes": True,
    "preserve_attempt_stdout_bytes": True,
    "joint_stdout_ctrf_stability": True,
    "lineage_schema": LINEAGE_SCHEMA,
}

_DEPENDENCY_TOOL_CONTEXT = re.compile(DEPENDENCY_TOOL_CONTEXT_PATTERN, re.IGNORECASE)
_DEPENDENCY_FAILURE_RULES = tuple(
    (code, re.compile(pattern, re.IGNORECASE))
    for code, pattern in DEPENDENCY_FAILURE_PATTERNS
)


def ctrf_state(value: object) -> str:
    """Classify already-decoded CTRF with the production retry gate."""

    if not isinstance(value, Mapping):
        return "invalid_shape"
    results = value.get("results")
    tests = results.get("tests") if isinstance(results, Mapping) else None
    if not isinstance(tests, list) or any(
        not isinstance(test, Mapping) for test in tests
    ):
        return "invalid_shape"
    return "valid_nonempty" if tests else "empty"


def dependency_setup_reasons(text: str) -> list[str]:
    """Return explicit dependency failure codes from the pre-pytest prefix."""

    lowered = text.lower()
    cut = len(text)
    for marker in PYTEST_START_MARKERS:
        index = lowered.find(marker.lower())
        if index >= 0:
            cut = min(cut, index)
    setup = text[:cut]
    if not setup:
        return []
    if not _DEPENDENCY_TOOL_CONTEXT.search(setup):
        return []
    return [
        code for code, pattern in _DEPENDENCY_FAILURE_RULES if pattern.search(setup)
    ]


def pytest_started(text: str) -> bool:
    """Return whether any pytest start marker occurs in bounded sealed stdout."""

    lowered = text.lower()
    return any(marker.lower() in lowered for marker in PYTEST_START_MARKERS)


def retry_decision(
    ctrf_value: object, stdout_text: str, attempt: int
) -> dict[str, object]:
    """Expose the immutable production gate for focused regression tests."""

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    state = ctrf_state(ctrf_value)
    reasons = dependency_setup_reasons(stdout_text)
    started = pytest_started(stdout_text)
    transient_reasons = [
        code for code in reasons if code in TRANSIENT_RETRY_REASON_CODES
    ]
    should_retry = bool(
        state != "valid_nonempty"
        and transient_reasons
        and not started
        and attempt < MAX_ATTEMPTS
    )
    return {
        "ctrf_state": state,
        "dependency_reason_codes": reasons,
        "transient_retry_reason_codes": transient_reasons,
        "pytest_started": started,
        "retry": should_retry,
        "retry_backoff_seconds": (
            RETRY_BACKOFF_SECONDS[attempt - 1] if should_retry else None
        ),
    }


IMPORT_OLD = """from typing import override
import json
import logging
"""
IMPORT_NEW = """from typing import override
import asyncio
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
"""


def _render_helpers() -> str:
    rules = "\n".join(
        f"    ({code!r}, re.compile({pattern!r}, re.IGNORECASE)),"
        for code, pattern in DEPENDENCY_FAILURE_PATTERNS
    )
    template = """_Q38_VERIFIER_RETRY_MAX_ATTEMPTS = __MAX_ATTEMPTS__
_Q38_VERIFIER_RETRY_BACKOFF_SECONDS = __BACKOFF__
_Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES = __MAX_STDOUT__
_Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES = __MAX_ATTEMPT_STDOUT__
_Q38_VERIFIER_RETRY_MAX_CANONICAL_STDOUT_BYTES = __MAX_CANONICAL_STDOUT__
_Q38_VERIFIER_RETRY_CTRF_WAIT_SECONDS = __CTRF_WAIT__
_Q38_VERIFIER_RETRY_POLL_SECONDS = __POLL__
_Q38_VERIFIER_RETRY_STABLE_SECONDS = __STABLE__
_Q38_VERIFIER_RETRY_MAX_CTRF_BYTES = __MAX_CTRF__
_Q38_VERIFIER_RETRY_MAX_REWARD_BYTES = __MAX_REWARD__
_Q38_VERIFIER_RETRY_PERMISSION_HANDOFF_TIMEOUT_SECONDS = __HANDOFF_TIMEOUT__
_Q38_VERIFIER_RETRY_LINEAGE_SCHEMA = __LINEAGE_SCHEMA__
_Q38_VERIFIER_RETRY_LINEAGE_NAME = __LINEAGE_NAME__
_Q38_PYTEST_START_MARKERS = __PYTEST_MARKERS__
_Q38_DEPENDENCY_TOOL_CONTEXT = re.compile(__TOOL_PATTERN__, re.IGNORECASE)
_Q38_TRANSIENT_RETRY_REASON_CODES = __TRANSIENT_CODES__
_Q38_DEPENDENCY_FAILURE_RULES = (
__RULES__
)


def _q38_ctrf_state(value: object) -> str:
    if not isinstance(value, dict):
        return "invalid_shape"
    results = value.get("results")
    tests = results.get("tests") if isinstance(results, dict) else None
    if not isinstance(tests, list) or any(not isinstance(test, dict) for test in tests):
        return "invalid_shape"
    return "valid_nonempty" if tests else "empty"


def _q38_regular_file_observation(
    path: Path, max_bytes: int, *, parse_ctrf: bool = False
) -> tuple[dict[str, object], bytes | None]:
    record: dict[str, object] = {
        "state": "missing",
        "sha256": None,
        "bytes": None,
        "read_error_type": "FileNotFoundError",
        "file_dev": None,
        "file_ino": None,
        "file_mtime_ns": None,
        "file_ctime_ns": None,
    }
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            record["state"] = "unsafe_file"
            record["read_error_type"] = "SymlinkArtifactError"
        elif isinstance(exc, FileNotFoundError):
            record["state"] = "missing"
            record["read_error_type"] = "FileNotFoundError"
        else:
            record["state"] = "read_error"
            record["read_error_type"] = type(exc).__name__
        return record, None
    try:
        before = os.fstat(fd)
        record.update(
            {
                "file_dev": before.st_dev,
                "file_ino": before.st_ino,
                "file_mtime_ns": before.st_mtime_ns,
                "file_ctime_ns": before.st_ctime_ns,
            }
        )
        if not stat.S_ISREG(before.st_mode):
            record.update(state="unsafe_file", read_error_type="NonRegularArtifactError")
            return record, None
        if before.st_size > max_bytes:
            record.update(
                state="oversize",
                bytes=before.st_size,
                read_error_type="ArtifactTooLargeError",
            )
            return record, None
        chunks = bytearray()
        while len(chunks) <= max_bytes:
            block = os.read(fd, min(1024 * 1024, max_bytes + 1 - len(chunks)))
            if not block:
                break
            chunks.extend(block)
        after = os.fstat(fd)
        data = bytes(chunks)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            identity_before != identity_after
            or len(data) != before.st_size
            or len(data) > max_bytes
        ):
            record.update(
                state="unstable_read",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                read_error_type="ArtifactChangedDuringRead",
            )
            return record, None
        record.update(
            state="regular",
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            read_error_type=None,
        )
    except OSError as exc:
        record.update(state="read_error", read_error_type=type(exc).__name__)
        return record, None
    finally:
        os.close(fd)
    if not parse_ctrf:
        return record, data
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        record.update(state="malformed", read_error_type=type(exc).__name__)
        return record, data
    record["state"] = _q38_ctrf_state(value)
    return record, data


def _q38_observation_identity(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        record.get(key)
        for key in (
            "state",
            "sha256",
            "bytes",
            "read_error_type",
            "file_dev",
            "file_ino",
            "file_mtime_ns",
            "file_ctime_ns",
        )
    )


async def _q38_stable_file_snapshot(
    path: Path,
    max_bytes: int,
    *,
    parse_ctrf: bool = False,
) -> tuple[dict[str, object], bytes | None]:
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + _Q38_VERIFIER_RETRY_CTRF_WAIT_SECONDS
    record, data = _q38_regular_file_observation(
        path, max_bytes, parse_ctrf=parse_ctrf
    )
    identity = _q38_observation_identity(record)
    stable_since = started
    samples = 1
    while True:
        now = loop.time()
        stable = now - stable_since >= _Q38_VERIFIER_RETRY_STABLE_SECONDS
        may_finish_early = not parse_ctrf or record.get("state") == "valid_nonempty"
        if stable and may_finish_early and data is not None:
            record.update(
                snapshot_stable=True,
                snapshot_samples=samples,
                snapshot_wait_exhausted=False,
            )
            return record, data
        if now >= deadline:
            record.update(
                snapshot_stable=stable,
                snapshot_samples=samples,
                snapshot_wait_exhausted=True,
            )
            return record, data
        await asyncio.sleep(
            min(_Q38_VERIFIER_RETRY_POLL_SECONDS, max(deadline - now, 0.0))
        )
        now = loop.time()
        next_record, next_data = _q38_regular_file_observation(
            path, max_bytes, parse_ctrf=parse_ctrf
        )
        next_identity = _q38_observation_identity(next_record)
        samples += 1
        if next_identity != identity or next_data != data:
            stable_since = now
        record, data, identity = next_record, next_data, next_identity


async def _q38_stable_attempt_snapshots(
    stdout_path: Path,
    ctrf_path: Path,
) -> tuple[
    dict[str, object], bytes | None, dict[str, object], bytes | None
]:
    # stdout supplies the retry signal while CTRF supplies the retry gate. They
    # must describe one concurrently settled terminal observation; stabilizing
    # them serially would let a detached stdout writer append during CTRF wait.
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + _Q38_VERIFIER_RETRY_CTRF_WAIT_SECONDS
    stdout_record, stdout_data = _q38_regular_file_observation(
        stdout_path, _Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES
    )
    ctrf_record, ctrf_data = _q38_regular_file_observation(
        ctrf_path, _Q38_VERIFIER_RETRY_MAX_CTRF_BYTES, parse_ctrf=True
    )
    stdout_identity = _q38_observation_identity(stdout_record)
    ctrf_identity = _q38_observation_identity(ctrf_record)
    stdout_stable_since = started
    ctrf_stable_since = started
    samples = 1
    while True:
        now = loop.time()
        stdout_stable = bool(
            stdout_data is not None
            and now - stdout_stable_since
            >= _Q38_VERIFIER_RETRY_STABLE_SECONDS
        )
        ctrf_stable = bool(
            now - ctrf_stable_since >= _Q38_VERIFIER_RETRY_STABLE_SECONDS
        )
        if (
            stdout_stable
            and ctrf_stable
            and ctrf_record.get("state") == "valid_nonempty"
        ):
            stdout_record.update(
                snapshot_stable=True,
                snapshot_samples=samples,
                snapshot_wait_exhausted=False,
            )
            ctrf_record.update(
                snapshot_stable=True,
                snapshot_samples=samples,
                snapshot_wait_exhausted=False,
            )
            return stdout_record, stdout_data, ctrf_record, ctrf_data
        if now >= deadline:
            stdout_record.update(
                snapshot_stable=stdout_stable,
                snapshot_samples=samples,
                snapshot_wait_exhausted=True,
            )
            ctrf_record.update(
                snapshot_stable=ctrf_stable,
                snapshot_samples=samples,
                snapshot_wait_exhausted=True,
            )
            return stdout_record, stdout_data, ctrf_record, ctrf_data
        await asyncio.sleep(
            min(_Q38_VERIFIER_RETRY_POLL_SECONDS, max(deadline - now, 0.0))
        )
        now = loop.time()
        next_stdout_record, next_stdout_data = _q38_regular_file_observation(
            stdout_path, _Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES
        )
        next_ctrf_record, next_ctrf_data = _q38_regular_file_observation(
            ctrf_path, _Q38_VERIFIER_RETRY_MAX_CTRF_BYTES, parse_ctrf=True
        )
        next_stdout_identity = _q38_observation_identity(next_stdout_record)
        next_ctrf_identity = _q38_observation_identity(next_ctrf_record)
        samples += 1
        if (
            next_stdout_identity != stdout_identity
            or next_stdout_data != stdout_data
        ):
            stdout_stable_since = now
        if next_ctrf_identity != ctrf_identity or next_ctrf_data != ctrf_data:
            ctrf_stable_since = now
        stdout_record = next_stdout_record
        stdout_data = next_stdout_data
        stdout_identity = next_stdout_identity
        ctrf_record = next_ctrf_record
        ctrf_data = next_ctrf_data
        ctrf_identity = next_ctrf_identity


def _q38_write_atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _q38_write_lineage(path: Path, value: dict[str, object]) -> None:
    _q38_write_atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\\n").encode("utf-8"),
    )


def _q38_preserve_snapshot(
    destination: Path,
    record: dict[str, object],
    data: bytes | None,
    trial_dir: Path,
) -> dict[str, object]:
    evidence = dict(record)
    if data is None:
        evidence["path"] = None
        return evidence
    _q38_write_atomic_bytes(destination, data)
    evidence["path"] = str(destination.relative_to(trial_dir))
    return evidence


def _q38_reward_value(kind: str, data: bytes) -> tuple[object | None, str | None]:
    try:
        text = data.decode("utf-8")
        value = json.loads(text) if kind == "json" else {"reward": float(text)}
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, type(exc).__name__
    return value, None


def _q38_dependency_setup_reasons(text: str) -> list[str]:
    lowered = text.lower()
    cut = len(text)
    for marker in _Q38_PYTEST_START_MARKERS:
        index = lowered.find(marker.lower())
        if index >= 0:
            cut = min(cut, index)
    setup = text[:cut]
    if not setup:
        return []
    if not _Q38_DEPENDENCY_TOOL_CONTEXT.search(setup):
        return []
    return [
        code
        for code, pattern in _Q38_DEPENDENCY_FAILURE_RULES
        if pattern.search(setup)
    ]


def _q38_pytest_started(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _Q38_PYTEST_START_MARKERS)


async def _q38_dependency_retry_backoff(seconds: int) -> None:
    await asyncio.sleep(seconds)


def _q38_stdout_evidence(path: Path) -> tuple[str | None, int, bool, str]:
    if not path.is_file():
        return None, 0, False, ""
    digest = hashlib.sha256()
    size = 0
    prefix = bytearray()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
            if len(prefix) < _Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES:
                remaining = _Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES - len(prefix)
                prefix.extend(block[:remaining])
    return (
        digest.hexdigest(),
        size,
        size > _Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES,
        bytes(prefix).decode("utf-8", errors="replace"),
    )


"""
    replacements = {
        "__MAX_ATTEMPTS__": str(MAX_ATTEMPTS),
        "__BACKOFF__": repr(RETRY_BACKOFF_SECONDS),
        "__MAX_STDOUT__": str(MAX_STDOUT_SCAN_BYTES),
        "__MAX_ATTEMPT_STDOUT__": str(MAX_ATTEMPT_STDOUT_BYTES),
        "__MAX_CANONICAL_STDOUT__": str(MAX_CANONICAL_STDOUT_BYTES),
        "__CTRF_WAIT__": repr(CTRF_SNAPSHOT_WAIT_SECONDS),
        "__POLL__": repr(CTRF_SNAPSHOT_POLL_SECONDS),
        "__STABLE__": repr(CTRF_SNAPSHOT_STABLE_SECONDS),
        "__MAX_CTRF__": str(MAX_CTRF_BYTES),
        "__MAX_REWARD__": str(MAX_REWARD_BYTES),
        "__HANDOFF_TIMEOUT__": repr(PERMISSION_HANDOFF_TIMEOUT_SECONDS),
        "__LINEAGE_SCHEMA__": repr(LINEAGE_SCHEMA),
        "__LINEAGE_NAME__": repr(LINEAGE_NAME),
        "__PYTEST_MARKERS__": repr(PYTEST_START_MARKERS),
        "__TOOL_PATTERN__": repr(DEPENDENCY_TOOL_CONTEXT_PATTERN),
        "__TRANSIENT_CODES__": repr(TRANSIENT_RETRY_REASON_CODES),
        "__RULES__": rules,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


HELPERS_NEW = _render_helpers()
CLASS_OLD = """class RewardFileEmptyError(Exception):
    pass


class Verifier(BaseVerifier):
"""
CLASS_NEW = """class RewardFileEmptyError(Exception):
    pass


""" + HELPERS_NEW + "class Verifier(BaseVerifier):\n"

EXECUTION_OLD = """        command = build_execution_command(
            test_script_path,
            stdout_path=test_stdout_path,
            task_os=self.environment.os,
        )

        if needs_chmod(test_script_path):
            await self.environment.exec(
                command=f"chmod +x {quote_shell_arg(test_script_path, self.environment.os)}",
                user="root",
            )

        # Runs as ``environment.default_user``, which the caller must set to the
        # effective verifier user (step-level override or task-level fallback).
        await self.environment.exec(
            command=command,
            env=env,
        )

        if not self.environment.capabilities.mounted:
            try:
                if self.include_logs or self.exclude_logs:
                    await self.environment.download_dir_filtered(
                        source_dir=str(env_paths.verifier_dir),
                        target_dir=self.trial_paths.verifier_dir,
                        include=self.include_logs or None,
                        exclude=self.exclude_logs or None,
                        protect=[
                            self.trial_paths.reward_text_path.name,
                            self.trial_paths.reward_json_path.name,
                        ],
                    )
                else:
                    await self.environment.download_dir(
                        source_dir=str(env_paths.verifier_dir),
                        target_dir=self.trial_paths.verifier_dir,
                    )
            except Exception as e:
                raise DownloadVerifierDirError(
                    "Failed to download verifier directory from environment"
                ) from e
"""

EXECUTION_NEW = """        if needs_chmod(test_script_path):
            await self.environment.exec(
                command=f"chmod +x {quote_shell_arg(test_script_path, self.environment.os)}",
                user="root",
            )

        ctrf_path = self.trial_paths.verifier_dir / "ctrf.json"
        lineage_path = self.trial_paths.verifier_dir / _Q38_VERIFIER_RETRY_LINEAGE_NAME
        attempt_stdout_paths = [
            self.trial_paths.verifier_dir / f"test-stdout.attempt-{attempt}.txt"
            for attempt in range(1, _Q38_VERIFIER_RETRY_MAX_ATTEMPTS + 1)
        ]
        attempt_ctrf_paths = [
            self.trial_paths.verifier_dir / f"ctrf.attempt-{attempt}.json"
            for attempt in range(1, _Q38_VERIFIER_RETRY_MAX_ATTEMPTS + 1)
        ]
        attempt_reward_text_paths = [
            self.trial_paths.verifier_dir / f"reward.attempt-{attempt}.txt"
            for attempt in range(1, _Q38_VERIFIER_RETRY_MAX_ATTEMPTS + 1)
        ]
        attempt_reward_json_paths = [
            self.trial_paths.verifier_dir / f"reward.attempt-{attempt}.json"
            for attempt in range(1, _Q38_VERIFIER_RETRY_MAX_ATTEMPTS + 1)
        ]

        # A regrade may reuse a verifier directory. Remove exactly this patch's
        # bookkeeping paths, including every bounded attempt snapshot.
        for stale_path in [
            self.trial_paths.test_stdout_path,
            lineage_path,
            *attempt_stdout_paths,
            *attempt_ctrf_paths,
            *attempt_reward_text_paths,
            *attempt_reward_json_paths,
        ]:
            stale_path.unlink(missing_ok=True)

        async def clear_official_outputs() -> None:
            # Snapshot files are deliberately excluded. Only mutable official
            # outputs are cleared before attempt one and between retries.
            if not self.environment.capabilities.mounted:
                remote_outputs = [
                    str(env_paths.reward_text_path),
                    str(env_paths.reward_json_path),
                    str(env_paths.verifier_dir / "ctrf.json"),
                ]
                if self.environment.os.value == "windows":
                    quoted_outputs = " ".join(
                        quote_shell_arg(path, self.environment.os)
                        for path in remote_outputs
                    )
                    cleanup_command = f"del /f /q {quoted_outputs} >nul 2>&1 & exit /b 0"
                else:
                    cleanup_command = "rm -f -- " + " ".join(
                        quote_shell_arg(path, self.environment.os)
                        for path in remote_outputs
                    )
                cleanup_result = await self.environment.exec(command=cleanup_command)
                if cleanup_result.return_code != 0:
                    raise RuntimeError(
                        "Failed to clear verifier outputs before dependency retry"
                    )
            for output_path in (
                self.trial_paths.reward_text_path,
                self.trial_paths.reward_json_path,
                ctrf_path,
            ):
                output_path.unlink(missing_ok=True)

        attempts = []
        attempt_stdout_payloads = []
        attempt_reward_payloads = []
        lineage = {}
        loop_complete = False

        def publish_lineage(
            terminal_state: str,
            *,
            complete: bool,
            evidence_complete: bool,
            extra: dict[str, object] | None = None,
        ) -> None:
            canonical_record, _ = _q38_regular_file_observation(
                self.trial_paths.test_stdout_path,
                _Q38_VERIFIER_RETRY_MAX_CANONICAL_STDOUT_BYTES,
            )
            canonical = {
                "path": str(
                    self.trial_paths.test_stdout_path.relative_to(
                        self.trial_paths.trial_dir
                    )
                ),
                "sha256": canonical_record.get("sha256"),
                "bytes": canonical_record.get("bytes"),
                "read_error_type": canonical_record.get("read_error_type"),
            }
            value: dict[str, object] = {
                "schema_version": _Q38_VERIFIER_RETRY_LINEAGE_SCHEMA,
                "policy": {
                    "max_attempts": _Q38_VERIFIER_RETRY_MAX_ATTEMPTS,
                    "retry_backoff_seconds": list(
                        _Q38_VERIFIER_RETRY_BACKOFF_SECONDS
                    ),
                    "max_cumulative_backoff_seconds": sum(
                        _Q38_VERIFIER_RETRY_BACKOFF_SECONDS
                    ),
                    "stdout_scan_max_bytes": _Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES,
                    "attempt_stdout_max_bytes": _Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES,
                    "canonical_stdout_max_bytes": _Q38_VERIFIER_RETRY_MAX_CANONICAL_STDOUT_BYTES,
                    "pytest_marker_scan_max_bytes": _Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES,
                    "ctrf_snapshot_wait_seconds": _Q38_VERIFIER_RETRY_CTRF_WAIT_SECONDS,
                    "ctrf_snapshot_poll_seconds": _Q38_VERIFIER_RETRY_POLL_SECONDS,
                    "ctrf_snapshot_stable_seconds": _Q38_VERIFIER_RETRY_STABLE_SECONDS,
                    "max_ctrf_bytes": _Q38_VERIFIER_RETRY_MAX_CTRF_BYTES,
                    "max_reward_bytes": _Q38_VERIFIER_RETRY_MAX_REWARD_BYTES,
                    "permission_handoff": "environment.prepare_logs_for_host",
                    "permission_handoff_timeout_seconds": _Q38_VERIFIER_RETRY_PERMISSION_HANDOFF_TIMEOUT_SECONDS,
                    "permission_handoff_per_attempt": True,
                    "retry_gate": [
                        "invalid_or_empty_ctrf",
                        "pre_pytest_apt_pip_uv_context",
                        "explicit_transient_transport_signal",
                        "pytest_not_started",
                    ],
                    "dependency_reason_codes": [
                        code for code, _ in _Q38_DEPENDENCY_FAILURE_RULES
                    ],
                    "transient_retry_reason_codes": list(
                        _Q38_TRANSIENT_RETRY_REASON_CODES
                    ),
                    "pytest_start_markers": list(_Q38_PYTEST_START_MARKERS),
                    "same_verifier_environment": True,
                    "timeout_budget": "one existing outer verifier timeout spans all attempts",
                    "preserve_attempt_ctrf_bytes": True,
                    "preserve_attempt_reward_bytes": True,
                    "preserve_attempt_stdout_bytes": True,
                    "joint_stdout_ctrf_stability": True,
                    "lineage_schema": _Q38_VERIFIER_RETRY_LINEAGE_SCHEMA,
                },
                "complete": complete,
                "evidence_complete": evidence_complete,
                "terminal_state": terminal_state,
                "terminal_reason": (
                    attempts[-1]["terminal_reason"] if attempts else None
                ),
                "retry_budget_exhausted": bool(
                    attempts
                    and attempts[-1]["terminal_reason"] == "retry_budget_exhausted"
                ),
                "attempt_count": len(attempts),
                "attempts": attempts,
                "canonical_stdout": canonical,
            }
            if extra:
                value.update(extra)
            lineage.clear()
            lineage.update(value)
            _q38_write_lineage(lineage_path, lineage)

        def preserve_reward_outputs(
            attempt_index: int,
        ) -> tuple[dict[str, object], dict[str, bytes | None]]:
            outputs: dict[str, object] = {}
            payloads: dict[str, bytes | None] = {}
            specifications = (
                (
                    "json",
                    self.trial_paths.reward_json_path,
                    attempt_reward_json_paths[attempt_index - 1],
                ),
                (
                    "text",
                    self.trial_paths.reward_text_path,
                    attempt_reward_text_paths[attempt_index - 1],
                ),
            )
            for kind, source, destination in specifications:
                record, data = _q38_regular_file_observation(
                    source, _Q38_VERIFIER_RETRY_MAX_REWARD_BYTES
                )
                evidence = _q38_preserve_snapshot(
                    destination, record, data, self.trial_paths.trial_dir
                )
                value, parse_error = (
                    _q38_reward_value(kind, data)
                    if data is not None
                    else (None, record.get("read_error_type"))
                )
                evidence["value"] = value
                evidence["parse_error_type"] = parse_error
                outputs[kind] = evidence
                payloads[kind] = data
            return outputs, payloads

        await clear_official_outputs()

        for attempt, attempt_stdout_path in enumerate(attempt_stdout_paths, start=1):
            env_attempt_stdout_path = str(
                env_paths.verifier_dir
                / attempt_stdout_path.relative_to(
                    self.trial_paths.verifier_dir
                ).as_posix()
            )
            command = build_execution_command(
                test_script_path,
                stdout_path=env_attempt_stdout_path,
                task_os=self.environment.os,
            )

            execution_error = None
            try:
                await self.environment.exec(command=command, env=env)
            except BaseException as e:
                execution_error = e

            if execution_error is None and not self.environment.capabilities.mounted:
                try:
                    if self.include_logs or self.exclude_logs:
                        await self.environment.download_dir_filtered(
                            source_dir=str(env_paths.verifier_dir),
                            target_dir=self.trial_paths.verifier_dir,
                            include=self.include_logs or None,
                            exclude=self.exclude_logs or None,
                            protect=[
                                self.trial_paths.reward_text_path.name,
                                self.trial_paths.reward_json_path.name,
                            ],
                        )
                    else:
                        await self.environment.download_dir(
                            source_dir=str(env_paths.verifier_dir),
                            target_dir=self.trial_paths.verifier_dir,
                        )
                except asyncio.CancelledError as e:
                    execution_error = e
                except Exception as e:
                    execution_error = DownloadVerifierDirError(
                        "Failed to download verifier directory from environment"
                    )
                    execution_error.__cause__ = e

            permission_handoff_error = None
            permission_handoff_required = bool(
                self.environment.capabilities.mounted
            )
            permission_handoff_completed = (
                False if permission_handoff_required else None
            )
            if execution_error is None and permission_handoff_required:
                try:
                    await asyncio.wait_for(
                        self.environment.prepare_logs_for_host(),
                        timeout=_Q38_VERIFIER_RETRY_PERMISSION_HANDOFF_TIMEOUT_SECONDS,
                    )
                    permission_handoff_completed = True
                except BaseException as e:
                    permission_handoff_error = e

            snapshot_error = None
            stdout_record: dict[str, object]
            stdout_data: bytes | None
            ctrf_record: dict[str, object]
            ctrf_data: bytes | None
            if execution_error is None and permission_handoff_error is None:
                try:
                    (
                        stdout_record,
                        stdout_data,
                        ctrf_record,
                        ctrf_data,
                    ) = await _q38_stable_attempt_snapshots(
                        attempt_stdout_path,
                        ctrf_path,
                    )
                except BaseException as e:
                    snapshot_error = e
                    stdout_record, stdout_data = _q38_regular_file_observation(
                        attempt_stdout_path,
                        _Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES,
                    )
                    ctrf_record, ctrf_data = _q38_regular_file_observation(
                        ctrf_path,
                        _Q38_VERIFIER_RETRY_MAX_CTRF_BYTES,
                        parse_ctrf=True,
                    )
                    for record in (stdout_record, ctrf_record):
                        record.update(
                            snapshot_stable=False,
                            snapshot_samples=None,
                            snapshot_wait_exhausted=False,
                        )
            else:
                stdout_record, stdout_data = _q38_regular_file_observation(
                    attempt_stdout_path,
                    _Q38_VERIFIER_RETRY_MAX_ATTEMPT_STDOUT_BYTES,
                )
                ctrf_record, ctrf_data = _q38_regular_file_observation(
                    ctrf_path,
                    _Q38_VERIFIER_RETRY_MAX_CTRF_BYTES,
                    parse_ctrf=True,
                )
                for record in (stdout_record, ctrf_record):
                    record.update(
                        snapshot_stable=False,
                        snapshot_samples=None,
                        snapshot_wait_exhausted=False,
                    )

            # Replace the stdout path with the stable bytes. A tardy writer can
            # keep its old fd, but cannot mutate the preserved pathname.
            if stdout_data is not None:
                _q38_write_atomic_bytes(attempt_stdout_path, stdout_data)
            stdout_evidence = dict(stdout_record)
            stdout_evidence["path"] = (
                str(attempt_stdout_path.relative_to(self.trial_paths.trial_dir))
                if stdout_data is not None
                else None
            )
            stdout_text = (
                stdout_data[:_Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES].decode(
                    "utf-8", errors="replace"
                )
                if stdout_data is not None
                else ""
            )
            pytest_started = bool(
                stdout_data is not None
                and _q38_pytest_started(
                    stdout_data.decode("utf-8", errors="replace")
                )
            )
            stdout_evidence["scan_truncated"] = bool(
                stdout_data is not None
                and len(stdout_data) > _Q38_VERIFIER_RETRY_MAX_STDOUT_BYTES
            )
            ctrf_evidence = _q38_preserve_snapshot(
                attempt_ctrf_paths[attempt - 1],
                ctrf_record,
                ctrf_data,
                self.trial_paths.trial_dir,
            )
            reward_outputs, reward_payloads = preserve_reward_outputs(attempt)

            state = str(ctrf_record.get("state"))
            reason_codes = _q38_dependency_setup_reasons(stdout_text)
            transient_reason_codes = [
                code
                for code in reason_codes
                if code in _Q38_TRANSIENT_RETRY_REASON_CODES
            ]
            stdout_stable = bool(
                stdout_record.get("snapshot_stable") and stdout_data is not None
            )
            ctrf_stable = bool(ctrf_record.get("snapshot_stable"))
            stdout_observation_complete = bool(
                stdout_stable
                and stdout_record.get("state") == "regular"
                and stdout_record.get("read_error_type") is None
            )
            ctrf_observation_complete = bool(
                ctrf_stable
                and (
                    ctrf_data is not None
                    or (
                        state == "missing"
                        and ctrf_record.get("read_error_type")
                        == "FileNotFoundError"
                    )
                )
            )
            retryable_ctrf_state = state in {
                "missing",
                "malformed",
                "invalid_shape",
                "empty",
            }
            retry = bool(
                execution_error is None
                and permission_handoff_error is None
                and snapshot_error is None
                and stdout_observation_complete
                and ctrf_observation_complete
                and retryable_ctrf_state
                and transient_reason_codes
                and not pytest_started
                and attempt < _Q38_VERIFIER_RETRY_MAX_ATTEMPTS
            )
            retry_backoff_seconds = (
                _Q38_VERIFIER_RETRY_BACKOFF_SECONDS[attempt - 1]
                if retry
                else None
            )
            if execution_error is not None:
                terminal_reason = "execution_error"
            elif permission_handoff_error is not None:
                terminal_reason = "permission_handoff_error"
            elif snapshot_error is not None:
                terminal_reason = "snapshot_wait_interrupted"
            elif not stdout_observation_complete:
                terminal_reason = "stdout_snapshot_unusable"
            elif not ctrf_observation_complete:
                terminal_reason = "ctrf_snapshot_unusable"
            elif retry:
                terminal_reason = "retry_scheduled"
            elif state == "valid_nonempty":
                terminal_reason = "valid_ctrf"
            elif pytest_started:
                terminal_reason = "pytest_started_no_retry"
            elif transient_reason_codes and attempt >= _Q38_VERIFIER_RETRY_MAX_ATTEMPTS:
                terminal_reason = "retry_budget_exhausted"
            elif reason_codes:
                terminal_reason = "non_retryable_dependency_failure"
            else:
                terminal_reason = "no_retryable_dependency_signal"
            attempt_evidence_complete = bool(
                execution_error is None
                and permission_handoff_error is None
                and snapshot_error is None
                and (
                    not permission_handoff_required
                    or permission_handoff_completed is True
                )
                and stdout_observation_complete
                and ctrf_observation_complete
            )
            evidence_error = None
            if (
                execution_error is None
                and permission_handoff_error is None
                and snapshot_error is None
                and not attempt_evidence_complete
            ):
                evidence_error = RuntimeError(
                    "Verifier artifacts remained unreadable or unstable after "
                    "prepare_logs_for_host"
                )
            attempts.append(
                {
                    "attempt": attempt,
                    "ctrf": ctrf_evidence,
                    "reward_outputs": reward_outputs,
                    "dependency_reason_codes": reason_codes,
                    "transient_retry_reason_codes": transient_reason_codes,
                    "pytest_started": pytest_started,
                    "retry_scheduled": retry,
                    "retry_backoff_seconds": retry_backoff_seconds,
                    "retry_backoff_completed": False if retry else None,
                    "retry_backoff_error_type": None,
                    "retry_backoff_error_message_sha256": None,
                    "terminal_reason": terminal_reason,
                    "stdout": stdout_evidence,
                    "evidence_complete": attempt_evidence_complete,
                    "execution_error_type": (
                        type(execution_error).__name__
                        if execution_error is not None
                        else None
                    ),
                    "execution_error_message_sha256": (
                        hashlib.sha256(
                            str(execution_error).encode("utf-8", errors="replace")
                        ).hexdigest()
                        if execution_error is not None
                        else None
                    ),
                    "permission_handoff_required": permission_handoff_required,
                    "permission_handoff_completed": permission_handoff_completed,
                    "permission_handoff_error_type": (
                        type(permission_handoff_error).__name__
                        if permission_handoff_error is not None
                        else None
                    ),
                    "permission_handoff_error_message_sha256": (
                        hashlib.sha256(
                            str(permission_handoff_error).encode(
                                "utf-8", errors="replace"
                            )
                        ).hexdigest()
                        if permission_handoff_error is not None
                        else None
                    ),
                    "snapshot_wait_error_type": (
                        type(snapshot_error).__name__
                        if snapshot_error is not None
                        else None
                    ),
                    "snapshot_wait_error_message_sha256": (
                        hashlib.sha256(
                            str(snapshot_error).encode("utf-8", errors="replace")
                        ).hexdigest()
                        if snapshot_error is not None
                        else None
                    ),
                    "evidence_error_type": (
                        type(evidence_error).__name__
                        if evidence_error is not None
                        else None
                    ),
                    "evidence_error_message_sha256": (
                        hashlib.sha256(
                            str(evidence_error).encode("utf-8", errors="replace")
                        ).hexdigest()
                        if evidence_error is not None
                        else None
                    ),
                }
            )
            attempt_stdout_payloads.append(stdout_data or b"")
            attempt_reward_payloads.append(reward_payloads)

            # Canonical stdout is an atomic deterministic merge of only sealed
            # attempt snapshots. Retry authorization never reads this aggregate.
            if len(attempts) == 1:
                canonical_data = attempt_stdout_payloads[0]
            else:
                merged = bytearray()
                for record, data in zip(attempts, attempt_stdout_payloads):
                    merged.extend(
                        (
                            f"===== verifier attempt {record['attempt']}/"
                            f"{_Q38_VERIFIER_RETRY_MAX_ATTEMPTS} "
                            f"ctrf={record['ctrf']['state']} "
                            f"dependency={','.join(record['dependency_reason_codes']) or 'none'} "
                            f"error={record['execution_error_type'] or record['permission_handoff_error_type'] or record['snapshot_wait_error_type'] or record['evidence_error_type'] or 'none'} "
                            f"retry={str(record['retry_scheduled']).lower()} =====\\n"
                        ).encode("utf-8")
                    )
                    merged.extend(data)
                    if data and not data.endswith(b"\\n"):
                        merged.extend(b"\\n")
                canonical_data = bytes(merged)
            _q38_write_atomic_bytes(
                self.trial_paths.test_stdout_path, canonical_data
            )

            all_evidence_complete = all(
                bool(record.get("evidence_complete")) for record in attempts
            )
            if (
                execution_error is not None
                or permission_handoff_error is not None
                or snapshot_error is not None
                or evidence_error is not None
            ):
                publish_lineage(
                    "raised_attempt_error",
                    complete=False,
                    evidence_complete=all_evidence_complete,
                )
                if execution_error is not None:
                    raise execution_error
                if permission_handoff_error is not None:
                    raise permission_handoff_error
                if snapshot_error is not None:
                    raise snapshot_error
                raise evidence_error

            if not retry:
                loop_complete = all_evidence_complete
                publish_lineage(
                    "awaiting_reward",
                    complete=False,
                    evidence_complete=all_evidence_complete,
                )
                break

            publish_lineage(
                "retry_pending",
                complete=False,
                evidence_complete=all_evidence_complete,
            )
            try:
                await clear_official_outputs()
                # Still inside Verifier.verify(): Harbor's existing outer
                # wait_for bounds snapshot waits, cleanup, backoff, and retries.
                await _q38_dependency_retry_backoff(retry_backoff_seconds)
            except BaseException as e:
                attempts[-1]["retry_backoff_error_type"] = type(e).__name__
                attempts[-1]["retry_backoff_error_message_sha256"] = hashlib.sha256(
                    str(e).encode("utf-8", errors="replace")
                ).hexdigest()
                publish_lineage(
                    "raised_retry_transition_error",
                    complete=False,
                    evidence_complete=all_evidence_complete,
                )
                raise
            attempts[-1]["retry_backoff_completed"] = True
            publish_lineage(
                "retry_transition_complete",
                complete=False,
                evidence_complete=all_evidence_complete,
            )
"""

REWARD_OLD = """        if self.trial_paths.reward_json_path.exists():
            rewards = self._parse_reward_json()
        elif self.trial_paths.reward_text_path.exists():
            rewards = self._parse_reward_text()
        else:
            raise RewardFileNotFoundError(
                f"No reward file found at {self.trial_paths.reward_text_path} or {
                    self.trial_paths.reward_json_path
                }"
            )

        return VerifierResult(rewards=rewards)
"""

REWARD_NEW = """        reward_error = None
        reward_kind = None
        reward_path = None
        reward_record = None
        reward_data = None
        reward_value = None
        reward_parse_error = None
        rewards = None
        terminal_reward_outputs = (
            attempts[-1].get("reward_outputs", {}) if attempts else {}
        )
        terminal_reward_payloads = attempt_reward_payloads[-1] if attempts else {}
        final_reward_outputs = {}
        for kind, source_path in (
            ("json", self.trial_paths.reward_json_path),
            ("text", self.trial_paths.reward_text_path),
        ):
            live_record, live_data = _q38_regular_file_observation(
                source_path, _Q38_VERIFIER_RETRY_MAX_REWARD_BYTES
            )
            live_value, live_parse_error = (
                _q38_reward_value(kind, live_data)
                if live_data is not None
                else (None, live_record.get("read_error_type"))
            )
            evidence = dict(live_record)
            evidence["path"] = (
                str(source_path.relative_to(self.trial_paths.trial_dir))
                if live_data is not None
                else None
            )
            evidence["value"] = live_value
            evidence["parse_error_type"] = live_parse_error
            final_reward_outputs[kind] = evidence
        reward_outputs_match_snapshot = bool(
            attempts
            and all(
                all(
                    final_reward_outputs[kind].get(key)
                    == terminal_reward_outputs.get(kind, {}).get(key)
                    for key in (
                        "sha256",
                        "bytes",
                        "read_error_type",
                        "value",
                        "parse_error_type",
                    )
                )
                for kind in ("json", "text")
            )
        )
        try:
            json_evidence = terminal_reward_outputs.get("json", {})
            text_evidence = terminal_reward_outputs.get("text", {})
            if json_evidence.get("state") != "missing":
                reward_kind = "json"
                reward_path = self.trial_paths.reward_json_path
            elif text_evidence.get("state") != "missing":
                reward_kind = "text"
                reward_path = self.trial_paths.reward_text_path
            else:
                raise RewardFileNotFoundError(
                    f"No reward file found at {self.trial_paths.reward_text_path} or {
                        self.trial_paths.reward_json_path
                    }"
                )
            reward_record = terminal_reward_outputs.get(reward_kind, {})
            reward_data = terminal_reward_payloads.get(reward_kind)
            if reward_data is None:
                raise VerifierOutputParseError(
                    f"Unsafe or unreadable captured reward at {reward_path}: "
                    f"{reward_record.get('read_error_type')}"
                )
            if len(reward_data) == 0:
                raise RewardFileEmptyError(
                    f"Reward file is empty at {reward_path}"
                )
            reward_value, reward_parse_error = _q38_reward_value(
                reward_kind, reward_data
            )
            if reward_parse_error is not None:
                raise VerifierOutputParseError(
                    f"Failed to parse bounded reward snapshot at {reward_path}: "
                    f"{reward_parse_error}"
                )
            rewards = reward_value
        except BaseException as e:
            reward_error = e

        final_ctrf_record, final_ctrf_data = _q38_regular_file_observation(
            ctrf_path,
            _Q38_VERIFIER_RETRY_MAX_CTRF_BYTES,
            parse_ctrf=True,
        )
        final_ctrf = dict(final_ctrf_record)
        final_ctrf["path"] = (
            str(ctrf_path.relative_to(self.trial_paths.trial_dir))
            if final_ctrf_data is not None
            else None
        )
        last_ctrf = attempts[-1]["ctrf"] if attempts else {}
        ctrf_content_keys = ("state", "sha256", "bytes", "read_error_type")
        final_ctrf_matches_snapshot = bool(
            attempts
            and all(
                final_ctrf.get(key) == last_ctrf.get(key)
                for key in ctrf_content_keys
            )
        )

        final_reward = None
        final_reward_matches_snapshot = False
        if reward_path is not None and reward_kind is not None:
            if reward_record is None:
                reward_record = {
                    "state": "missing",
                    "sha256": None,
                    "bytes": None,
                    "read_error_type": "FileNotFoundError",
                }
            final_reward = dict(final_reward_outputs[reward_kind])
            final_reward.update(
                {
                    "kind": reward_kind,
                    "value": reward_value,
                    "parse_error_type": reward_parse_error,
                }
            )
            last_reward = (
                attempts[-1].get("reward_outputs", {}).get(reward_kind, {})
                if attempts
                else {}
            )
            final_reward_matches_snapshot = bool(
                reward_data is not None
                and all(
                    final_reward.get(key) == last_reward.get(key)
                    for key in ("sha256", "bytes", "read_error_type", "value", "parse_error_type")
                )
                and reward_value == rewards
            )

        terminal_error = reward_error
        verifier_result = None
        if terminal_error is None:
            try:
                verifier_result = VerifierResult(rewards=rewards)
            except BaseException as e:
                terminal_error = e
        mismatch_reason = None
        if terminal_error is None and not final_ctrf_matches_snapshot:
            mismatch_reason = "final_ctrf_changed_after_snapshot"
            terminal_error = RuntimeError(mismatch_reason)
        if terminal_error is None and not reward_outputs_match_snapshot:
            mismatch_reason = "final_reward_outputs_changed_after_snapshot"
            terminal_error = RuntimeError(mismatch_reason)
        if terminal_error is None and not final_reward_matches_snapshot:
            mismatch_reason = "final_reward_changed_after_snapshot"
            terminal_error = RuntimeError(mismatch_reason)

        all_evidence_complete = bool(
            loop_complete
            and attempts
            and all(bool(record.get("evidence_complete")) for record in attempts)
            and final_ctrf_matches_snapshot
            and final_reward_matches_snapshot
            and reward_outputs_match_snapshot
        )
        final_extra = {
            "final_ctrf": final_ctrf,
            "final_ctrf_matches_last_attempt": final_ctrf_matches_snapshot,
            "final_reward": final_reward,
            "final_reward_matches_last_attempt": final_reward_matches_snapshot,
            "final_reward_outputs": final_reward_outputs,
            "final_reward_outputs_match_last_attempt": reward_outputs_match_snapshot,
            "returned_reward_value": rewards,
            "terminal_error_type": (
                type(terminal_error).__name__ if terminal_error is not None else None
            ),
            "terminal_error_message_sha256": (
                hashlib.sha256(
                    str(terminal_error).encode("utf-8", errors="replace")
                ).hexdigest()
                if terminal_error is not None
                else None
            ),
            "terminal_mismatch_reason": mismatch_reason,
        }
        if terminal_error is not None:
            publish_lineage(
                "raised_reward_or_final_evidence_error",
                complete=False,
                evidence_complete=False,
                extra=final_extra,
            )
            raise terminal_error

        publish_lineage(
            "returned_verifier_result",
            complete=all_evidence_complete,
            evidence_complete=all_evidence_complete,
            extra=final_extra,
        )
        return verifier_result
"""

TRANSFORMS = (
    (IMPORT_OLD, IMPORT_NEW, "retry imports"),
    (CLASS_OLD, CLASS_NEW, "retry evidence helpers"),
    (EXECUTION_OLD, EXECUTION_NEW, "verifier attempt loop"),
    (REWARD_OLD, REWARD_NEW, "verifier terminal reward evidence"),
)


def transform(text: str) -> tuple[str, bool]:
    states: list[str] = []
    for old, new, label in TRANSFORMS:
        old_count = text.count(old)
        new_count = text.count(new)
        if old_count == 1 and new_count == 0:
            states.append("old")
        elif old_count == 0 and new_count == 1:
            states.append("new")
        else:
            raise RuntimeError(
                f"unexpected Harbor verifier source shape for {label}: "
                f"old={old_count} new={new_count}"
            )
    if len(set(states)) != 1:
        raise RuntimeError(f"partially applied verifier retry patch: {states}")
    if states[0] == "new":
        return text, False
    for old, new, _ in TRANSFORMS:
        text = text.replace(old, new, 1)
    return text, True


def patch(target: Path) -> dict[str, object]:
    before = target.read_text(encoding="utf-8")
    after, changed = transform(before)
    if changed:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            handle.write(after)
            temporary = Path(handle.name)
        os.chmod(temporary, target.stat().st_mode)
        temporary.replace(target)
    return {
        "schema": "qwen38.harbor-verifier-retry-patch.v3.1",
        "target": str(target),
        "changed": changed,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "policy": POLICY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    report = patch(args.target.resolve())
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

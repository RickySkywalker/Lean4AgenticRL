#!/usr/bin/env python3
"""Fit constructed summarizer requests within the configured input budget."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path

MARKER = "_q38_fit_subagent_history"
REQUIRES = "_q38_custom_tokenizer"
SCHEMA = "qwen38.harbor-summarizer-input-fit.v1"

HELPERS = '''    def _q38_count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Token count for an arbitrary message list, with the same tokenizer as _count_total_tokens."""
        from litellm.utils import token_counter

        tokenizer = self._q38_custom_tokenizer()
        if tokenizer is not None:
            return token_counter(
                model=self._model_name, messages=messages, custom_tokenizer=tokenizer
            )
        return token_counter(model=self._model_name, messages=messages)

    def _q38_fit_subagent_history(
        self, message_history: list[dict[str, Any]], prompt: str, label: str
    ) -> list[dict[str, Any]]:
        """Return a history whose tokens plus the prompt fit under the advertised input wall."""
        import os

        if os.environ.get("Q38_SUMMARIZER_FIT", "1") != "1":
            return message_history
        try:
            wall = int(self._llm.get_model_context_limit())
            margin = int(os.environ.get("Q38_SUMMARIZER_FIT_MARGIN_TOKENS", "2048"))
        except Exception:
            return message_history
        budget = wall - margin
        if budget <= 0:
            return message_history
        hist = list(message_history)
        tail_keep = 2 if len(hist) >= 3 else 0
        try:
            used = self._q38_count_messages(hist + [{"role": "user", "content": prompt}])
        except Exception:
            return message_history
        if used <= budget:
            return message_history
        dropped = 0
        while used > budget and len(hist) - 1 - tail_keep >= 2:
            del hist[1:3]
            dropped += 2
            used = self._q38_count_messages(hist + [{"role": "user", "content": prompt}])
        truncated = 0
        if used > budget and tail_keep and isinstance(hist[-1].get("content"), str):
            text = hist[-1]["content"]
            keep = max(2000, len(text) - (used - budget) * 4 - 2000)
            if keep < len(text):
                half = keep // 2
                hist[-1] = dict(
                    hist[-1],
                    content=text[:half]
                    + "\\n[... summary truncated to fit the model's input window ...]\\n"
                    + text[-half:],
                )
                truncated = len(text) - keep
                used = self._q38_count_messages(
                    hist + [{"role": "user", "content": prompt}]
                )
        self.logger.info(
            f"Q38_FIT_APPLIED {label}: dropped {dropped} message(s), "
            f"truncated {truncated} summary char(s); input now ~{used} tokens "
            f"against an input wall of {wall}"
        )
        return hist

'''

ANCHOR_HELPERS = "    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:\n"

ANCHOR_CALL = """        start_time = time.time()
        response: LLMResponse = await self._llm.call(
            prompt=prompt,
            message_history=message_history,
            **self._llm_call_kwargs,
        )
"""

CALL_NEW = """        message_history = self._q38_fit_subagent_history(
            message_history, prompt, subagent_name_for_logging
        )
        start_time = time.time()
        response: LLMResponse = await self._llm.call(
            prompt=prompt,
            message_history=message_history,
            **self._llm_call_kwargs,
        )
"""


def transform(text: str) -> tuple[str, bool]:
    """Apply the summarizer patch after the tokenizer patch, rejecting unexpected source."""
    if MARKER in text:
        return text, False
    if REQUIRES not in text:
        raise ValueError(
            f"source has no {REQUIRES}: the qwen-tokenizer patch must be declared/applied first"
        )
    out = _replace_once(text, ANCHOR_HELPERS, HELPERS + ANCHOR_HELPERS, "helpers")
    out = _replace_once(out, ANCHOR_CALL, CALL_NEW, "subagent call")
    ast.parse(out)
    if MARKER not in out or "Q38_FIT_APPLIED" not in out:
        raise ValueError("patched source lost its own markers")
    return out, True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(f"anchor {what!r} occurs {n} times, expected exactly 1")
    return text.replace(old, new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="path to the venv's terminus_2.py")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    target = Path(args.target)
    if not target.is_file():
        raise SystemExit(f"FAIL target not found: {target}")
    before = target.read_text()
    before_sha = _sha256(target)

    if REQUIRES not in before:
        raise SystemExit(
            f"FAIL {target} has no {REQUIRES}: run patch_harbor_qwen_tokenizer.py FIRST. Counting with "
            "litellm's default tokenizer undercounts this model and would let the fit overshoot the wall."
        )

    try:
        source, changed = transform(before)
    except ValueError as exc:
        raise SystemExit(f"FAIL {exc}")
    if not changed:
        status = "ALREADY_PATCHED"
        after_sha = before_sha
    else:
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".q38fit.")
        with os.fdopen(fd, "w") as fh:
            fh.write(source)
        os.replace(tmp, target)
        after_sha = _sha256(target)
        status = "PATCHED"

    text = target.read_text()
    manifest = {
        "schema": SCHEMA,
        "status": status,
        "target": str(target),
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "marker_present": MARKER in text,
        "log_marker": "Q38_FIT_APPLIED",
        "env_flags": {
            "Q38_SUMMARIZER_FIT": "1 enables (default 1); any other value is a no-op passthrough",
            "Q38_SUMMARIZER_FIT_MARGIN_TOKENS": "safety margin under the wall, default 2048",
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"HARBOR_SUMMARIZER_FIT_{status} {target}")


if __name__ == "__main__":
    main()

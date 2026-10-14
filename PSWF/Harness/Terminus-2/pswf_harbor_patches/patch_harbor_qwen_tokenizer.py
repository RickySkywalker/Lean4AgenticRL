#!/usr/bin/env python3
"""Patch Harbor to count tokens with the served model's tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "qwen38.harbor-qwen-tokenizer.v1"

ORIGINAL = '''    def _count_total_tokens(self, chat: Chat) -> int:
        """Count total tokens across all messages in the chat."""
        from litellm.utils import token_counter

        return token_counter(model=self._model_name, messages=chat.messages)
'''

REPLACEMENT = '''    _q38_tokenizer_cache = None

    @classmethod
    def _q38_custom_tokenizer(cls):
        """Load and cache the configured tokenizer, or return None on failure."""
        if cls._q38_tokenizer_cache is None:
            import os

            path = os.environ.get("Q38_TOKENIZER_JSON", "")
            try:
                from tokenizers import Tokenizer

                cls._q38_tokenizer_cache = {
                    "type": "huggingface_tokenizer",
                    "tokenizer": Tokenizer.from_file(path),
                }
            except Exception:
                cls._q38_tokenizer_cache = False
        return cls._q38_tokenizer_cache or None

    def _count_total_tokens(self, chat: Chat) -> int:
        """Count total tokens across all messages in the chat."""
        from litellm.utils import token_counter

        tokenizer = self._q38_custom_tokenizer()
        if tokenizer is not None:
            return token_counter(
                model=self._model_name,
                messages=chat.messages,
                custom_tokenizer=tokenizer,
            )
        return token_counter(model=self._model_name, messages=chat.messages)
'''


def transform(text: str) -> tuple[str, bool]:
    """Apply the tokenizer patch to the pinned source, rejecting an unexpected anchor."""
    if "_q38_custom_tokenizer" in text:
        return text, False
    if text.count(ORIGINAL) != 1:
        raise ValueError(
            "pinned Terminus-2 does not contain exactly one _count_total_tokens body of the expected shape"
        )
    return text.replace(ORIGINAL, REPLACEMENT, 1), True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path, help="model dir holding tokenizer.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    target = args.harbor_root / "agents" / "terminus_2" / "terminus_2.py"
    if not target.is_file():
        print(f"patch target missing: {target}", file=sys.stderr)
        return 2
    tokenizer_json = args.tokenizer / "tokenizer.json"
    if not tokenizer_json.is_file():
        print(f"tokenizer.json missing: {tokenizer_json}", file=sys.stderr)
        return 2

    before = sha256(target)
    text = target.read_text(encoding="utf-8")
    if "_q38_custom_tokenizer" in text:
        status = "ALREADY_APPLIED"
    elif ORIGINAL in text:
        target.write_text(text.replace(ORIGINAL, REPLACEMENT, 1), encoding="utf-8")
        status = "APPLIED"
    else:
        print(
            "refusing to patch: the exact original _count_total_tokens body was not found, so this Harbor "
            "differs from the pinned revision and a blind edit could change unrelated behaviour",
            file=sys.stderr,
        )
        return 3

    import ast

    ast.parse(target.read_text(encoding="utf-8"))

    payload = {
        "schema_version": SCHEMA,
        "status": status,
        "target": str(target),
        "sha256_before": before,
        "sha256_after": sha256(target),
        "tokenizer_json": str(tokenizer_json),
        "tokenizer_json_sha256": sha256(tokenizer_json),
        "env_var_consumed": "Q38_TOKENIZER_JSON",







    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[qwen-tokenizer] {status} {target} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract workflow candidates and validate their structure before injection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

SPEC_MIN_CHARS_DEFAULT = 200

_SPEC_STOPWORDS = frozenset((
    "the", "and", "to", "of", "a", "in", "is", "it", "for", "on", "with", "that", "this", "be", "are",
    "as", "at", "by", "or", "an", "if", "from", "will", "then", "not", "we", "you", "should", "must",
    "each", "all", "any", "can", "run", "file", "files", "check", "step", "steps", "output",
))

_SECTION_MARKERS = (
    "acceptance",
    "environment finding",
    "plan",
    "checkpoint",
    "verif",
)


@dataclass
class SpecCandidate:
    text: str
    source: str
    score: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


def marker_score(text: str) -> int:
    """Count workflow section markers in a candidate."""
    low = (text or "").lower()
    return sum(1 for m in _SECTION_MARKERS if m in low)


def _json_plan(content: str) -> str | None:
    """Extract string-valued plan fields from assistant JSON objects."""
    if not content:
        return None
    start = content.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(content)):
            c = content[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(content[start : i + 1])
                    except Exception:  # noqa: BLE001
                        break
                    if isinstance(obj, dict) and isinstance(obj.get("plan"), str):
                        return obj["plan"]
                    break
        start = content.find("{", start + 1)
    return None


def extract_spec(
    chat_messages: list[dict[str, Any]] | None,
    trajectory_steps: list[Any] | None,
    min_chars: int = SPEC_MIN_CHARS_DEFAULT,
) -> tuple[str, str, list[SpecCandidate]]:
    """Rank candidates from structured plans, trajectories, assistant text, and reasoning."""
    cands: list[SpecCandidate] = []
    msgs = chat_messages or []
    steps = trajectory_steps or []

    assistant_contents = [m.get("content") or "" for m in msgs if m.get("role") == "assistant"]
    for idx, content in enumerate(reversed(assistant_contents)):
        plan = _json_plan(content if isinstance(content, str) else str(content))
        if plan and len(plan.strip()) >= min_chars:
            cands.append(SpecCandidate(plan.strip(), "json_plan", detail={"turns_from_end": idx}))

    for st in steps:
        if getattr(st, "source", None) != "agent":
            continue
        msg = getattr(st, "message", None) or ""
        m = re.search(r"^Plan:\s*(.+)\Z", msg, flags=re.S | re.M)
        if m and len(m.group(1).strip()) >= min_chars:
            cands.append(SpecCandidate(m.group(1).strip(), "trajectory_plan"))

    for content in assistant_contents:
        if isinstance(content, str) and len(content.strip()) >= min_chars:
            cands.append(SpecCandidate(content.strip(), "assistant_text"))

    for st in steps:
        rc = getattr(st, "reasoning_content", None)
        if isinstance(rc, str) and len(rc.strip()) >= min_chars:
            cands.append(SpecCandidate(rc.strip(), "reasoning_content"))

    if not cands:
        return "", "none", []

    rung_rank = {"json_plan": 3, "trajectory_plan": 2, "assistant_text": 1, "reasoning_content": 0}
    for c in cands:
        c.score = marker_score(c.text)
    best = max(cands, key=lambda c: (c.score, rung_rank[c.source], len(c.text)))
    return best.text, best.source, cands


def sanitize_spec(txt: str, min_chars: int = SPEC_MIN_CHARS_DEFAULT) -> tuple[str, str, str]:
    """Return an accepted workflow or the rejected text with a validation reason."""
    if not txt:
        return "", "", "empty"

    if "</think>" in txt:
        txt = txt.rsplit("</think>", 1)[1]
    txt = re.sub(r"^\s*(?:THOUGHT|Thought)\s*:.*?(?=\n\s*(?:#|\d+\.\s|\*\*|---))", "", txt, flags=re.S)
    txt = txt.lstrip("\n ")

    stripped = txt.lstrip()
    low = txt.lower()
    words = re.findall(r"[A-Za-z']+", low)
    sw_ratio = (sum(1 for w in words if w in _SPEC_STOPWORDS) / len(words)) if words else 0.0
    coherent = len(words) >= 40 and sw_ratio >= 0.08
    shaped = (
        "specified workflow" in low
        or re.match(r"^#{1,6}\s+\S", stripped) is not None
        or sum(k in low for k in ("acceptance", "environment", "plan", "verif")) >= 3
    )
    if len(stripped) >= min_chars and coherent and shaped:
        return txt, "", "ok"

    why = []
    if len(stripped) < min_chars:
        why.append(f"chars={len(stripped)}<{min_chars}")
    if len(words) < 40:
        why.append(f"words={len(words)}<40")
    elif sw_ratio < 0.08:
        why.append(f"stopword_ratio={sw_ratio:.3f}<0.08")
    if not shaped:
        hits = [k for k in ("acceptance", "environment", "plan", "verif") if k in low]
        why.append(f"unshaped(keywords={len(hits)}/4{hits},title={'specified workflow' in low})")
    return "", txt, "; ".join(why) or "rejected"

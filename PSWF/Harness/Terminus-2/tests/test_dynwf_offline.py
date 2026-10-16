#!/usr/bin/env python3
"""Synthetic regression tests for dynwf offline."""

from __future__ import annotations

from synthetic_fixtures import synthetic_plan

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminus2_dynwf import wf_render  # noqa: E402
from terminus2_dynwf.spec_extract import extract_spec, marker_score, sanitize_spec  # noqa: E402


FAILS: list[str] = []


def check(cond: bool, label: str, extra: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {extra}")
        FAILS.append(label)


@dataclass
class FakeStep:
    source: str = "agent"
    message: str = ""
    reasoning_content: str | None = None


SYNTHETIC_SPEC = synthetic_plan()

REASONING_PREAMBLE = 'THOUGHT: This handwritten synthetic reply contains deliberation rather than a workflow. I still need to inspect the invented workspace and decide which operation to attempt. The requested fixture file might already exist, or I might need to create it. I have not selected a concrete command or a verification procedure. I will first read the terminal output.'


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


print("== prompt asset ==")
doc = wf_render.load_wf()
names = [((e or {}).get("step") or {}).get("name") for e in doc["workflow"]]
check(
    names == [
        "explore_and_specify",
        "solve_task",
        "solve_task_fallback",
        "phase1_wrapup_nudge",
        "phase1_wrapup_reminder",
        "phase1_forced_spec",
        "phase1_hard_stop",
        "phase1_hard_stop_repeat",
    ],
    "asset has the expected steps, in order",
    str(names),
)

p1 = wf_render.render_phase1(doc, "Make the tests pass.", 40)
check("Make the tests pass." in p1, "phase-1 prompt carries the problem statement")
check("~40 turns" in p1, "phase-1 turn budget is substituted")
check("{{" not in p1, "phase-1 prompt has no unsubstituted placeholders", p1[:200])
check(
    "`commands`: the empty array `[]`" in p1
    and "`task_complete`: `true`" in p1
    and "`plan`: **the complete Specified Workflow**" in p1,
    "phase-1 prompt states the plan-field delivery channel",
)
for bad in (
    "Bash, Read, Grep, Glob",
    "at least one tool call",
    "each shell command runs in a new subshell",
    "end your turn with NO tool call",
):
    check(bad not in p1, f"phase-1 prompt uses the terminal JSON contract: excludes {bad!r}")

sections = (
    "Goal & acceptance criteria",
    "Environment findings",
    "Plan",
    "Execution checkpoints",
    "Verification",
)
section_positions = [p1.find(f"{index}. **{section}**") for index, section in enumerate(sections, 1)]
check(all(position >= 0 for position in section_positions),
      "phase-1 prompt requests all five workflow sections")
check(section_positions == sorted(section_positions),
      "workflow sections put acceptance criteria and findings before execution and verification")
check("exact commands you will run at the end to prove each acceptance" in p1,
      "workflow verification must connect commands to acceptance criteria")

print("== extraction ladder ==")
resp = json.dumps(
    {"analysis": "Exploration finished.", "plan": SYNTHETIC_SPEC, "commands": [], "task_complete": True}
)
txt, src, cands = extract_spec([msg("user", "..."), msg("assistant", resp)], [])
check(src == "json_plan", "rung 1 wins on a well-formed JSON response", src)
check(txt == SYNTHETIC_SPEC.strip(), "rung 1 returns the plan verbatim")

txt, src, _ = extract_spec([msg("assistant", "Here you go:\n" + resp + "\nDone.")], [])
check(src == "json_plan" and txt == SYNTHETIC_SPEC.strip(), "rung 1 survives prose around the JSON", src)

short = json.dumps({"analysis": "a", "plan": SYNTHETIC_SPEC[:600], "commands": [], "task_complete": True})
txt, src, _ = extract_spec([msg("assistant", resp), msg("assistant", short)], [])
check(len(txt) == len(SYNTHETIC_SPEC.strip()), "rung 1 picks the fuller plan, not merely the last", str(len(txt)))

steps = [FakeStep(message=f"Analysis: done\nPlan: {SYNTHETIC_SPEC}")]
txt, src, _ = extract_spec([msg("assistant", "not json at all")], steps)
check(src == "trajectory_plan", "rung 2 fires when only the trajectory has it", src)

broken = '{"analysis": "done", "plan": "' + SYNTHETIC_SPEC + '", "commands": [], "task_complete": true}'
txt, src, _ = extract_spec([msg("assistant", broken)], [])
check(src == "assistant_text", "rung 3 recovers an unparseable response", src)
check(SYNTHETIC_SPEC[:60].strip() in txt, "rung 3 text still contains the plan")

steps = [FakeStep(message="Analysis: x", reasoning_content=SYNTHETIC_SPEC)]
txt, src, _ = extract_spec([msg("assistant", "{}")], steps)
check(src == "reasoning_content", "rung 4 is the last resort", src)

txt, src, cands = extract_spec([], [])
check(src == "none" and txt == "" and cands == [], "empty phase 1 yields spec_source=none")

print("== workflow validation ==")
acc, rej, why = sanitize_spec(SYNTHETIC_SPEC)
check(bool(acc) and not rej, "a synthetic plan with a one-line summary first is ACCEPTED", why)
acc, rej, why = sanitize_spec(REASONING_PREAMBLE)
check(not acc and bool(rej), "a THOUGHT reasoning preamble is REJECTED", why)
check("stopword" in why or "unshaped" in why or "words=" in why, "rejection reason is specific", why)
acc, rej, why = sanitize_spec("#")
check(not acc, "a one-character '#' plan is REJECTED", why)
acc, rej, why = sanitize_spec("<think>" + REASONING_PREAMBLE + "</think>\n" + SYNTHETIC_SPEC)
check(bool(acc) and "THOUGHT" not in acc, "a leaked <think> block is stripped and the plan kept", why)
acc, rej, why = sanitize_spec("x " * 300)
check(not acc, "300 repetitions of one token are REJECTED as incoherent", why)
acc, rej, why = sanitize_spec(REASONING_PREAMBLE)
check(rej.strip().endswith("the terminal output."), "rejected text is returned intact, not blanked")

print("== phase-2 render ==")
p2, variant = wf_render.render_phase2(doc, "Make the tests pass.", SYNTHETIC_SPEC, same_chat=True)
check(variant == "injected", "spec present -> injected variant")
check(wf_render.SPEC_SECTION_HEADER in p2, "phase-2 prompt has the '## Specified Workflow' header")
check(SYNTHETIC_SPEC.strip()[:80] in p2, "phase-2 prompt contains the spec text")
check(wf_render.MEMORY_CLAIM_SAME in p2, "same_chat -> the same-session memory sentence")
check("{{" not in p2, "phase-2 prompt has no unsubstituted placeholders")
needles = wf_render.spec_needles(SYNTHETIC_SPEC)
check(len(needles) >= 5, f"spec_needles produced {len(needles)} needles")
check(all(n in p2 for n in needles), "every needle appears verbatim in the rendered prompt")

p2f, variant = wf_render.render_phase2(doc, "Make the tests pass.", None, same_chat=True)
check(variant == "fallback", "no spec -> fallback variant")
check(
    wf_render.SPEC_SECTION_HEADER not in p2f,
    "fallback strips the WHOLE section, not just its body (an empty heading is worse than none)",
)
check("Make the tests pass." in p2f, "fallback still carries the problem statement")

p2fresh, _ = wf_render.render_phase2(doc, "T", SYNTHETIC_SPEC, same_chat=False)
check(wf_render.MEMORY_CLAIM_FRESH in p2fresh, "fresh_chat -> the honest memory sentence")
check(
    wf_render.MEMORY_CLAIM_SAME not in p2fresh,
    "fresh_chat does NOT claim the exploration is in memory",
)
check(
    "Keep replying in the same JSON format" not in p2fresh,
    "fresh_chat does not append the format restatement (the template supplies it)",
)

print("== marker scoring ==")
check(marker_score(SYNTHETIC_SPEC) == 5, f"a full plan scores 5/5, got {marker_score(SYNTHETIC_SPEC)}")
check(marker_score(REASONING_PREAMBLE) <= 1, f"a preamble scores low, got {marker_score(REASONING_PREAMBLE)}")

print()
if FAILS:
    print(f"OFFLINE GATE FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("OFFLINE GATE: ALL GREEN")

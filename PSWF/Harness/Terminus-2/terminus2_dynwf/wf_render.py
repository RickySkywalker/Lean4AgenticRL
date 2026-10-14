"""Render workflow prompts and handoffs from the packaged YAML templates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_WF_YAML = PROMPTS_DIR / "dynamic_workflow_terminal_terminus_v1.yaml"

MEMORY_CLAIM_SAME = (
    "You already explored this environment earlier in this same conversation (it is in your memory, and this "
    "is the SAME container — everything you inspected is unchanged)."
)
MEMORY_CLAIM_FRESH = (
    "You explored this environment in a previous phase. That conversation is NOT in your context any more, but "
    "this is the SAME container, so every change and every file you saw is still there — re-check anything you "
    "need rather than assuming."
)

MEMORY_CLAIM_COMPACT = (
    "You explored this environment earlier in this session. That exploration has been COMPACTED: the raw "
    "conversation is no longer in your context, but the plan below survived it verbatim, and the handoff notes "
    "at the end of this message are a summary of what you found plus answers to the questions you asked about "
    "it. This is the SAME container \u2014 every change and every file you saw is still there. Trust the plan and "
    "the notes; re-check anything they do not settle rather than assuming."
)

COMPACTED_NOTES_HEADER = "## Handoff notes from the exploration phase (compacted)"

SPEC_SECTION_HEADER = "## Specified Workflow"

CONFIRM_CLAIM_DOUBLE = (
    "You will then receive one confirmation message. Reply to it with `task_complete: true` again and the\n"
    "SAME Specified Workflow in `plan`; that second confirmation is what ends this phase."
)
CONFIRM_CLAIM_SINGLE = (
    "That one response ends this phase — there is no confirmation round and you will not be asked again, so\n"
    "send the workflow complete the first time."
)


class PromptAssetError(RuntimeError):
    pass


def load_wf(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_WF_YAML
    if not p.is_file():
        raise PromptAssetError(f"workflow prompt asset not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text())
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim; a bad asset must not start a run
        raise PromptAssetError(f"{p}: not parseable as YAML: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("workflow"), list):
        raise PromptAssetError(f"{p}: no `workflow:` list")
    return doc


def step_instruction(doc: dict[str, Any], name: str) -> str:
    for entry in doc["workflow"]:
        step = (entry or {}).get("step") or {}
        if step.get("name") == name:
            instr = step.get("instruction")
            if not isinstance(instr, str) or not instr.strip():
                raise PromptAssetError(f"step {name!r} has no instruction text")
            return instr
    have = [((e or {}).get("step") or {}).get("name") for e in doc["workflow"]]
    raise PromptAssetError(f"step {name!r} not in asset (have {have})")


def subst(text: str, **values: Any) -> str:
    """Replace named placeholders without interpreting unrelated braces."""
    out = text
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", "" if v is None else str(v))
    return out


def render_phase1(
    doc: dict[str, Any],
    problem_statement: str,
    phase1_max_turns: int,
    confirm_claim: str = CONFIRM_CLAIM_DOUBLE,
) -> str:
    return subst(
        step_instruction(doc, "explore_and_specify"),
        problem_statement=problem_statement,
        phase1_max_turns=phase1_max_turns,
        confirm_claim=confirm_claim,
    )


def _fmt_left(turns_left: int | None, seconds_left: float | None) -> dict[str, str]:
    """Format a nonnegative remaining budget for a prompt."""
    t = "no more" if turns_left is not None and turns_left <= 0 else (
        str(int(turns_left)) if turns_left is not None else "a few"
    )
    s = "under 60" if seconds_left is not None and seconds_left < 60 else (
        str(int(seconds_left)) if seconds_left is not None else "a few hundred"
    )
    return {"turns_left": t, "seconds_left": s}


def render_wrapup(
    doc: dict[str, Any],
    *,
    turns_left: int | None,
    seconds_left: float | None,
    short: bool = False,
) -> str:
    """Render the full wrap-up request or its shorter reminder."""
    name = "phase1_wrapup_reminder" if short else "phase1_wrapup_nudge"
    return subst(step_instruction(doc, name), **_fmt_left(turns_left, seconds_left))


def render_hard_stop(
    doc: dict[str, Any],
    *,
    n_refused: int,
    turns_left: int | None,
    seconds_left: float | None,
    short: bool = False,
) -> str:
    """Render the response used after exploration commands are refused."""
    name = "phase1_hard_stop_repeat" if short else "phase1_hard_stop"
    return subst(step_instruction(doc, name), n_refused=int(n_refused), **_fmt_left(turns_left, seconds_left))


def render_forced_spec(doc: dict[str, Any]) -> str:
    """Render the reserved turn that requests the specified workflow."""
    return step_instruction(doc, "phase1_forced_spec")


TRUNCATION_STOP_HEADER = (
    "=====================  YOUR LAST RESPONSE WAS CUT OFF AT THE OUTPUT LIMIT  ====================="
)


def render_truncation_stop(doc: dict[str, Any], *, short: bool = False) -> str:
    """Request the workflow after a phase-1 response reaches the output cap."""
    lines = [TRUNCATION_STOP_HEADER,
             "Your previous response reached the model's maximum output length and was truncated, so NONE "
             "of the actions in it were performed and nothing in the terminal has changed. Exploration for "
             "phase 1 is now CLOSED: no further commands will be executed in this phase.",
             ""]
    if short:
        lines += ["This has now happened more than once. Write a SHORT workflow this time -- aim for about "
                  "3,000 characters, five brief sections, no code listings and no reasoning about "
                  "alternatives. A short plan that arrives is worth far more than a long one that is cut "
                  "off again.", ""]
    return "\n".join(lines) + "\n" + step_instruction(doc, "phase1_forced_spec")


PHASE2_REPLAN_HEADER = (
    "=====================  CONTEXT IS ABOUT TO BE COMPACTED \u2014 REWRITE YOUR PLAN FIRST  ====================="
)


def render_phase2_replan(doc: dict[str, Any], *, free_tokens: int, index: int, have_spec: bool) -> str:
    """Request an updated plan before context compaction."""
    parts = [
        PHASE2_REPLAN_HEADER,
        f"Your conversation has grown to within about {int(free_tokens):,} tokens of the model's input limit, so "
        "it is about to be summarised and replaced. Everything you can still see will be gone from your next "
        "turn; a summary of it, plus the workflow you write below, is what survives.",
        "",
        "So write the workflow you would want to be holding on the other side of that boundary. This is a "
        "REWRITE, not a repetition: it must describe the task as it stands NOW.",
        "",
        "Reply with ONE JSON object and nothing else:",
        "- `commands`: the empty array `[]`. Commands in this response will NOT be executed -- the summarisation "
        "happens immediately after it.",
        "- `task_complete`: `false`.",
        "- `plan`: the complete, updated Specified Workflow as Markdown -- a one-line task summary, then the "
        "five sections: **Goal & acceptance criteria**, **Environment findings**, **Plan**, **Execution "
        "checkpoints**, **Verification**. This is a JSON string, so write real newlines as `\\n`.",
        "- `analysis`: one sentence. Do NOT put the workflow here.",
        "",
        "Carry forward, in the sections where they belong:",
        "- every requirement of the task, and which of them you have already SATISFIED and VERIFIED (name the "
        "command or test that proved it);",
        "- the exact paths, file names, commands, versions and values you will need again, written out in full "
        "-- after the compaction you cannot look them up in this conversation;",
        "- what is half-done, and the precise next action for it;",
        "- what you tried that did NOT work, so you do not pay for it twice;",
        "- the verification you still owe before the task can be called complete.",
        "",
        "Keep it under about 8,000 characters. Facts you have checked belong here; speculation does not.",
    ]
    if have_spec:
        parts += ["", "You wrote a Specified Workflow for this task in phase 1 and it has been re-stated to you "
                      "since. Treat it as a draft to correct, not as text to copy: drop what is done, fix what "
                      "the environment disproved, and keep what still stands."]
    if index > 1:
        parts += ["", f"This is rewrite number {index} of this task. Each one replaces the last, so state the "
                      "current position plainly rather than describing how it changed."]
    return "\n".join(parts) + "\n"


PHASE2_ENDGAME_HEADER = "## Time budget \u2014 the session is about to end"


def render_phase2_endgame(doc: dict[str, Any], *, seconds_left: float, short: bool = False) -> str:
    """Render a phase-2 reminder to finish deliverables and verification."""
    left = int(max(0.0, seconds_left))
    if short:
        return (f"[time budget] ~{left:,} s left. Deliverables on disk, then run your Verification commands; "
                "nothing new.")
    return "\n".join([
        PHASE2_ENDGAME_HEADER,
        f"About {left:,} seconds of this session remain. When they run out the session is terminated wherever it "
        "happens to be, and the task is graded on the state of THIS CONTAINER at that moment -- not on your "
        "analysis, your intentions, or anything you have worked out but not written down.",
        "",
        "So spend what is left in this order:",
        "1. WRITE THE DELIVERABLE. Every file the task asks for, at the exact path it asks for, with your current "
        "best content. A partially correct file that exists scores; a perfect one you never wrote does not.",
        "2. RUN YOUR OWN VERIFICATION. Execute the commands in the **Verification** section of your Specified "
        "Workflow, and any test command the task named. You still have time to find a mistake; you will not have "
        "time to find it later.",
        "3. FIX ONLY WHAT THOSE COMMANDS REPORT. Smallest change that makes the check pass.",
        "",
        "Do not start a new investigation, a refactor, or an alternative approach now. If something is "
        "unfinished, leave the best working version in place rather than a half-applied change.",
    ]) + "\n"


BUDGET_HINT_HEADER = "## Phase-1 budget \u2014 ENFORCED, in numbers"


def render_budget_hint(
    *,
    explore_s: float,
    reserve_s: float,
    phase1_max_turns: int,
    hard_stop_s: float | None,
    hard_stop_turns: int | None,
    nudge_s: float | None,
    nudge_turns: int | None,
    context_max_frac: float | None,
    truncation_stop: bool = False,
    grace_s: float | None = None,
) -> str:
    """Describe the configured exploration budget and enforcement thresholds."""
    lines = [BUDGET_HINT_HEADER,
             f"- Wall clock: exploration has about {int(explore_s)} seconds in total, and at most "
             f"{int(phase1_max_turns)} turns."]
    if nudge_s is not None or nudge_turns is not None:
        parts = []
        if nudge_s is not None:
            parts.append(f"about {int(nudge_s)} seconds")
        if nudge_turns is not None:
            parts.append(f"{int(nudge_turns)} turns")
        lines.append(f"- A budget reminder is appended to the terminal output after {' or '.join(parts)}.")
    if hard_stop_s is not None or hard_stop_turns is not None:
        parts = []
        if hard_stop_s is not None:
            parts.append(f"about {int(hard_stop_s)} seconds")
        if hard_stop_turns is not None:
            parts.append(f"about {int(hard_stop_turns)} turns")
        lines.append(
            f"- From {' or '.join(parts)}, any command batch you send is REFUSED and NOT executed; from that "
            "point only the deliverable (`commands: []`, `task_complete: true`, the Specified Workflow in `plan`) "
            "can end this phase."
        )
    if context_max_frac:
        lines.append(
            f"- Exploration is also closed early if this conversation grows past {int(round(context_max_frac * 100))}% "
            "of the context window, so keep command output short (head, wc, grep) rather than dumping files."
        )
    if truncation_stop:
        lines.append(
            "- Each single response must also fit the model's output limit. A response that is cut off at "
            "that limit performs NO actions and CLOSES exploration immediately, so keep each response "
            "short: a few commands, brief analysis, no long listings or code dumps."
        )
    if grace_s:
        lines.append(
            f"- If a response is already being generated when the {int(explore_s)} seconds run out, it is "
            f"allowed up to {int(grace_s)} more seconds to finish, but only after the refusal above has "
            "started; exploration stays closed for that time."
        )
    if reserve_s > 0:
        lines.append(
            f"- If the phase still closes without a workflow you get ONE forced turn of at most {int(reserve_s)} "
            "seconds to write it from memory, with no commands. Do not let it come to that."
        )
    target = []
    if nudge_s is not None:
        target.append(f"{int(nudge_s)} seconds")
    if nudge_turns is not None:
        target.append(f"{int(nudge_turns)} turns")
    lines.append(
        "- Plan your exploration to fit: batch several read-only commands per response, and send the Specified "
        "Workflow BEFORE the reminder appears"
        + (f" \u2014 aim for well under {' and '.join(target)}." if target else ".")
    )
    return "\n".join(lines) + "\n"


def truncate_spec(spec: str, max_chars: int) -> tuple[str, int]:
    """Clip a workflow on a line boundary and return its original length when clipped."""
    if max_chars <= 0 or len(spec) <= max_chars:
        return spec, 0
    original = len(spec)
    head = spec[:max_chars]
    cut = head.rfind("\n")
    if cut > max_chars // 2:
        head = head[:cut]
    return (
        head.rstrip()
        + f"\n\n[... this workflow was clipped at {max_chars} characters;"
        + f" {original - len(head.rstrip())} characters of the tail are not shown ...]\n",
        original,
    )


def render_phase2(
    doc: dict[str, Any],
    problem_statement: str,
    spec: str | None,
    same_chat: bool,
    restate_format: bool = True,
    handoff: str | None = None,
) -> tuple[str, str]:
    """Render a solving prompt whose memory claim matches the selected handoff."""
    mode = (handoff or ("same_chat" if same_chat else "fresh_chat")).strip().lower()
    claim = {
        "same_chat": MEMORY_CLAIM_SAME,
        "fresh_chat": MEMORY_CLAIM_FRESH,
        "compact": MEMORY_CLAIM_COMPACT,
    }.get(mode, MEMORY_CLAIM_SAME if same_chat else MEMORY_CLAIM_FRESH)
    if spec:
        text = subst(
            step_instruction(doc, "solve_task"),
            problem_statement=problem_statement,
            specified_workflow=spec,
            memory_claim=claim,
        )
        variant = "injected"
    else:
        text = subst(step_instruction(doc, "solve_task_fallback"), problem_statement=problem_statement)
        variant = "fallback"

    if mode == "same_chat" and restate_format:
        text += (
            "\n\nKeep replying in the same JSON format as before: an object with `analysis`, `plan` and "
            "`commands` (a list of `{keystrokes, duration}` objects), plus `task_complete` when you are done.\n"
        )
    return text, variant


def spec_needles(spec: str, k: int = 12) -> list[str]:
    """Choose distinctive workflow lines for checking prompt injection."""
    lines = [ln.strip() for ln in (spec or "").splitlines()]
    cand = [ln for ln in lines if len(ln) >= 40 and re.search(r"[A-Za-z]{4}", ln)]
    if not cand:
        return []
    stride = max(1, len(cand) // k)
    return cand[::stride][:k]

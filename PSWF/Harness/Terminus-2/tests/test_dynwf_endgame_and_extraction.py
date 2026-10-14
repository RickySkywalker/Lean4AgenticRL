#!/usr/bin/env python3
"""Synthetic regression tests for dynwf endgame and extraction."""
from __future__ import annotations

from synthetic_fixtures import synthetic_plan
import asyncio, json, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harbor.llms.base import LLMResponse  # noqa: E402
from terminus2_dynwf import wf_render  # noqa: E402
from terminus2_dynwf import dynamic_agent as da  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402

FAILS: list[str] = []


def check(cond, label, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  {extra}"))
    if not cond:
        FAILS.append(label)


def plan_text(marker: str) -> str:
    return synthetic_plan(marker)


OLD_SPEC = plan_text("Fix")
NEW_SPEC = plan_text("Finish")
POOR_SPEC = "\n".join([ln for ln in plan_text("Finish").splitlines()
                       if "Verification" not in ln and "cat /fixture/output.txt" not in ln])
REASONING = 'This synthetic deliberation mentions acceptance criteria, environment findings, the plan, execution checkpoints and verification. Acceptance concerns the requested text file. Environment means the invented fixture directory. Plan means choosing a command, while checkpoints concern reading the output. Verification compares the file with the requested content. I am still considering these sections instead of delivering their concrete instructions. This synthetic deliberation mentions acceptance criteria, environment findings, the plan, execution checkpoints and verification. Acceptance concerns the requested text file. Environment means the invented fixture directory. Plan means choosing a command, while checkpoints concern reading the output. Verification compares the file with the requested content. I am still considering these sections instead of delivering their concrete instructions. This synthetic deliberation mentions acceptance criteria, environment findings, the plan, execution checkpoints and verification. Acceptance concerns the requested text file. Environment means the invented fixture directory. Plan means choosing a command, while checkpoints concern reading the output. Verification compares the file with the requested content. I am still considering these sections instead of delivering their concrete instructions. This synthetic deliberation mentions acceptance criteria, environment findings, the plan, execution checkpoints and verification. Acceptance concerns the requested text file. Environment means the invented fixture directory. Plan means choosing a command, while checkpoints concern reading the output. Verification compares the file with the requested content. I am still considering these sections instead of delivering their concrete instructions. '

PARENT_SUMMARIES: list[str] = []
PARENT_EXECS: list[object] = []


async def parent_check(self, chat, original_instruction, session):
    return None


async def parent_summarize(self, chat, original_instruction, session):
    PARENT_SUMMARIES.append(original_instruction)
    self._summarization_count += 1
    return ("STOCK SUMMARY OF THE WORK", None)


async def parent_execute(self, commands, session):
    PARENT_EXECS.append(commands)
    return False, "root@host:/app# ok\nlast line of terminal output"


da.Terminus2._check_proactive_summarization = parent_check
da.Terminus2._summarize = parent_summarize
da.Terminus2._execute_commands = parent_execute


class FakeSession:
    async def get_incremental_output(self):
        return "root@host:/app#"

    async def is_session_alive(self):
        return True


class FakeChat:
    def __init__(self, response=None):
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []
        self.messages: list[dict] = []
        self._response = response

    async def chat(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return self._response if self._response is not None else LLMResponse(content="")

    def reset_response_chain(self):
        pass


MODEL_INFO = {"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
              "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"}
ENDGAME_CONFIG = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
              phase1_spec_reserve_s=600.0, phase1_forced_min_s=90.0, phase1_wrapup_frac=0.60,
              phase1_wrapup_turn_frac=0.60, phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75,
              phase1_context_max_frac=0.25, phase1_single_confirm=True, phase2_same_chat=False,
              phase1_budget_hint=True, phase1_truncation_stop=True, phase1_deadline_grace_s=420.0,
              phase2_reinject_on_summarize=True)


def make_agent(**kw):
    tmp = Path(tempfile.mkdtemp())
    merged = dict(ENDGAME_CONFIG)
    merged.update(kw)
    a = Terminus2Dynamic(logs_dir=tmp / "agent", model_name="openai/x", max_turns=400,
                         api_base="http://127.0.0.1:1/v1", temperature=1.0, suppress_max_turns_warning=True,
                         model_info=MODEL_INFO, llm_kwargs={"api_key": "local"},
                         trusted_root=str(tmp / "trusted"), **merged)
    (tmp / "agent").mkdir(parents=True, exist_ok=True)
    a._session = FakeSession()
    a._dynwf_phase = 2
    a._dynwf_spec_for_reinject = OLD_SPEC
    return a


def replan_response(spec, reasoning=""):
    body = json.dumps({"analysis": "restating the plan", "plan": spec, "commands": [],
                       "task_complete": False})
    return LLMResponse(content=body, reasoning_content=reasoning)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


print("\n-- endgame: the endgame reserve")

import time


a = make_agent(phase2_endgame_s=1800.0)
check(a.phase2_endgame_s == 1800.0, "the reserve is accepted as a kwarg")
a._dynwf_run_deadline = time.time() + 5000.0
PARENT_EXECS.clear()
occ, out = run(a._execute_commands(["ls"], a._session))
check("Time budget" not in out, "no nudge while 5,000 s of the budget remain")
check(PARENT_EXECS and PARENT_EXECS[-1] == ["ls"], "the command batch reached the terminal unchanged")

a._dynwf_run_deadline = time.time() + 900.0
occ, out = run(a._execute_commands(["make test"], a._session))
check("Time budget" in out, "the nudge fires inside the reserve")
check(PARENT_EXECS[-1] == ["make test"], "commands are NEVER refused in phase 2 (unlike the phase-1 hard stop)")
check(out.index("last line of terminal output") < out.index("Time budget"),
      "the nudge is appended AFTER the observation, where truncation keeps it")
check("graded on the state of THIS CONTAINER" in out, "it states the asymmetry that motivates it")
check("Verification" in out, "it points at the workflow's own verification section")
check("stop working" not in out.lower(), "it does not tell the model to stop working")
first_len = len(out)
occ, out2 = run(a._execute_commands(["make test"], a._session))
check("[time budget]" in out2 and len(out2) < first_len, "the repeat is the one-line reminder")
check(a._dynwf_phase2_endgames == 2, "events are counted")
check(a._dynwf_phase2_endgame_first_remaining is not None
      and 800 < a._dynwf_phase2_endgame_first_remaining <= 900,
      "the first firing records how much budget was left", a._dynwf_phase2_endgame_first_remaining)

a = make_agent(phase2_endgame_s=1800.0)
a._dynwf_phase = 1
a._dynwf_run_deadline = time.time() + 10.0
occ, out = run(a._execute_commands([], a._session))
check("Time budget" not in out, "INERT in phase 1 even with the budget exhausted")

a = make_agent()
check(a.phase2_endgame_s == 0.0, "OFF by default")
a._dynwf_run_deadline = time.time() + 10.0
occ, out = run(a._execute_commands(["ls"], a._session))
check("Time budget" not in out, "INERT when the reserve is 0 (disabled)")

a = make_agent(phase2_endgame_s=1800.0)
a._dynwf_run_deadline = None
occ, out = run(a._execute_commands(["ls"], a._session))
check("Time budget" not in out, "INERT when no deadline was recorded (no dynamic run started)")

a = make_agent(phase2_endgame_s=1800.0)
a._dynwf_run_deadline = time.time() + 100.0
_real_render = wf_render.render_phase2_endgame
wf_render.render_phase2_endgame = lambda *a_, **k_: (_ for _ in ()).throw(RuntimeError("boom"))
try:
    occ, out = run(a._execute_commands(["ls"], a._session))
finally:
    wf_render.render_phase2_endgame = _real_render
check("Time budget" not in out and "last line of terminal output" in out,
      "a raise inside the nudge costs nothing: ordinary observation, trial continues")
check(a._dynwf_phase2_endgames == 1,
      "the failed attempt is still counted, so a silently broken nudge is visible in the artifacts")

for bad in (20000.0, 25000.0):
    try:
        make_agent(phase2_endgame_s=bad)
        check(False, f"a reserve of {bad:g}s at/above the 20,000 s budget is rejected")
    except ValueError:
        check(True, f"a reserve of {bad:g}s at/above the 20,000 s budget is rejected")
try:
    make_agent(phase2_endgame_s=-1.0)
    check(False, "a negative reserve is rejected")
except ValueError:
    check(True, "a negative reserve is rejected")
try:
    make_agent(phase2_endgame_s=1800.0, phase2_replan_before_compact=True)
    check(True, "construction with both replan and endgame controls on succeeds (the ordering bug)")
except Exception as exc:  # noqa: BLE001
    check(False, "construction with both replan and endgame controls on succeeds (the ordering bug)", repr(exc))

print("\n-- candidate ordering: rung-first replan extraction")

for flag in (False, True):
    a = make_agent(phase2_replan_before_compact=True, phase2_replan_prefer_structured=flag)
    fresh, source, reason = a._dynwf_replan_candidates(replan_response(NEW_SPEC, REASONING))
    check(source == "json_plan",
          f"prefer_structured={flag}: a COMPLETE five-section plan wins under either ranking", source)

a = make_agent(phase2_replan_before_compact=True)
fresh, source, reason = a._dynwf_replan_candidates(replan_response(POOR_SPEC, REASONING))
check(source == "reasoning_content",
      "OFF: a section-poor plan field LOSES to section-rich reasoning (source-ranking ambiguity)", source)

a = make_agent(phase2_replan_before_compact=True, phase2_replan_prefer_structured=True)
fresh, source, reason = a._dynwf_replan_candidates(replan_response(POOR_SPEC, REASONING))
check(source == "json_plan", "ON: the structured `plan` field wins even when it is section-poor", source)
check(fresh.strip() == POOR_SPEC.strip(), "and the accepted text IS the plan field")

a = make_agent(phase2_replan_before_compact=True, phase2_replan_prefer_structured=True)
fresh, source, reason = a._dynwf_replan_candidates(LLMResponse(content="ok", reasoning_content="hmm"))
check(fresh == "" and source == "none", "ON: an unusable response is still rejected (fail closed)", source)

short_plan = json.dumps({"plan": "too short", "commands": [], "task_complete": False})
resp = LLMResponse(content=short_plan + "\n\n" + NEW_SPEC, reasoning_content="")
a = make_agent(phase2_replan_before_compact=True, phase2_replan_prefer_structured=True)
fresh, source, reason = a._dynwf_replan_candidates(resp)
check(fresh.strip().endswith("expect the requested fixture text.") and source in ("assistant_text", "json_plan"),
      "ON: an unusable rung 1 falls through to a usable lower rung", f"{source} {fresh[:40]!r}")

print("\n-- plan witness: the witness")

a = make_agent(phase2_replan_before_compact=True, phase2_replan_prefer_structured=True)
w = a._dynwf_spec_witness(NEW_SPEC)
check(len(w["spec_sha256"]) == 64, "a sha256 of the accepted plan is recorded")
check(w["spec_lines"] == len(NEW_SPEC.splitlines()), "the line count is recorded")
check(1 <= len(w["spec_needles"]) <= 3, "up to three needles are recorded", w["spec_needles"])
check(all(n in NEW_SPEC for n in w["spec_needles"]), "every needle is a verbatim line of the plan")
check(all(len(n) >= 24 for n in w["spec_needles"]), "needles are long enough to be distinctive")
check(w["spec_needles"][0] != NEW_SPEC.splitlines()[0].strip(),
      "needles do NOT come from the head, which the sanitiser reshapes")
check(a._dynwf_spec_witness("")["spec_needles"] == [], "an empty plan yields no needles rather than raising")

a = make_agent(phase2_replan_before_compact=True, phase2_replan_prefer_structured=True)
a._chat = FakeChat(replan_response(NEW_SPEC, REASONING))
a._count_total_tokens = lambda chat: 190000
a._llm = types.SimpleNamespace(get_model_context_limit=lambda: 212992)
PARENT_SUMMARIES.clear()
res = run(a._check_proactive_summarization(a._chat, "solve it", a._session))
row = a._dynwf_phase2_replans[-1]
check(row.get("ok") is True and row.get("spec_source") == "json_plan",
      "end to end: the replan accepted the structured plan", row.get("spec_source"))
check(row.get("prefer_structured") is True, "the row records which ranking was used")
check(len(row.get("spec_sha256", "")) == 64 and row.get("spec_needles"),
      "the row carries the witness for the text that crossed the boundary")
check(a._dynwf_spec_for_reinject.strip() == NEW_SPEC.strip(), "the re-injection now carries the fresh plan")
check(res is not None and PARENT_SUMMARIES, "and the compaction happened immediately afterwards")

print("\n-- the renderer")
doc = wf_render.load_wf_doc(None) if hasattr(wf_render, "load_wf_doc") else {}
full = wf_render.render_phase2_endgame(doc, seconds_left=1234.0, short=False)
short = wf_render.render_phase2_endgame(doc, seconds_left=1234.0, short=True)
check("1,234" in full and "1,234" in short, "the remaining budget is stated in numbers")
check(len(short) < len(full) / 5, "the short form is much shorter")
check(full == wf_render.render_phase2_endgame(doc, seconds_left=1234.0, short=False),
      "the renderer is deterministic")
check("-1" not in wf_render.render_phase2_endgame(doc, seconds_left=-5.0, short=False),
      "a negative remainder is clamped rather than printed")

print()
if FAILS:
    print(f"ENDGAME_AND_EXTRACTION_GATE_FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("ENDGAME_AND_EXTRACTION_GATE_OK")

#!/usr/bin/env python3
"""Synthetic regression tests for dynwf replan before compact."""
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

PARENT_CHECKS: list[tuple] = []
PARENT_SUMMARIES: list[str] = []


async def parent_check(self, chat, original_instruction, session):
    PARENT_CHECKS.append((getattr(self, "_dynwf_phase", 0), original_instruction))
    return None


async def parent_summarize(self, chat, original_instruction, session):
    PARENT_SUMMARIES.append(original_instruction)
    self._summarization_count += 1
    return ("STOCK SUMMARY OF THE WORK", None)


da.Terminus2._check_proactive_summarization = parent_check
da.Terminus2._summarize = parent_summarize


class FakeSession:
    async def get_incremental_output(self):
        return "root@host:/app#"

    async def is_session_alive(self):
        return True


class FakeChat:
    """Only what `_query_llm` touches, plus a hook for the two failure modes."""

    def __init__(self, response=None, raises=None, on_call=None):
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []
        self.messages: list[dict] = []
        self._response = response
        self._raises = raises
        self._on_call = on_call

    async def chat(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if self._on_call is not None:
            self._on_call()
        if self._raises is not None:
            raise self._raises
        return self._response if self._response is not None else LLMResponse(content="")

    def reset_response_chain(self):
        pass


MODEL_INFO = {"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
              "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"}
REPLAN_CONFIG = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
              phase1_spec_reserve_s=600.0, phase1_forced_min_s=90.0, phase1_wrapup_frac=0.60,
              phase1_wrapup_turn_frac=0.60, phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75,
              phase1_context_max_frac=0.25, phase1_single_confirm=True, phase2_same_chat=False,
              phase1_budget_hint=True, phase1_truncation_stop=True, phase1_deadline_grace_s=420.0,
              phase2_reinject_on_summarize=True)


def make_agent(**kw):
    tmp = Path(tempfile.mkdtemp())
    merged = dict(REPLAN_CONFIG)
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


def used(agent, tokens):
    agent._count_total_tokens = lambda chat: tokens


def replan_response(spec, reasoning=""):
    body = json.dumps({"analysis": "restating the plan", "plan": spec, "commands": [],
                       "task_complete": False})
    return LLMResponse(content=body, reasoning_content=reasoning)


ON = dict(phase2_replan_before_compact=True, phase2_replan_free_tokens=32768.0,
          phase2_replan_min_free_tokens=16384.0, phase2_replan_max_tokens=8192)

print("== OFF by default, and every non-firing path delegates to the parent ==")
a = make_agent()
check(a.phase2_replan_before_compact is False, "phase2_replan_before_compact defaults to OFF")
used(a, 205000)
PARENT_CHECKS.clear()
chat = FakeChat()
r = asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
check(r is None and len(PARENT_CHECKS) == 1, "OFF: the parent's own check runs, unchanged")
check(chat.prompts == [] and a._dynwf_phase2_replans == [], "OFF: no replan turn, no rows")
check(a._dynwf_spec_for_reinject == OLD_SPEC, "OFF: the re-injected spec is untouched")

class SyntheticDynamic(Terminus2Dynamic):
    """Exercise inherited defaults without an external training package."""
    pass

rl = SyntheticDynamic(logs_dir=Path(tempfile.mkdtemp()) / "agent", model_name="openai/x", max_turns=250,
                        api_base="http://127.0.0.1:1/v1", suppress_max_turns_warning=True,
                        llm_kwargs={"api_key": "local"}, model_info=MODEL_INFO)
check(rl.phase2_replan_before_compact is False, "SyntheticDynamic (the synthetic subclass) is OFF")

print("== construction gates ==")
try:
    make_agent(phase2_replan_before_compact=True, phase2_replan_free_tokens=16384.0,
               phase2_replan_min_free_tokens=16384.0)
    check(False, "floor >= trigger refuses to construct")
except ValueError:
    check(True, "floor >= trigger refuses to construct")
try:
    make_agent(**{**ON, "phase2_replan_max_tokens": 0})
    check(False, "a non-positive response cap refuses to construct")
except ValueError:
    check(True, "a non-positive response cap refuses to construct")
real_getsource = da.inspect.getsource
try:
    da.inspect.getsource = lambda obj: "def _run_agent_loop(self):\n    pass\n"
    try:
        make_agent(**ON)
        check(False, "FAIL CLOSED: a reworked parent loop refuses to construct")
    except RuntimeError:
        check(True, "FAIL CLOSED: a reworked parent loop refuses to construct")
finally:
    da.inspect.getsource = real_getsource

print("== the trigger, the floor and the ceiling ==")
a = make_agent(**ON)
used(a, 212992 - 40000)
PARENT_CHECKS.clear()
chat = FakeChat()
asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
check(len(PARENT_CHECKS) == 1 and chat.prompts == [] and a._dynwf_phase2_replans == [],
      "free above the trigger: parent only")

a = make_agent(**ON)
used(a, 212992 - 12000)
PARENT_CHECKS.clear()
chat = FakeChat()
asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
row = a._dynwf_phase2_replans[-1]
check(len(PARENT_CHECKS) == 1 and chat.prompts == [], "free below the floor: skipped, parent only")
check(row["ok"] is False and "floor" in row.get("skipped_reason", ""), "the skip is recorded with a reason",
      str(row))

a = make_agent(**ON)
a._dynwf_phase = 1
used(a, 212992 - 20000)
PARENT_CHECKS.clear()
chat = FakeChat()
asyncio.run(a._check_proactive_summarization(chat, "explore it", a._session))
check(len(PARENT_CHECKS) == 1 and a._dynwf_phase2_replans == [], "inert in phase 1")

print("== the firing path: replan, then compact, carrying the FRESH plan ==")
a = make_agent(**ON)
used(a, 212992 - 20000)
PARENT_CHECKS.clear()
PARENT_SUMMARIES.clear()
chat = FakeChat(response=replan_response(NEW_SPEC))
result = asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
row = a._dynwf_phase2_replans[-1]
check(len(chat.prompts) == 1 and "REWRITE YOUR PLAN FIRST" in chat.prompts[0],
      "one replan turn is sent, with the replan prompt")
check("the empty array `[]`" in chat.prompts[0] and "will NOT be executed" in chat.prompts[0],
      "the prompt forbids commands, which cannot run before the compaction")
check(chat.kwargs and chat.kwargs[0].get("max_tokens") == 8192, "the replan response is capped at 8,192",
      str(chat.kwargs[:1]))
check((a._llm_call_kwargs or {}).get("max_tokens") != 8192 or "max_tokens" not in (a._llm_call_kwargs or {}),
      "`_llm_call_kwargs` is restored after the call", str(a._llm_call_kwargs))
check(a._dynwf_spec_for_reinject == NEW_SPEC and a._dynwf_spec_is_refreshed is True,
      "the FRESH plan replaces the re-injected spec")
check(row["ok"] is True and row.get("replaced_spec") is True and row["spec_source"] == "json_plan",
      "the event row records the replacement", str(row))
check(len(PARENT_SUMMARIES) == 1 and result[1] is None
      and result[0].startswith("STOCK SUMMARY OF THE WORK") and NEW_SPEC in result[0],
      "the compaction happens immediately, through the parent's summariser, carrying the fresh plan",
      str(result)[:160])
check(len(PARENT_CHECKS) == 0, "the parent's own check is NOT also run on the firing path")
check(row.get("compacted_immediately") is True and row["at_used_tokens"] == 212992 - 20000,
      "the row carries where it fired", str(row))
check(any(getattr(s, "source", "") == "user" and "REWRITE YOUR PLAN" in (getattr(s, "message", "") or "")
          for s in a._trajectory_steps), "the replan prompt is recorded as a user step")

print("== the plan is extracted from the REPLAN RESPONSE ALONE ==")
a = make_agent(**ON)
used(a, 212992 - 20000)
chat = FakeChat(response=replan_response(NEW_SPEC))
chat.messages = [{"role": "assistant", "content": json.dumps({"plan": OLD_SPEC, "commands": []})}]
asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
check(a._dynwf_spec_for_reinject == NEW_SPEC, "an older plan in the chat cannot win over the new response")

print("== fail-closed paths ==")
a = make_agent(**ON)
used(a, 212992 - 20000)
PARENT_SUMMARIES.clear()
chat = FakeChat(response=LLMResponse(content="Sure, I will keep going."))
result = asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
row = a._dynwf_phase2_replans[-1]
check(a._dynwf_spec_for_reinject == OLD_SPEC and a._dynwf_spec_is_refreshed is False,
      "an unusable response leaves the previous spec in place")
check(row["ok"] is False and row["spec_chars"] == 0 and row.get("spec_reason"),
      "the rejection is recorded with the predicate's reason", str(row))
check(len(PARENT_SUMMARIES) == 1 and result[0].startswith("STOCK SUMMARY OF THE WORK")
      and OLD_SPEC in result[0],
      "the compaction still happens and re-injects the PREVIOUS plan (unrefreshed-plan behaviour, plus one turn)",
      str(result)[:160])

a = make_agent(**ON)
used(a, 212992 - 20000)
PARENT_CHECKS.clear()
PARENT_SUMMARIES.clear()
chat = FakeChat(raises=RuntimeError("transport died"))
result = asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
row = a._dynwf_phase2_replans[-1]
check(result is None and len(PARENT_CHECKS) == 1 and len(PARENT_SUMMARIES) == 0,
      "a failed replan turn falls through to the parent's own check")
check("transport died" in row.get("error", ""), "the failure is recorded", str(row))
check((a._llm_call_kwargs or {}).get("max_tokens") != 8192 or "max_tokens" not in (a._llm_call_kwargs or {}),
      "`_llm_call_kwargs` is restored even when the call raises")

print("== no double compaction when the parent's recovery fires underneath ==")
a = make_agent(**ON)
used(a, 212992 - 20000)
PARENT_CHECKS.clear()
PARENT_SUMMARIES.clear()


def parent_recovered():
    a._summarization_count += 1


chat = FakeChat(response=replan_response(NEW_SPEC), on_call=parent_recovered)
result = asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
row = a._dynwf_phase2_replans[-1]
check(result is None and len(PARENT_SUMMARIES) == 0,
      "we do not compact again over the parent's own handoff")
check("context recovery" in row.get("skipped_reason", ""), "and it is recorded", str(row))
check(len(PARENT_CHECKS) == 0, "nor is the parent's proactive check re-run in that turn")

print("== the event ceiling ==")
a = make_agent(**ON)
used(a, 212992 - 20000)
for i in range(da._PHASE2_REPLAN_MAX):
    asyncio.run(a._check_proactive_summarization(FakeChat(response=replan_response(NEW_SPEC)),
                                                "solve it", a._session))
check(len(a._dynwf_phase2_replans) == da._PHASE2_REPLAN_MAX, "three replans are allowed",
      str(len(a._dynwf_phase2_replans)))
PARENT_CHECKS.clear()
chat = FakeChat(response=replan_response(NEW_SPEC))
asyncio.run(a._check_proactive_summarization(chat, "solve it", a._session))
check(len(PARENT_CHECKS) == 1 and chat.prompts == [] and len(a._dynwf_phase2_replans) == da._PHASE2_REPLAN_MAX,
      "the fourth stands down and stock compaction takes over")

print("== an unmeasurable window is the parent's problem, as before ==")
a = make_agent(**ON)


def boom(chat):
    raise RuntimeError("tokenizer missing")


a._count_total_tokens = boom
PARENT_CHECKS.clear()
asyncio.run(a._check_proactive_summarization(FakeChat(), "solve it", a._session))
check(len(PARENT_CHECKS) == 1 and a._dynwf_phase2_replans == [], "unmeasurable context: parent only")

print("== the re-injection wording ==")
a = make_agent(**ON)
a._dynwf_spec_for_reinject = NEW_SPEC
a._dynwf_spec_is_refreshed = True
handoff, _ = asyncio.run(a._summarize(FakeChat(), "solve it", a._session))
check("UPDATED Specified Workflow you wrote a moment ago" in handoff and NEW_SPEC in handoff,
      "a refreshed plan is described as current")
check("earlier in this session" not in handoff, "and NOT as the plan from earlier in the session")
a = make_agent(**ON)
a._dynwf_spec_for_reinject = OLD_SPEC
handoff2, _ = asyncio.run(a._summarize(FakeChat(), "solve it", a._session))
check("This is the Specified Workflow you wrote for this task earlier in this session." in handoff2,
      "an un-refreshed plan keeps the unrefreshed-plan sentence byte-for-byte")
check(a._dynwf_reinjections == 1, "the re-injection is still counted")

print("== the prompt renderer ==")
doc = wf_render.load_wf(None)
p1 = wf_render.render_phase2_replan(doc, free_tokens=20000, index=1, have_spec=True)
p2 = wf_render.render_phase2_replan(doc, free_tokens=20000, index=2, have_spec=False)
for name in ("**Goal & acceptance criteria**", "**Environment findings**", "**Plan**",
             "**Execution checkpoints**", "**Verification**"):
    check(name in p1, f"the rendered prompt names {name}")
check("20,000 tokens" in p1, "the remaining window is stated in numbers")
check("draft to correct" in p1 and "draft to correct" not in p2,
      "the phase-1 plan is called a draft only when one exists")
check("rewrite number 2" in p2 and "rewrite number" not in p1, "repeat rewrites say which one this is")
check(p1 == wf_render.render_phase2_replan(doc, free_tokens=20000, index=1, have_spec=True),
      "the renderer is deterministic")

print()
if FAILS:
    print(f"REPLAN_BEFORE_COMPACT_GATE_FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("REPLAN_BEFORE_COMPACT_GATE_OK")

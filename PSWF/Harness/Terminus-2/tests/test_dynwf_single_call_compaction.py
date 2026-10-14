#!/usr/bin/env python3
"""Synthetic regression tests for dynwf single call compaction."""
from __future__ import annotations

from synthetic_fixtures import synthetic_plan
import asyncio, inspect, json, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harbor.llms.base import LLMResponse  # noqa: E402
from harbor.agents.terminus_2 import terminus_2 as parent_mod  # noqa: E402
from terminus2_dynwf import dynamic_agent as da  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402

FAILS: list[str] = []


def check(cond, label, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  {extra}"))
    if not cond:
        FAILS.append(label)


def plan_text(marker: str) -> str:
    return synthetic_plan(marker)


SPEC = synthetic_plan()
STOCK_CALLS: list[str] = []


async def parent_summarize(self, chat, oi, session):
    """Stands in for upstream's THREE-call `_summarize`, and charges the count the way upstream does."""
    STOCK_CALLS.append(oi)
    self._summarization_count += 1
    return ("STOCK HANDOFF: three legs ran", ["r1", "r2", "r3"])


PARENT_SUMMARIZE_SRC = inspect.getsource(parent_mod.Terminus2._summarize)

da.Terminus2._summarize = parent_summarize


class FakeSession:
    def __init__(self, pane="root@host:/app# make -j8"):
        self.pane = pane
        self.captures = 0

    async def capture_pane(self, capture_entire=False):
        self.captures += 1
        if isinstance(self.pane, Exception):
            raise self.pane
        return self.pane


class FakeChat:
    def __init__(self, n_messages=9, boom_on_reset=False):
        self._messages = [{"role": "system", "content": "PHASE-2 PROMPT with the injected plan"}]
        self._messages += [{"role": "user" if i % 2 else "assistant", "content": f"m{i}"}
                           for i in range(1, n_messages)]
        self.resets = 0
        self.boom_on_reset = boom_on_reset

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        self.resets += 1
        if self.boom_on_reset:
            raise RuntimeError("reset blew up AFTER the summary call")


MODEL_INFO = {"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
              "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"}
BASE = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
            phase1_spec_reserve_s=600.0, phase1_forced_min_s=90.0, phase1_wrapup_frac=0.60,
            phase1_wrapup_turn_frac=0.60, phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75,
            phase1_context_max_frac=0.25, phase1_single_confirm=True, phase2_same_chat=False,
            phase1_budget_hint=True, phase1_truncation_stop=True, phase1_deadline_grace_s=420.0,
            phase2_reinject_on_summarize=True, phase2_replan_before_compact=True,
            phase2_replan_prefer_structured=True, phase2_endgame_s=1800.0, replan_trusted_dump=True)

SUBAGENT_CALLS: list[dict] = []


def build(threshold=10000.0, summary="SYNTHETIC SUMMARY: the fixture file still needs its requested text", subagent_exc=None,
          prepare_exc=None, **over):
    tmp = Path(tempfile.mkdtemp())
    logs = tmp / "agent"
    logs.mkdir(parents=True, exist_ok=True)
    kw = dict(BASE)
    kw.update(over)
    a = Terminus2Dynamic(logs_dir=logs, model_name="openai/x", max_turns=400,
                         api_base="http://127.0.0.1:1/v1", temperature=1.0, suppress_max_turns_warning=True,
                         model_info=MODEL_INFO, llm_kwargs={"api_key": "local"},
                         trusted_root=str(tmp / "trusted"), compact_fast_below_s=threshold, **kw)

    async def run_subagent(prompt, message_history, steps, session_id, agent_name, filename_suffix,
                           summary_text, subagent_name_for_logging="subagent"):
        SUBAGENT_CALLS.append({"session_id": session_id, "agent_name": agent_name,
                               "suffix": filename_suffix, "prompt": prompt,
                               "history_len": len(message_history)})
        if subagent_exc is not None:
            raise subagent_exc
        return LLMResponse(content=summary, reasoning_content=""), f"ref-{filename_suffix}"

    def prepare(n):
        if prepare_exc is not None:
            raise prepare_exc
        return ([], 1)

    a._run_subagent = run_subagent
    a._prepare_copied_trajectory_steps = prepare
    a._llm = types.SimpleNamespace(get_model_context_limit=lambda: 212992)
    return a


def summarize(a, chat, session):
    return asyncio.new_event_loop().run_until_complete(a._summarize(chat, "solve the task", session))


print("-- construction and validation (initialization ordering)")
try:
    a = build(threshold=10000.0)
    check(a.compact_fast_below_s == 10000.0, "constructing with a real threshold does not raise")
except Exception as exc:  # noqa: BLE001
    check(False, "constructing with a real threshold does not raise", f"{type(exc).__name__}: {exc}")
for bad, why in ((20000.0, "equal to agent_budget_s"), (25000.0, "above agent_budget_s"), (-1.0, "negative")):
    try:
        build(threshold=bad)
        check(False, f"a threshold {why} is rejected")
    except ValueError:
        check(True, f"a threshold {why} is rejected")
    except Exception as exc:  # noqa: BLE001
        check(False, f"a threshold {why} is rejected with ValueError", f"{type(exc).__name__}: {exc}")
check(build(threshold=0.0).compact_fast_below_s == 0.0, "0 is accepted and means OFF")

print("\n-- the clock test fails closed")
import time as _time

a = build(threshold=10000.0)
a._dynwf_phase = 2
a._dynwf_run_deadline = _time.time() + 5000
check(a._dynwf_compact_fast_due()[0] is True, "phase 2, 5,000 s left, threshold 10,000 -> FIRES")
a._dynwf_run_deadline = _time.time() + 15000
check(a._dynwf_compact_fast_due()[0] is False, "phase 2, 15,000 s left -> stock")
a._dynwf_run_deadline = _time.time() + 5000
a._dynwf_phase = 1
check(a._dynwf_compact_fast_due()[0] is False, "phase 1 never fires (the boundary compaction stays stock)")
a._dynwf_phase = 2
a._dynwf_run_deadline = None
check(a._dynwf_compact_fast_due()[0] is False, "no deadline recorded -> stock")
a._dynwf_run_deadline = _time.time() - 10
check(a._dynwf_compact_fast_due()[0] is False, "already past the wall -> stock (no clock left to buy)")
off = build(threshold=0.0)
off._dynwf_phase = 2
off._dynwf_run_deadline = _time.time() + 100
check(off._dynwf_compact_fast_due()[0] is False, "knob OFF -> stock even at 100 s left")

print("\n-- the fast path spends exactly one call")
SUBAGENT_CALLS.clear(); STOCK_CALLS.clear()
a = build(threshold=10000.0)
a._dynwf_phase = 2
a._dynwf_run_deadline = _time.time() + 4000
a._dynwf_spec_for_reinject = SPEC
chat, sess = FakeChat(), FakeSession()
history_length = len(chat.messages)
handoff, refs = summarize(a, chat, sess)
check(len(SUBAGENT_CALLS) == 1, "exactly ONE subagent call", len(SUBAGENT_CALLS))
check(not STOCK_CALLS, "the parent's three-call `_summarize` was NOT used")
check(SUBAGENT_CALLS[0]["suffix"] == "summary" and
      SUBAGENT_CALLS[0]["agent_name"] == "terminus-2-summarization-summary",
      "it is the parent's own summary leg, so the `-summary` artifact is still written")
check(SUBAGENT_CALLS[0]["history_len"] == history_length, "the complete pre-compaction history was passed")
check(a._summarization_count == 1, "summarization_count advanced by exactly 1", a._summarization_count)
check(refs == ["ref-summary"], "one trajectory ref is returned", refs)
check(sess.captures == 1, "the terminal screen was captured once (free, no model call)")
check("SYNTHETIC SUMMARY" in handoff, "the handoff carries the summary")
check("root@host:/app# make -j8" in handoff, "the handoff carries the current terminal screen")
check("follow the spec to interact with the terminal" in handoff,
      "the handoff keeps the response-format instruction")
check("can no longer ask questions" in handoff, "the handoff closes the question channel, as upstream's does")
check(SPEC in handoff, "the re-injected Specified Workflow is appended (the same injection contract applies)")
check(a._dynwf_reinjections == 1, "the re-injection is counted")
check(chat.messages == [{"role": "system", "content": "PHASE-2 PROMPT with the injected plan"}],
      "history was rebuilt to messages[0] only -- no synthesized assistant turn", chat.messages)
check(chat.resets == 1, "the response chain was reset")
row = a._dynwf_compactions[-1]
check(row["mode"] == "fast" and row["legs"] == 1 and row["legs_saved"] == 2,
      "the event row records mode/legs/legs_saved", row)
check(row["threshold_s"] == 10000.0 and 3900 < row["at_remaining_s"] < 4100,
      "the row records the threshold and the remaining budget it fired at", row)

print("\n-- above the threshold it is the stock three-call path, unchanged")
SUBAGENT_CALLS.clear(); STOCK_CALLS.clear()
b = build(threshold=10000.0)
b._dynwf_phase = 2
b._dynwf_run_deadline = _time.time() + 16000
b._dynwf_spec_for_reinject = SPEC
hb, rb = summarize(b, FakeChat(), FakeSession())
check(len(STOCK_CALLS) == 1 and not SUBAGENT_CALLS, "the parent's `_summarize` ran and the fast path did not")
check("STOCK HANDOFF" in hb and SPEC in hb, "the stock handoff still gets the plan appended")
check(b._dynwf_compactions[-1]["mode"] == "stock" and b._dynwf_compactions[-1]["legs"] == 3,
      "the stock compaction is recorded too, so the meta counts BOTH kinds", b._dynwf_compactions[-1])

print("\n-- an exception BEFORE the call falls back; AFTER the call it propagates")
SUBAGENT_CALLS.clear(); STOCK_CALLS.clear()
c = build(threshold=10000.0, prepare_exc=RuntimeError("cannot copy trajectory steps"))
c._dynwf_phase = 2
c._dynwf_run_deadline = _time.time() + 4000
c._dynwf_spec_for_reinject = SPEC
hc, rc = summarize(c, FakeChat(), FakeSession())
check(not SUBAGENT_CALLS and len(STOCK_CALLS) == 1,
      "a failure before the summary call spends nothing and falls back to the stock path")
check("STOCK HANDOFF" in hc, "the fallback handoff is the stock one")
check(c._summarization_count == 1, "the count is exactly 1, not 2 -- the aborted fast path did not charge one",
      c._summarization_count)
check(c._dynwf_compactions[-1]["mode"] == "stock", "and the row says stock")

SUBAGENT_CALLS.clear(); STOCK_CALLS.clear()
d = build(threshold=10000.0)
d._dynwf_phase = 2
d._dynwf_run_deadline = _time.time() + 4000
try:
    summarize(d, FakeChat(boom_on_reset=True), FakeSession())
    check(False, "a failure AFTER the summary call propagates")
except RuntimeError as exc:
    check("AFTER the summary call" in str(exc), "a failure AFTER the summary call propagates")
    check(not STOCK_CALLS, "and it does NOT re-run the stock path, which would double-charge the trial")

print("\n-- a pane-capture failure is survivable, not fatal")
SUBAGENT_CALLS.clear()
e = build(threshold=10000.0)
e._dynwf_phase = 2
e._dynwf_run_deadline = _time.time() + 4000
he, _ = summarize(e, FakeChat(), FakeSession(pane=RuntimeError("tmux gone")))
check("terminal screen unavailable" in he, "the handoff records that the screen could not be read")
check(len(SUBAGENT_CALLS) == 1, "and still spends only the one call")

print("\n-- upstream drift: the duplicated summary prompt still matches the installed parent's")
psrc = PARENT_SUMMARIZE_SRC
for anchor in ("You are about to hand off your work to another AI agent.",
               "**Major Actions Completed**", "**Current Status** - Exactly where you are in the task",
               "Be comprehensive and detailed. The next agent needs to understand everything"):
    check(anchor in psrc, f"the installed parent still contains: {anchor[:44]}...")
fsrc = inspect.getsource(Terminus2Dynamic._dynwf_fast_summarize)
for anchor in ("You are about to hand off your work to another AI agent.",
               "**Major Actions Completed**"):
    check(anchor in fsrc, f"and so does the fast path: {anchor[:44]}...")

print()
if FAILS:
    print(f"SINGLE_CALL_COMPACTION_GATE_FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("SINGLE_CALL_COMPACTION_GATE_OK")

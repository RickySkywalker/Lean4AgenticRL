#!/usr/bin/env python3
"""Synthetic regression tests for dynwf truncation and grace."""
from __future__ import annotations

from synthetic_fixtures import synthetic_plan
import asyncio, json, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harbor.llms.base import LLMResponse  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402
from terminus2_dynwf import wf_render  # noqa: E402
from terminus2_dynwf import dynamic_agent as da  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402


FAILS: list[str] = []


def check(cond, label, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  {extra}"))
    if not cond:
        FAILS.append(label)


SPEC = synthetic_plan()
STOCK_RETRY = ("ERROR!! NONE of the actions you just requested were performed because you exceeded 32768 "
               "tokens. Your outputs must be less than 32768 tokens. Re-issue this request, breaking it "
               "into chunks each of which is less than 32768 tokens.")


class FakeSession:
    def __init__(self):
        self.sent: list[str] = []
        self.n = 0

    async def get_incremental_output(self):
        self.n += 1
        return f"root@host:/app# (read {self.n})"

    async def is_session_alive(self):
        return True

    async def send_keys(self, keystrokes, block=False, min_timeout_sec=0.0):
        self.sent.append(keystrokes)


class FakeChat:
    """Only what `_query_llm` touches."""

    def __init__(self):
        self.prompts: list[str] = []
        self.messages: list[dict] = []

    async def chat(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return LLMResponse(content=json.dumps(
            {"analysis": "done", "plan": SPEC, "commands": [], "task_complete": True}))

    def reset_response_chain(self):
        pass


def make_agent(tmp, **kw):
    a = Terminus2Dynamic(logs_dir=tmp / "agent", model_name="openai/x", max_turns=250,
                         api_base="http://127.0.0.1:1/v1", temperature=1.0, suppress_max_turns_warning=True,
                         model_info={"max_input_tokens": 212992, "max_output_tokens": 32768,
                                     "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                                     "litellm_provider": "openai", "mode": "chat"},
                         llm_kwargs={"api_key": "local"}, trusted_root=str(tmp / "trusted"), **kw)
    (tmp / "agent").mkdir(parents=True, exist_ok=True)
    a._session = FakeSession()
    return a


PHASE1_CONFIG = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
              phase1_spec_reserve_s=600.0, phase1_forced_min_s=90.0, phase1_wrapup_frac=0.60,
              phase1_wrapup_turn_frac=0.60, phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75,
              phase1_context_max_frac=0.25, phase1_single_confirm=True, phase2_same_chat=False,
              phase1_budget_hint=True)
FAST = dict(agent_budget_s=8.0, phase1_frac=0.5, phase1_min_s=0.5, phase1_spec_reserve_s=2.5,
            phase1_forced_min_s=0.5, phase1_max_turns=30, phase2_same_chat=False,
            phase1_single_confirm=True)


class Cmd(types.SimpleNamespace):
    pass


print("== the truncation stop is OFF by default and the prompt is byte-inert ==")
tmp = Path(tempfile.mkdtemp())
a = make_agent(tmp, **PHASE1_CONFIG)
check(a.phase1_truncation_stop is False and a.phase1_deadline_grace_s == 0.0,
      "both controls default to OFF", f"{a.phase1_truncation_stop} {a.phase1_deadline_grace_s}")
a._dynwf_phase = 1
chat = FakeChat()
asyncio.run(a._query_llm(chat, STOCK_RETRY))
check(chat.prompts == [STOCK_RETRY], "OFF: the stock retry prompt reaches the model unchanged")
check(a._dynwf_phase1_truncations == 0 and a._dynwf_exploration_closed is False,
      "OFF: nothing is counted and exploration is not closed")
hint_off = wf_render.render_budget_hint(explore_s=1344, reserve_s=600, phase1_max_turns=30, hard_stop_s=940,
                                        hard_stop_turns=23, nudge_s=806, nudge_turns=18, context_max_frac=0.25)
check("cut off at that limit" not in hint_off and "allowed up to" not in hint_off,
      "OFF: the budget-hint paragraph is byte-unchanged")
class SyntheticDynamic(Terminus2Dynamic):
    """Exercise inherited defaults without an external training package."""
    pass

rl = SyntheticDynamic(logs_dir=Path(tempfile.mkdtemp()) / "agent", model_name="openai/x", max_turns=250,
                        api_base="http://127.0.0.1:1/v1", suppress_max_turns_warning=True,
                        llm_kwargs={"api_key": "local"},
                        model_info={"max_input_tokens": 212992, "max_output_tokens": 32768,
                                    "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                                    "litellm_provider": "openai", "mode": "chat"})
check(rl.phase1_truncation_stop is False and rl.phase1_deadline_grace_s == 0.0,
      "SyntheticDynamic (the synthetic subclass) constructs with both OFF")

print("== truncation stop ON: the retry becomes the deliverable request and exploration closes ==")
tmp = Path(tempfile.mkdtemp())
a = make_agent(tmp, phase1_truncation_stop=True, **PHASE1_CONFIG)
a._dynwf_phase = 1
a._dynwf_explore_deadline = None
chat = FakeChat()
asyncio.run(a._query_llm(chat, STOCK_RETRY))
sent = chat.prompts[0]
check(sent.startswith(wf_render.TRUNCATION_STOP_HEADER), "the retry prompt is replaced by the truncation stop")
check("Re-issue this request" not in sent and "breaking it into chunks" not in sent,
      "the stock 're-issue this request in chunks' instruction is gone")
check("commands`: the empty array `[]`" in sent or "commands" in sent,
      "the asset's phase1_forced_spec deliverable format is carried")
check("about 3,000 characters" not in sent, "the first swap does NOT ask for a short plan")
check(a._dynwf_phase1_truncations == 1 and a._dynwf_truncation_swaps == 1, "counted once")
check(a._dynwf_exploration_closed is True, "exploration is closed")
check(a._pending_completion is True, "single-confirm: the next task_complete ends the phase")
due, _, _ = a._dynwf_hard_stop_due()
check(due is True, "the hard-stop path now refuses command batches, whatever the clock says")
ok, out = asyncio.run(a._execute_commands([Cmd(keystrokes="ls\n", duration_sec=0.1)], a._session))
check(a._session.sent == [], "a phase-1 command batch after the closure is NOT executed")
check("EXPLORATION IS CLOSED" in out, "and the observation is the closing instruction", out[:80])
chat2 = FakeChat()
asyncio.run(a._query_llm(chat2, STOCK_RETRY))
check("about 3,000 characters" in chat2.prompts[0], "the SECOND truncation asks for a shorter workflow")
for _ in range(3):
    asyncio.run(a._query_llm(FakeChat(), STOCK_RETRY))
check(a._dynwf_truncation_swaps == da._TRUNCATION_SWAP_MAX,
      f"swaps are bounded at {da._TRUNCATION_SWAP_MAX}", str(a._dynwf_truncation_swaps))
chat3 = FakeChat()
asyncio.run(a._query_llm(chat3, STOCK_RETRY))
check(chat3.prompts == [STOCK_RETRY], "past the ceiling the stock behaviour is restored (escape hatch)")
a2 = make_agent(Path(tempfile.mkdtemp()), phase1_truncation_stop=True, **PHASE1_CONFIG)
a2._dynwf_phase = 2
chat4 = FakeChat()
asyncio.run(a2._query_llm(chat4, STOCK_RETRY))
check(chat4.prompts == [STOCK_RETRY] and a2._dynwf_phase1_truncations == 0,
      "inert in phase 2: a truncated solving response is still re-issued")

print("== truncation stop ON: it refuses to construct if harbor reworded the retry ==")
original = da.inspect.getsource
try:
    da.inspect.getsource = lambda _f: "def _query_llm(self): pass"
    try:
        make_agent(Path(tempfile.mkdtemp()), phase1_truncation_stop=True, **PHASE1_CONFIG)
        check(False, "construction fails closed when the parent message is missing")
    except RuntimeError as exc:
        check("pinned Terminus2 output-length recovery" in str(exc),
              "construction fails closed when the parent message is missing", str(exc))
finally:
    da.inspect.getsource = original

print("== the budget hint states both new rules, rendered from the knobs ==")
hint_on = wf_render.render_budget_hint(explore_s=1344, reserve_s=600, phase1_max_turns=30, hard_stop_s=940,
                                       hard_stop_turns=23, nudge_s=806, nudge_turns=18, context_max_frac=0.25,
                                       truncation_stop=True, grace_s=420)
check("cut off at that limit" in hint_on and "CLOSES exploration" in hint_on,
      "the output-limit rule is stated")
check("allowed up to 420 more seconds" in hint_on, "the grace is stated with its configured number")
check("ONE forced turn of at most 600 seconds" in hint_on, "the reserve clause is still there")


class GraceLoop:
    """Stands in for _run_agent_loop: explores past the deadline, then emits the plan."""

    def __init__(self, agent, *, sleep_s: float, tell: bool, plan: bool = True):
        self.agent, self.sleep_s, self.tell, self.plan = agent, sleep_s, tell, plan
        self.closed_during: list[bool] = []
        self.kinds: list[str] = []

    async def __call__(self, initial_prompt, chat, original_instruction=""):
        self.kinds.append("explore" if "PHASE 1 of 2" in initial_prompt else
                          ("forced" if "EXPLORATION IS CLOSED" in initial_prompt else "solve"))
        if self.kinds[-1] != "explore":
            self.agent._api_request_times.append(1.0)
            self.agent._n_episodes = 1
            return
        self.agent._n_episodes = 2
        self.agent._api_request_times.append(1.0)
        if self.tell:
            self.agent._dynwf_wrapups = 1
        await asyncio.sleep(self.sleep_s)
        self.closed_during.append(self.agent._dynwf_exploration_closed)
        if self.plan:
            chat._messages.extend([{"role": "user", "content": "t"}, {"role": "assistant", "content": json.dumps(
                {"analysis": "done", "plan": SPEC, "commands": [], "task_complete": True})}])
            self.agent._api_request_times.append(1.0)
            self.agent._pending_completion = True


def run_grace(**kw):
    loop_kw = {k: kw.pop(k) for k in ("sleep_s", "tell", "plan") if k in kw}
    tmp = Path(tempfile.mkdtemp())
    a = make_agent(tmp, **kw)
    loop = GraceLoop(a, **loop_kw)
    a._run_agent_loop = loop
    asyncio.run(a.run("Make the failing tests pass.", None, AgentContext()))
    return a, loop, json.loads((tmp / "agent" / "dynwf_meta.json").read_text())

print("== deadline grace: an in-flight deliverable is allowed to land ==")
a, loop, meta = run_grace(phase1_deadline_grace_s=1.5, sleep_s=2.2, tell=True, **FAST)
g = meta["phase1_deadline_grace"]
check(meta["phase1_explore_budget_s"] == 1.5 and meta["phase1_reserve_s"] == 2.5,
      "the slice arithmetic is unchanged (explore 1.5 / reserve 2.5)",
      f"{meta['phase1_explore_budget_s']} / {meta['phase1_reserve_s']}")
check(g["granted"] is True and g["landed"] is True, "grace granted and the response landed", str(g))
check(meta["phase1_stopped_by"] == "completed", "a landed deliverable counts as completed, not a deadline")
check(meta["have_spec"] is True and meta["phase2_spec_injected"] is True, "the plan reached phase 2")
check(g["available_s"] == 1.5, "grace is clamped by the reserve minus the forced-turn floor", str(g))
check(loop.closed_during == [True], "exploration was forced CLOSED for the grace window", str(loop.closed_during))
check(meta["phase1_exploration_closed_early"] is True, "recorded in the metadata for the audit")
check(meta["phase1_elapsed_s"] <= meta["phase1_budget_s"] + 0.6,
      "phase 1 still fits its own ceiling, so phase 2 loses nothing",
      f"{meta['phase1_elapsed_s']} vs {meta['phase1_budget_s']}")

print("== deadline grace: NOT granted when the model was never told to deliver ==")
a, loop, meta = run_grace(phase1_deadline_grace_s=1.5, sleep_s=2.2, tell=False, **FAST)
g = meta["phase1_deadline_grace"]
check(g["granted"] is False and g.get("already_instructed") is False,
      "grace withheld: the nudge/hard stop had not fired", str(g))
check(meta["phase1_stopped_by"] == "phase_deadline", "the deadline cancels exactly as it did before")
check(loop.closed_during == [], "the exploration task really was cancelled", str(loop.closed_during))

print("== deadline grace: expiring leaves the forced plan turn possible ==")
a, loop, meta = run_grace(phase1_deadline_grace_s=1.5, sleep_s=9.0, tell=True, **FAST)
g = meta["phase1_deadline_grace"]
check(g["granted"] is True and g["landed"] is False, "grace expired without a deliverable", str(g))
check(meta["phase1_stopped_by"] == "phase_deadline", "so the phase is recorded as a deadline")
check(meta["phase1_forced_spec"]["attempted"] is True, "the forced plan turn still ran",
      str(meta["phase1_forced_spec"]))
check("forced" in loop.kinds and loop.kinds[-1] == "solve", "and phase 2 still started", str(loop.kinds))

print("== OFF: no grace key surprises, deadline behaviour identical to budget-hint mode ==")
a, loop, meta = run_grace(sleep_s=2.2, tell=True, **FAST)
check(meta["phase1_deadline_grace"]["granted"] is False and meta["phase1_stopped_by"] == "phase_deadline",
      "with the knob at 0 the deadline still cancels the in-flight call",
      str(meta["phase1_deadline_grace"]))

print()
print("TRUNCATION_AND_GRACE_GATE_OK" if not FAILS else f"TRUNCATION_AND_GRACE_GATE_FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)

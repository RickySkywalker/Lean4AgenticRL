#!/usr/bin/env python3
"""Synthetic regression tests for dynwf budget hint."""
from __future__ import annotations

from synthetic_fixtures import synthetic_plan
import asyncio, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harbor.models.agent.context import AgentContext  # noqa: E402
from terminus2_dynwf import wf_render  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402


FAILS: list[str] = []
def check(cond, label, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  {extra}"))
    if not cond: FAILS.append(label)

SPEC = synthetic_plan()

class FakeSession:
    async def get_incremental_output(self): return "root@host:/app# "
    async def is_session_alive(self): return True
    async def send_keys(self, *a, **k): return None

def make_agent(tmp, **kw):
    a = Terminus2Dynamic(logs_dir=tmp / "agent", model_name="openai/x", max_turns=250, api_base="http://127.0.0.1:1/v1",
                         temperature=1.0, suppress_max_turns_warning=True,
                         model_info={"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
                                     "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"},
                         llm_kwargs={"api_key": "local"}, trusted_root=str(tmp / "trusted"), **kw)
    (tmp / "agent").mkdir(parents=True, exist_ok=True)
    a._session = FakeSession()
    return a

class Recorder:
    def __init__(self, agent): self.agent, self.prompts = agent, []
    async def __call__(self, initial_prompt, chat, original_instruction=""):
        self.prompts.append(initial_prompt)
        if "PHASE 1 of 2" in initial_prompt:
            chat._messages.extend([{"role": "user", "content": "t"}, {"role": "assistant", "content": json.dumps(
                {"analysis": "done", "plan": SPEC, "commands": [], "task_complete": True})}])
            self.agent._api_request_times.append(1.0); self.agent._n_episodes = 1
            self.agent._pending_completion = True
        else:
            self.agent._api_request_times.append(1.0); self.agent._n_episodes = 1

def run_case(**kw):
    tmp = Path(tempfile.mkdtemp()); a = make_agent(tmp, **kw); rec = Recorder(a); a._run_agent_loop = rec
    asyncio.run(a.run("Make the failing tests pass.", None, AgentContext()))
    meta = json.loads((tmp / "agent" / "dynwf_meta.json").read_text())
    return a, rec, meta

BUDGET_CONFIG = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
              phase1_spec_reserve_s=600.0, phase1_wrapup_frac=0.60, phase1_wrapup_turn_frac=0.60,
              phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75, phase1_context_max_frac=0.25,
              phase1_single_confirm=True, phase2_same_chat=False)

print("== OFF by default: byte-inert ==")
a, rec, meta = run_case(**BUDGET_CONFIG)
p1 = rec.prompts[0]
check(wf_render.BUDGET_HINT_HEADER not in p1, "no hint in the phase-1 prompt when the knob is absent")
check(meta.get("phase1_budget_hint") is False, "meta records phase1_budget_hint=False", str(meta.get("phase1_budget_hint")))
expected_instr = wf_render.render_phase1(a._wf_doc, "Make the failing tests pass.", 30, wf_render.CONFIRM_CLAIM_SINGLE)
check(expected_instr in p1 and "Phase-boundary handoff:" in p1, "phase-1 prompt == render_phase1 + the fresh-chat handoff paragraph (unchanged)")
check(meta["phase1_explore_budget_s"] == 1344.0 and meta["phase1_reserve_s"] == 600.0, "budget arithmetic unchanged by the move (1344 / 600)",
      f"{meta['phase1_explore_budget_s']} / {meta['phase1_reserve_s']}")
class SyntheticDynamic(Terminus2Dynamic):
    """Exercise inherited defaults without an external training package."""
    pass

tmp = Path(tempfile.mkdtemp())
rl = SyntheticDynamic(logs_dir=tmp / "agent", model_name="openai/x", max_turns=250, api_base="http://127.0.0.1:1/v1",
                        suppress_max_turns_warning=True, llm_kwargs={"api_key": "local"},
                        model_info={"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
                                    "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"})
check(rl.phase1_budget_hint is False, "SyntheticDynamic (the synthetic subclass) constructs with the hint OFF")

print("== ON with the budget-hint configuration: the enforced numbers, phase 1 only ==")
a, rec, meta = run_case(phase1_budget_hint=True, **BUDGET_CONFIG)
p1, p2 = rec.prompts[0], rec.prompts[1]
check(wf_render.BUDGET_HINT_HEADER in p1, "hint present in the phase-1 prompt")
check(wf_render.BUDGET_HINT_HEADER not in p2, "hint ABSENT from the phase-2 prompt")
check(meta.get("phase1_budget_hint") is True, "meta records phase1_budget_hint=True")
for needle in ["about 1344 seconds in total, and at most 30 turns", "after about 806 seconds or 18 turns",
               "From about 940 seconds or about 23 turns", "REFUSED and NOT executed", "past 25% of the context window",
               "ONE forced turn of at most 600 seconds", "aim for well under 806 seconds and 18 turns"]:
    check(needle in p1, f"renders the configured number: {needle!r}")
check(p1.index("Phase-boundary handoff:") < p1.index(wf_render.BUDGET_HINT_HEADER), "hint comes after the handoff paragraph (asset bytes untouched)")
check(meta["phase2_spec_injected"] is True and SPEC.splitlines()[2] in p2, "phase-2 injection unaffected")

print("== ON with the hard stop OFF: no refusal clause is invented ==")
kw = dict(BUDGET_CONFIG); kw.update(phase1_hard_stop_frac=0.0, phase1_hard_stop_turn_frac=0.0, phase1_context_max_frac=0.0)
a, rec, meta = run_case(phase1_budget_hint=True, **kw)
p1 = rec.prompts[0]
check("REFUSED" not in p1 and "context window" not in p1, "no refusal / context clause when those controls are off")
check("after about 806 seconds or 18 turns" in p1 and "600 seconds" in p1, "reminder and reserve clauses still rendered")

print()
print("BUDGET_HINT_GATE_OK" if not FAILS else f"BUDGET_HINT_GATE_FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)

#!/usr/bin/env python3
"""Synthetic regression tests for dynwf trusted dump."""
from __future__ import annotations

from synthetic_fixtures import synthetic_plan
import asyncio, hashlib, json, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harbor.llms.base import LLMResponse  # noqa: E402
from terminus2_dynwf import dynamic_agent as da  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402


FAILS: list[str] = []


def check(cond, label, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  {extra}"))
    if not cond:
        FAILS.append(label)


def plan_text(marker: str) -> str:
    return synthetic_plan(marker)


NEW_SPEC = plan_text("Finish")
SUMMARIES: list[str] = []


async def parent_check(self, chat, oi, session):
    return None


async def parent_summarize(self, chat, oi, session):
    SUMMARIES.append(oi)
    self._summarization_count += 1
    return ("STOCK SUMMARY", None)


da.Terminus2._check_proactive_summarization = parent_check
da.Terminus2._summarize = parent_summarize


class FakeSession:
    async def get_incremental_output(self):
        return "root@host:/app#"

    async def is_session_alive(self):
        return True


class FakeChat:
    def __init__(self, response):
        self.prompts, self.kwargs, self.messages = [], [], []
        self._r = response

    async def chat(self, prompt, **kw):
        self.prompts.append(prompt)
        self.kwargs.append(kw)
        return self._r

    def reset_response_chain(self):
        pass


MODEL_INFO = {"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
              "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"}
BASE = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
            phase1_spec_reserve_s=600.0, phase1_forced_min_s=90.0, phase1_wrapup_frac=0.60,
            phase1_wrapup_turn_frac=0.60, phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75,
            phase1_context_max_frac=0.25, phase1_single_confirm=True, phase2_same_chat=False,
            phase1_budget_hint=True, phase1_truncation_stop=True, phase1_deadline_grace_s=420.0,
            phase2_reinject_on_summarize=True, phase2_replan_before_compact=True,
            phase2_replan_prefer_structured=True, phase2_endgame_s=1800.0)


def run_replan(dump: bool):
    tmp = Path(tempfile.mkdtemp())
    logs = tmp / "agent"
    logs.mkdir(parents=True, exist_ok=True)
    trusted = tmp / "trusted"
    a = Terminus2Dynamic(logs_dir=logs, model_name="openai/x", max_turns=400,
                         api_base="http://127.0.0.1:1/v1", temperature=1.0, suppress_max_turns_warning=True,
                         model_info=MODEL_INFO, llm_kwargs={"api_key": "local"},
                         trusted_root=str(trusted), replan_trusted_dump=dump, **BASE)
    a._session = FakeSession()
    a._dynwf_phase = 2
    a._dynwf_spec_for_reinject = plan_text("Fix")
    a._chat = FakeChat(LLMResponse(content=json.dumps(
        {"analysis": "restate", "plan": NEW_SPEC, "commands": [], "task_complete": False}), reasoning_content=""))
    a._count_total_tokens = lambda chat: 190000
    a._llm = types.SimpleNamespace(get_model_context_limit=lambda: 212992)
    asyncio.new_event_loop().run_until_complete(
        a._check_proactive_summarization(a._chat, "solve it", a._session))
    return a, logs, trusted


print("\n-- the dump itself")
a, logs, trusted = run_replan(dump=True)
row = a._dynwf_phase2_replans[-1]
check(row.get("ok") is True, "the rewrite was accepted")
check(row.get("spec_dump") == "specified_workflow_refreshed_1.md", "the row names the dumped file",
      row.get("spec_dump"))
dumped = list(trusted.rglob("specified_workflow_refreshed_*.md"))
check(len(dumped) == 1, f"exactly one dump exists under the trusted root", [str(p) for p in dumped])
if dumped:
    body = dumped[0].read_text()
    check(body == NEW_SPEC, "the dumped bytes ARE the accepted plan")
    check(hashlib.sha256(body.encode()).hexdigest() == row.get("spec_sha256"),
          "the dump's sha256 matches the witness recorded in the row -- the audit can now diff, not infer")

print("\n-- containment: it must NOT be written where the container can see it")
in_logs = list(Path(logs).rglob("specified_workflow_refreshed_*.md"))
check(not in_logs, "nothing was written under logs_dir, which is bind-mounted into the task container",
      [str(p) for p in in_logs])

print("\n-- inertness: with the knob OFF, and behaviour identical either way")
b, logs_b, trusted_b = run_replan(dump=False)
rb = b._dynwf_phase2_replans[-1]
check(not list(trusted_b.rglob("specified_workflow_refreshed_*.md")), "OFF: no dump is written")
check("spec_dump" not in rb, "OFF: the row carries no dump bookkeeping")
ignore = {"spec_dump", "spec_dump_error", "elapsed_s"}
diff = {k: (row.get(k), rb.get(k)) for k in set(row) | set(rb)
        if k not in ignore and row.get(k) != rb.get(k)}
check(not diff, "every other recorded field is identical with the knob on and off (telemetry inertness property)", diff)
check(a._dynwf_spec_for_reinject == b._dynwf_spec_for_reinject,
      "the plan handed to the re-injection is the same either way")

print()
if FAILS:
    print(f"TRUSTED_DUMP_GATE_FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("TRUSTED_DUMP_GATE_OK")

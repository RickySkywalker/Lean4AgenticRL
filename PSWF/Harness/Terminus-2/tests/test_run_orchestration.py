#!/usr/bin/env python3
"""Synthetic regression tests for run orchestration."""

from __future__ import annotations

from synthetic_fixtures import synthetic_plan

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor.models.agent.context import AgentContext  # noqa: E402

from terminus2_dynwf import wf_render  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402


FAILS: list[str] = []


def check(cond: bool, label: str, extra: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {extra}")
        FAILS.append(label)


SYNTHETIC_SPEC = synthetic_plan()


class FakeSession:
    """Only the two methods run() touches before handing off to _run_agent_loop."""

    def __init__(self) -> None:
        self.n_reads = 0

    async def get_incremental_output(self) -> str:
        self.n_reads += 1
        return f"root@host:/app# (terminal read {self.n_reads})"

    async def is_session_alive(self) -> bool:
        return True


class Recorder:
    """Stands in for _run_agent_loop and records what each phase was actually given."""

    def __init__(self, agent: Terminus2Dynamic, *, p1_timeout: bool = False, emit_plan: str | None = SYNTHETIC_SPEC):
        self.agent = agent
        self.p1_timeout = p1_timeout
        self.emit_plan = emit_plan
        self.calls: list[dict] = []

    async def __call__(self, initial_prompt: str, chat, original_instruction: str = "") -> None:
        phase = len(self.calls) + 1
        self.calls.append({
            "phase": phase,
            "prompt": initial_prompt,
            "original_instruction": original_instruction,
            "max_episodes_at_entry": self.agent._max_episodes,
            "pending_completion_at_entry": self.agent._pending_completion,
        })
        if phase == 1:
            for i in range(3):
                chat._messages.extend([
                    {"role": "user", "content": f"turn {i}"},
                    {"role": "assistant", "content": json.dumps(
                        {"analysis": "looking around", "plan": f"ls -la step {i}",
                         "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}]})},
                ])
                self.agent._api_request_times.append(1.0)
                self.agent._n_episodes = i + 1
            if self.emit_plan:
                chat._messages.extend([
                    {"role": "user", "content": "terminal"},
                    {"role": "assistant", "content": json.dumps(
                        {"analysis": "Exploration finished.", "plan": self.emit_plan,
                         "commands": [], "task_complete": True})},
                ])
                self.agent._api_request_times.append(1.0)
                self.agent._n_episodes += 1
            self.agent._pending_completion = True
            if self.p1_timeout:
                await asyncio.sleep(3600)
        else:
            self.agent._api_request_times.append(1.0)
            self.agent._n_episodes = 1


def make_agent(tmp: Path, **kw) -> Terminus2Dynamic:
    defaults = {"phase1_max_turns": 40, "agent_budget_s": 7200}
    defaults.update(kw)
    kw = defaults
    a = Terminus2Dynamic(
        logs_dir=tmp / "agent",
        model_name="openai/amazon/LAR-27B",
        max_turns=250,
        api_base="http://127.0.0.1:8800/v1",
        temperature=0.6,
        reasoning_effort="max",
        suppress_max_turns_warning=True,
        model_info={"max_input_tokens": 1048576, "max_output_tokens": 307200,
                    "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
                    "litellm_provider": "openai", "mode": "chat"},
        llm_kwargs={"api_key": "local"},
        trusted_root=str(tmp / "trusted"),
        **kw,
    )
    (tmp / "agent").mkdir(parents=True, exist_ok=True)
    a._session = FakeSession()          # type: ignore[assignment]
    return a


def run_case(**kw) -> tuple[Terminus2Dynamic, Recorder, Path, dict]:
    tmp = Path(tempfile.mkdtemp())
    rec_kw = {k: kw.pop(k) for k in ("p1_timeout", "emit_plan") if k in kw}
    a = make_agent(tmp, **kw)
    rec = Recorder(a, **rec_kw)
    a._run_agent_loop = rec             # type: ignore[assignment]
    ctx = AgentContext()
    asyncio.run(a.run("Make the failing tests pass.", None, ctx))  # type: ignore[arg-type]
    meta_p = tmp / "agent" / "dynwf_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    return a, rec, tmp, meta


print("== happy path: same_chat, plan delivered in the `plan` field ==")
a, rec, tmp, meta = run_case()

check(len(rec.calls) == 2, f"_run_agent_loop called twice (phases), got {len(rec.calls)}")
p1, p2 = rec.calls[0], rec.calls[1]

check("PHASE 1 of 2" in p1["prompt"], "phase 1 prompt carries the phase-1 banner")
check("Make the failing tests pass." in p1["prompt"], "phase 1 prompt carries the instruction")
check("Current terminal state:" in p1["prompt"], "phase 1 prompt went through terminus-2's own template")
check(p1["max_episodes_at_entry"] == 40, f"phase 1 capped at 40 turns, got {p1['max_episodes_at_entry']}")

check("PHASE 2 of 2" in p2["prompt"], "phase 2 prompt carries the phase-2 banner")
check(wf_render.SPEC_SECTION_HEADER in p2["prompt"], "phase 2 prompt has the '## Specified Workflow' header")
check(SYNTHETIC_SPEC.strip()[:70] in p2["prompt"], "phase 2 prompt contains the plan the model wrote")
check(wf_render.MEMORY_CLAIM_SAME in p2["prompt"], "same_chat -> the same-session memory sentence")
check(p2["max_episodes_at_entry"] == 250 - 4,
      f"phase 2 got the REMAINING turns (250-4=246), got {p2['max_episodes_at_entry']}")
check(p2["pending_completion_at_entry"] is False,
      "_pending_completion was reset to False before phase 2")
check("Current terminal state:" not in p2["prompt"],
      "same_chat phase 2 is NOT re-wrapped in the template (the format already lives in messages[0])")
check(p2["original_instruction"] == "Make the failing tests pass.",
      "phase 2 passes the ORIGINAL instruction, so summarization anchors on the real task")

usteps = [s for s in a._trajectory_steps if getattr(s, "source", None) == "user"]
check(len(usteps) == 2, f"exactly two user Steps appended (one per phase), got {len(usteps)}")
check(any("PHASE 1 of 2" in (s.message or "") for s in usteps), "phase-1 user Step recorded")
check(any("PHASE 2 of 2" in (s.message or "") for s in usteps), "phase-2 user Step recorded")

print("== meta ==")
check(meta.get("spec_source") == "json_plan", f"spec_source=json_plan, got {meta.get('spec_source')}")
check(meta.get("have_spec") is True, "have_spec true")
check(meta.get("phase2_spec_injected") is True, "phase2_spec_injected true")
check(meta.get("phase2_variant") == "injected", "phase2_variant=injected")
check(meta.get("phase1_stopped_by") == "completed", "phase1_stopped_by=completed")
check(meta.get("phase1_calls") == 4, f"phase1_calls=4 (the boundary), got {meta.get('phase1_calls')}")
check(meta.get("phase1_episodes") == 4, f"phase1_episodes=4, got {meta.get('phase1_episodes')}")
check(meta.get("total_calls") == 5, f"total_calls=5, got {meta.get('total_calls')}")
check(meta["phase1_calls"] < meta["total_calls"], "phase boundary is a strict prefix of the episode")
check(meta.get("phase2_handoff") == "same_chat", "phase2_handoff=same_chat")
check(len(meta.get("spec_needles") or []) >= 5, f"needles recorded: {len(meta.get('spec_needles') or [])}")
check(all(n in p2["prompt"] for n in meta["spec_needles"]),
      "every recorded needle really is in the phase-2 prompt (this is what the audit checks)")

print("== artifacts, in logs_dir AND in the trusted dir ==")
for tag, base in (("logs_dir", tmp / "agent"),
                  ("trusted", tmp / "trusted" / (a._session_id or ""))):
    for f in ("dynwf_meta.json", "specified_workflow.md", "phase1_prompt.md", "phase2_prompt.md",
              "phase1_calls.txt"):
        check((base / f).is_file(), f"{tag}/{f} written")
    if (base / "specified_workflow.md").is_file():
        check((base / "specified_workflow.md").read_text().strip()[:60] == SYNTHETIC_SPEC.strip()[:60],
              f"{tag}/specified_workflow.md is the plan verbatim")
check(not str(tmp / "trusted").startswith(str(tmp / "agent")),
      "the trusted copy is NOT under logs_dir (which is bind-mounted into the task container)")

print("== context backfill (harbor reads these for its own metrics) ==")
ctx_meta = None
tmp2 = Path(tempfile.mkdtemp())
a2 = make_agent(tmp2)
a2._run_agent_loop = Recorder(a2)  # type: ignore[assignment]
ctx2 = AgentContext()
asyncio.run(a2.run("T", None, ctx2))  # type: ignore[arg-type]
check(ctx2.metadata is not None and "n_episodes" in ctx2.metadata, "context.metadata carries n_episodes")
check(ctx2.metadata is not None and "dynwf" in ctx2.metadata, "context.metadata carries the dynwf block")
check(ctx2.metadata is not None and "api_request_times_msec" in ctx2.metadata,
      "context.metadata carries api_request_times_msec")

print("== fallback: phase 1 emits no usable plan ==")
a3, rec3, tmp3, meta3 = run_case(emit_plan=None)
p2b = rec3.calls[1]
check(meta3.get("have_spec") is False, "have_spec false")
check(meta3.get("spec_source") == "none", f"spec_source=none, got {meta3.get('spec_source')}")
check(meta3.get("phase2_variant") == "fallback", "phase2_variant=fallback")
check(meta3.get("phase2_spec_injected") is False, "phase2_spec_injected false")
check(wf_render.SPEC_SECTION_HEADER not in p2b["prompt"],
      "the fallback prompt has NO '## Specified Workflow' section at all (not an empty heading)")
check("Make the failing tests pass." in p2b["prompt"], "fallback still carries the task")
check(len(rec3.calls) == 2, "phase 2 still runs — a bad plan must not lose the task outright")

print("== phase-1 deadline: the episode continues into phase 2 ==")
a4, rec4, tmp4, meta4 = run_case(p1_timeout=True, agent_budget_s=10.0, phase1_min_s=1.0)
check(len(rec4.calls) == 2, f"phase 2 ran after the phase-1 deadline, calls={len(rec4.calls)}")
check(meta4.get("phase1_stopped_by") == "phase_deadline",
      f"phase1_stopped_by=phase_deadline, got {meta4.get('phase1_stopped_by')}")
check(meta4.get("have_spec") is True,
      "the plan emitted BEFORE the deadline is still extracted — a killed phase 1 is a handoff, not a loss")

print("== vanilla mode routes to stock terminus-2 ==")
tmp5 = Path(tempfile.mkdtemp())
a5 = make_agent(tmp5, tb_mode="vanilla")
seen: list[str] = []


async def stock_loop(initial_prompt: str, chat, original_instruction: str = "") -> None:
    seen.append(initial_prompt)


a5._run_agent_loop = stock_loop  # type: ignore[assignment]
asyncio.run(a5.run("Make the failing tests pass.", None, AgentContext()))  # type: ignore[arg-type]
check(len(seen) == 1, f"vanilla mode runs ONE loop, got {len(seen)}")
check("PHASE 1 of 2" not in seen[0] and "Specified Workflow" not in seen[0],
      "vanilla mode injects nothing at all")
check("Make the failing tests pass." in seen[0] and "Current terminal state:" in seen[0],
      "vanilla mode is the stock templated prompt")
check(not (tmp5 / "agent" / "dynwf_meta.json").is_file(),
      "vanilla mode writes no dynwf artifacts (it is literally the parent's run())")

print()
if FAILS:
    print(f"ORCHESTRATION GATE FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("ORCHESTRATION GATE: ALL GREEN")

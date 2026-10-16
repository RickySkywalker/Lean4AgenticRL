#!/usr/bin/env python3
"""Synthetic regression tests for dynwf compact handoff."""

from __future__ import annotations

from synthetic_fixtures import synthetic_plan

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor.llms.chat import Chat  # noqa: E402
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

NEEDLE = "`/fixture/output.txt` must contain the exact requested text; the fixture checks the bytes literally."
P1_ONLY = "Do NOT solve the task in this phase"
HANDOFF = (
    'SYNTHETIC NOTES: the fixture directory is ready for a text-file task.\n'
)


class FakeSession:
    def __init__(self) -> None:
        self.n_reads = 0
        self.sent: list[str] = []

    async def get_incremental_output(self) -> str:
        self.n_reads += 1
        return f"root@host:/app# (terminal read {self.n_reads})"

    async def is_session_alive(self) -> bool:
        return True

    async def send_keys(self, keystrokes: str, block: bool = False, min_timeout_sec: float = 0.0) -> None:
        self.sent.append(keystrokes)


def make_agent(tmp: Path, **kw) -> Terminus2Dynamic:
    defaults = {"phase1_max_turns": 40, "agent_budget_s": 7200}
    defaults.update(kw)
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
        **defaults,
    )
    (tmp / "agent").mkdir(parents=True, exist_ok=True)
    a._session = FakeSession()          # type: ignore[assignment]
    return a


class Recorder:
    """Stands in for _run_agent_loop; records the identity of the chat each phase is driven on."""

    def __init__(self, agent: Terminus2Dynamic, *, plan_at: str | None = "explore", plan: str = SYNTHETIC_SPEC):
        self.agent = agent
        self.plan_at = plan_at
        self.plan = plan
        self.calls: list[dict] = []

    def _kind(self, prompt: str) -> str:
        if "EXPLORATION IS CLOSED" in prompt:
            return "forced"
        if "PHASE 1 of 2" in prompt:
            return "explore"
        return "solve"

    def _emit(self, chat, text: str) -> None:
        chat._messages.extend([
            {"role": "user", "content": "terminal"},
            {"role": "assistant", "content": json.dumps(
                {"analysis": "Exploration finished.", "plan": text,
                 "commands": [], "task_complete": True})},
        ])
        self.agent._api_request_times.append(1.0)
        self.agent._n_episodes += 1

    async def __call__(self, initial_prompt: str, chat, original_instruction: str = "") -> None:
        kind = self._kind(initial_prompt)
        self.calls.append({
            "kind": kind,
            "prompt": initial_prompt,
            "chat_id": id(chat),
            "chat_msgs_at_entry": len(chat._messages),
            "chat_text_at_entry": json.dumps(chat._messages)[:400000],
        })
        if kind == "explore":
            self.agent._n_episodes = 0
            chat._messages.append({"role": "user", "content": initial_prompt})
            for i in range(3):
                chat._messages.extend([
                    {"role": "assistant", "content": json.dumps(
                        {"analysis": "looking around", "plan": f"ls -la step {i}",
                         "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}]})},
                    {"role": "user", "content": f"turn {i}"},
                ])
                self.agent._api_request_times.append(1.0)
                self.agent._n_episodes = i + 1
            if self.plan_at == "explore":
                self._emit(chat, self.plan)
            self.agent._pending_completion = True
        else:
            self.agent._api_request_times.append(1.0)
            self.agent._n_episodes = 1


class Summarizer:
    """Stands in for the parent's three-subagent `_summarize`. Records what the agent's state was AT the call."""

    def __init__(self, agent: Terminus2Dynamic, *, text: str = HANDOFF, n_refs: int = 3,
                 raises: BaseException | None = None):
        self.agent = agent
        self.text = text
        self.n_refs = n_refs
        self.raises = raises
        self.calls: list[dict] = []

    async def __call__(self, chat, original_instruction: str, session) -> tuple[str, list]:
        self.calls.append({
            "spec_for_reinject": self.agent._dynwf_spec_for_reinject,
            "chat_id": id(chat),
            "n_msgs": len(chat._messages),
        })
        if self.raises is not None:
            raise self.raises
        chat._messages = [chat.messages[0], {"role": "user", "content": "what is missing?"},
                          {"role": "assistant", "content": "these things"}]
        return self.text, [f"ref{i}" for i in range(self.n_refs)]


def run_case(*, plan_at: str | None = "explore", ctx_frac: float | None = 0.5,
             summarizer: dict | None = None, **kw):
    """Drive one full run. `ctx_frac` fixes the mocked phase-1 context fraction; None makes it RAISE."""
    tmp = Path(tempfile.mkdtemp())
    a = make_agent(tmp, **kw)
    rec = Recorder(a, plan_at=plan_at)
    a._run_agent_loop = rec             # type: ignore[assignment]
    summ = Summarizer(a, **(summarizer or {}))
    a._summarize = summ                 # type: ignore[assignment]
    if ctx_frac is None:
        def _boom(_chat):
            raise RuntimeError("tokenizer unavailable")
        a._count_total_tokens = _boom   # type: ignore[assignment]
    else:
        limit = int(a._llm.get_model_context_limit())
        a._count_total_tokens = lambda _chat, _n=int(limit * ctx_frac): _n  # type: ignore[assignment]
    ctx = AgentContext()
    asyncio.run(a.run("Make the failing tests pass.", None, ctx))  # type: ignore[arg-type]
    meta_p = tmp / "agent" / "dynwf_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    solve = next((c for c in rec.calls if c["kind"] == "solve"), None)
    explore = next((c for c in rec.calls if c["kind"] == "explore"), None)
    return a, rec, summ, meta, explore, solve


print("== phase1_max_s: the ceiling, at the configured total budget ==")
BIG = dict(agent_budget_s=20000.0, phase1_frac=0.27, phase1_min_s=60.0)

_, _, _, meta, _, _ = run_case(**BIG)
check(meta["phase1_budget_s"] == 5400.0,
      "ceiling OFF (0.0): the slice is frac x budget, i.e. the 5400 s this control exists to prevent",
      str(meta.get("phase1_budget_s")))
check(meta["phase1_max_s"] == 0.0, "phase1_max_s recorded as off", str(meta.get("phase1_max_s")))

_, _, _, meta, _, _ = run_case(phase1_max_s=1800.0, **BIG)
check(meta["phase1_budget_s"] == 1800.0, "ceiling ON: the slice is clamped to the absolute seconds",
      str(meta.get("phase1_budget_s")))
check(meta["phase1_max_s"] == 1800.0, "the value used is recorded in the meta")

_, _, _, meta, _, _ = run_case(phase1_max_s=99999.0, **BIG)
check(meta["phase1_budget_s"] == 5400.0, "a ceiling ABOVE the fractional slice does not raise it",
      str(meta.get("phase1_budget_s")))

_, _, _, meta, _, _ = run_case(agent_budget_s=20000.0, phase1_frac=0.27,
                               phase1_min_s=2400.0, phase1_max_s=1800.0)
check(meta["phase1_budget_s"] == 2400.0, "the FLOOR wins over the ceiling, so a contradictory pair still "
      "leaves phase 1 long enough to emit", str(meta.get("phase1_budget_s")))

_, _, _, meta, _, _ = run_case(agent_budget_s=200.0, phase1_frac=0.27, phase1_min_s=60.0, phase1_max_s=1800.0)
check(meta["phase1_budget_s"] == 60.0, "with a small budget the floor still applies and the ceiling is inert",
      str(meta.get("phase1_budget_s")))

print()
print("== defaults: the handoff control is inert ==")
_, rec, summ, meta, explore, solve = run_case()
check(meta["phase2_handoff"] == "same_chat" and meta["phase2_handoff_requested"] == "same_chat",
      "no phase2_handoff -> phase2_same_chat decides, as before",
      f"{meta.get('phase2_handoff')} / {meta.get('phase2_handoff_requested')}")
check(meta["phase2_boundary_compaction"] == {"requested": False, "applied": False},
      "compaction not requested and not applied", str(meta.get("phase2_boundary_compaction")))
check(len(summ.calls) == 0, "the summarizer is never called", str(len(summ.calls)))
check(meta["memory_claim"] == wf_render.MEMORY_CLAIM_SAME, "the same-chat memory claim is used")
check(explore["chat_id"] == solve["chat_id"], "phase 2 runs on the SAME Chat object")

_, rec, summ, meta, explore, solve = run_case(phase2_same_chat=False)
check(meta["phase2_handoff"] == "fresh_chat", "phase2_same_chat=False still means fresh_chat",
      str(meta.get("phase2_handoff")))
check(len(summ.calls) == 0 and meta["memory_claim"] == wf_render.MEMORY_CLAIM_FRESH,
      "fresh_chat does not summarize and carries the fresh claim")
check(explore["chat_id"] != solve["chat_id"] and solve["chat_msgs_at_entry"] == 0,
      "fresh_chat gives phase 2 an empty Chat")

_, rec, summ, meta, explore, solve = run_case(phase2_handoff="fresh_chat", phase2_same_chat=True)
check(meta["phase2_handoff"] == "fresh_chat" and len(summ.calls) == 0,
      "an explicit fresh_chat overrides phase2_same_chat and still does not summarize")

print()
print("== compact: the handoff applies, in a fresh Chat, with the spec injected exactly once ==")
a, rec, summ, meta, explore, solve = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.0,
                                              ctx_frac=0.5, **BIG)
comp = meta["phase2_boundary_compaction"]
check(meta["phase2_handoff"] == "compact" and comp["requested"] and comp["applied"],
      "mode is compact and the compaction applied", json.dumps(comp))
check(len(summ.calls) == 1, "the summarizer ran exactly once", str(len(summ.calls)))
check(comp.get("handoff_chars") == len(HANDOFF) and comp.get("subagent_refs") == 3
      and isinstance(comp.get("elapsed_s"), (int, float)),
      "handoff size, subagent count and elapsed time are all recorded", json.dumps(comp))
check(a._pending_subagent_refs == ["ref0", "ref1", "ref2"],
      "the three subagent trajectories are attached to the trial, so the compaction is auditable",
      str(a._pending_subagent_refs))
check(meta["memory_claim"] == wf_render.MEMORY_CLAIM_COMPACT,
      "the prompt claims COMPACTED memory -- neither 'it is in your context' nor 'it is gone'")
check(wf_render.MEMORY_CLAIM_COMPACT in solve["prompt"], "and that claim is actually in the rendered prompt")
check(wf_render.COMPACTED_NOTES_HEADER in solve["prompt"] and HANDOFF in solve["prompt"],
      "the handoff notes are appended under their own heading")
check(solve["prompt"].count(NEEDLE) == 1,
      "THE SPEC IS INJECTED EXACTLY ONCE (never doubled by the summarizer's re-injection)",
      str(solve["prompt"].count(NEEDLE)))
check(summ.calls[0]["spec_for_reinject"] == "",
      "_dynwf_spec_for_reinject is EMPTY during the boundary compaction, which is what prevents the double",
      repr(summ.calls[0]["spec_for_reinject"]))
check(explore["chat_id"] != solve["chat_id"] and solve["chat_msgs_at_entry"] == 0,
      "phase 2 runs on a FRESH Chat",
      f"{explore['chat_id']} vs {solve['chat_id']} / {solve['chat_msgs_at_entry']}")
check(P1_ONLY not in solve["chat_text_at_entry"],
      "the phase-1 'do NOT solve the task' prompt did NOT survive the compaction into phase 2")
check(P1_ONLY not in solve["prompt"], "nor is it anywhere in the phase-2 prompt text")
check(meta["phase2_spec_injected"] and meta["phase2_variant"] == "injected",
      "the injection audit still sees an injected phase 2")
check("(terminal read" in solve["prompt"],
      "the fresh-chat prompt was re-wrapped through terminus-2's own template with a live terminal read")
check(meta["phase2_max_episodes"] == 250 - 4, "the phase boundary turn accounting is unaffected",
      str(meta.get("phase2_max_episodes")))

print()
print("== phase2_compact_min_frac: three extra LLM calls are spent only on the tail that felt the pressure ==")
_, rec, summ, meta, explore, solve = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.50,
                                              ctx_frac=0.05, **BIG)
comp = meta["phase2_boundary_compaction"]
check(meta["phase2_handoff"] == "same_chat" and comp["requested"] and not comp["applied"],
      "below the threshold the mode DOWNGRADES to same_chat", json.dumps(comp))
check("skipped_reason" in comp and comp.get("phase1_context_frac") == 0.05 and comp.get("min_frac") == 0.50,
      "the mocked fraction, the threshold and the reason are all recorded", json.dumps(comp))
check(len(summ.calls) == 0, "and the summarizer is NOT called, so nothing is paid for", str(len(summ.calls)))
check(meta["memory_claim"] == wf_render.MEMORY_CLAIM_SAME
      and wf_render.MEMORY_CLAIM_SAME in solve["prompt"],
      "the downgraded trial is told the truth: the exploration IS still in its context")
check(explore["chat_id"] == solve["chat_id"], "a downgraded trial keeps its chat")
check(meta["phase2_handoff_requested"] == "compact",
      "the REQUEST is still recorded, so a run that never compacted cannot look like one that did")

_, rec, summ, meta, explore, solve = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.50,
                                              ctx_frac=0.80, **BIG)
check(meta["phase2_handoff"] == "compact" and meta["phase2_boundary_compaction"]["applied"]
      and len(summ.calls) == 1,
      "at or above the threshold it compacts", json.dumps(meta.get("phase2_boundary_compaction")))

_, rec, summ, meta, _, _ = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.50,
                                    ctx_frac=None, **BIG)
check(meta["phase2_handoff"] == "compact" and meta["phase2_boundary_compaction"]["applied"],
      "if the fraction CANNOT be measured it compacts anyway -- the gate is an optimisation, not a permission",
      json.dumps(meta.get("phase2_boundary_compaction")))
check(meta["phase2_boundary_compaction"].get("phase1_context_frac") is None,
      "and the unmeasurable fraction is recorded as null rather than guessed")

print()
print("== a failed compaction costs NOTHING: every path lands on the configured fallback behaviour ==")
for label, summarizer, expect_err in (
    ("the summarizer RAISES", {"raises": RuntimeError("subagent 2 timed out")}, "RuntimeError"),
    ("the summarizer returns an EMPTY handoff", {"text": "   "}, "empty handoff"),
):
    _, rec, summ, meta, explore, solve = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.0,
                                                  ctx_frac=0.9, summarizer=summarizer, **BIG)
    comp = meta["phase2_boundary_compaction"]
    print(f"  -- {label}")
    check(meta["phase2_handoff"] == "same_chat" and not comp["applied"],
          f"{label}: falls back to same_chat", json.dumps(comp))
    check(expect_err in comp.get("error", ""), f"{label}: the error is recorded verbatim",
          str(comp.get("error")))
    check(solve is not None and [c["kind"] for c in rec.calls] == ["explore", "solve"],
          f"{label}: the episode still runs both phases -- no lost trial",
          str([c["kind"] for c in rec.calls]))
    check(solve["prompt"].count(NEEDLE) == 1 and meta["phase2_spec_injected"],
          f"{label}: the plan is still injected exactly once", str(solve["prompt"].count(NEEDLE)))
    check(wf_render.COMPACTED_NOTES_HEADER not in solve["prompt"],
          f"{label}: no empty notes heading is left behind")
    check(meta["memory_claim"] == wf_render.MEMORY_CLAIM_SAME,
          f"{label}: the memory claim matches what actually happened")

print()
print("== compact + no usable plan: the fallback still degrades gracefully ==")
_, rec, summ, meta, explore, solve = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.0,
                                              ctx_frac=0.9, plan_at=None, **BIG)
check(meta["phase2_variant"] == "fallback" and not meta["phase2_spec_injected"],
      "no plan -> the fallback prompt, reported as an uninjected trial",
      f"{meta.get('phase2_variant')} / {meta.get('phase2_spec_injected')}")
check(wf_render.SPEC_SECTION_HEADER not in solve["prompt"],
      "and there is no empty `## Specified Workflow` heading")
check(meta["phase2_boundary_compaction"]["applied"] and HANDOFF in solve["prompt"],
      "the compaction still runs, so a planless phase 2 at least inherits what was learned",
      json.dumps(meta.get("phase2_boundary_compaction")))
check(explore["chat_id"] != solve["chat_id"], "still a fresh Chat")

print()
print("== mid-phase-2 re-injection is re-armed AFTER the boundary, not during it ==")
a, rec, summ, meta, explore, solve = run_case(phase2_handoff="compact", phase2_compact_min_frac=0.0,
                                              ctx_frac=0.9, phase2_reinject_on_summarize=True, **BIG)
check(summ.calls[0]["spec_for_reinject"] == "",
      "empty DURING the boundary compaction", repr(summ.calls[0]["spec_for_reinject"]))
check(a._dynwf_spec_for_reinject.strip().startswith("Prepare the synthetic fixture output"),
      "but set again for any LATER phase-2 compaction that evicts the plan",
      repr(a._dynwf_spec_for_reinject[:60]))
check(solve["prompt"].count(NEEDLE) == 1, "and the boundary prompt still carries the plan exactly once")

print()
print("== render_phase2 default and explicit handoff modes ==")
doc = wf_render.load_wf()
default_same, default_variant = wf_render.render_phase2(doc, "task", SYNTHETIC_SPEC, True)
explicit_same, explicit_variant = wf_render.render_phase2(doc, "task", SYNTHETIC_SPEC, True, handoff=None)
check(default_same == explicit_same and default_variant == explicit_variant == "injected",
      "handoff=None selects the same prompt as the default handoff")
compact_p, _ = wf_render.render_phase2(doc, "task", SYNTHETIC_SPEC, False, handoff="compact")
check(wf_render.MEMORY_CLAIM_COMPACT in compact_p
      and wf_render.MEMORY_CLAIM_FRESH not in compact_p
      and wf_render.MEMORY_CLAIM_SAME not in compact_p,
      "compact carries exactly one memory claim, its own")
check("Keep replying in the same JSON format" not in compact_p,
      "compact does not restate the JSON contract: it opens a fresh chat, so the template carries it")
check("Keep replying in the same JSON format" in default_same,
      "same_chat restates the JSON contract")

print()
if FAILS:
    print(f"COMPACT-HANDOFF GATE: {len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("COMPACT-HANDOFF GATE: ALL GREEN")

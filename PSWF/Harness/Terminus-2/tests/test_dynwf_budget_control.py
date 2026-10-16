#!/usr/bin/env python3
"""Synthetic regression tests for dynwf budget control."""

from __future__ import annotations

from synthetic_fixtures import synthetic_plan

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harbor.agents.terminus_2.terminus_2 import Terminus2  # noqa: E402
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

NUDGE_MARK = "WRITE THE SPECIFIED WORKFLOW NOW"
REMIND_MARK = "[PHASE-1 BUDGET:"
FORCED_MARK = "EXPLORATION IS CLOSED"


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
    """Stands in for _run_agent_loop. Knows the three prompts it can be handed apart."""

    def __init__(self, agent: Terminus2Dynamic, *, p1_timeout: bool = False,
                 plan_at: str | None = "explore", plan: str = SYNTHETIC_SPEC):
        self.agent = agent
        self.p1_timeout = p1_timeout
        self.plan_at = plan_at
        self.plan = plan
        self.calls: list[dict] = []

    def _kind(self, prompt: str) -> str:
        if FORCED_MARK in prompt:
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
            "max_episodes_at_entry": self.agent._max_episodes,
            "pending_completion_at_entry": self.agent._pending_completion,
        })
        if kind == "explore":
            self.agent._n_episodes = 0
            for i in range(3):
                chat._messages.extend([
                    {"role": "user", "content": f"turn {i}"},
                    {"role": "assistant", "content": json.dumps(
                        {"analysis": "looking around", "plan": f"ls -la step {i}",
                         "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}]})},
                ])
                self.agent._api_request_times.append(1.0)
                self.agent._n_episodes = i + 1
            if self.plan_at == "explore":
                self._emit(chat, self.plan)
            self.agent._pending_completion = True
            if self.p1_timeout:
                await asyncio.sleep(3600)
        elif kind == "forced":
            self.agent._n_episodes = 0
            if self.plan_at == "forced":
                self._emit(chat, self.plan)
            else:
                self.agent._api_request_times.append(1.0)
                self.agent._n_episodes = 1
        else:
            self.agent._api_request_times.append(1.0)
            self.agent._n_episodes = 1


def run_case(**kw):
    tmp = Path(tempfile.mkdtemp())
    rec_kw = {k: kw.pop(k) for k in ("p1_timeout", "plan_at", "plan") if k in kw}
    a = make_agent(tmp, **kw)
    rec = Recorder(a, **rec_kw)
    a._run_agent_loop = rec             # type: ignore[assignment]
    ctx = AgentContext()
    asyncio.run(a.run("Make the failing tests pass.", None, ctx))  # type: ignore[arg-type]
    meta_p = tmp / "agent" / "dynwf_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    return a, rec, tmp, meta


FAST = dict(agent_budget_s=8.0, phase1_frac=0.5, phase1_min_s=0.5,
            phase1_spec_reserve_s=2.5, phase1_forced_min_s=0.2)

print("== defaults: every control OFF, so default behavior is unchanged ==")
a, rec, tmp, meta = run_case()
check([c["kind"] for c in rec.calls] == ["explore", "solve"],
      f"two loops, no forced turn, got {[c['kind'] for c in rec.calls]}")
check(meta["phase1_reserve_s"] == 0.0 and meta["phase1_explore_budget_s"] == meta["phase1_budget_s"],
      "with no reserve the exploration slice IS the phase-1 budget",
      f"{meta.get('phase1_reserve_s')} / {meta.get('phase1_explore_budget_s')} / {meta.get('phase1_budget_s')}")
check(meta["phase1_wrapups"] == 0, "no nudge fired")
check(meta["phase1_forced_spec"] == {"attempted": False}, "no forced turn attempted",
      str(meta.get("phase1_forced_spec")))
check(meta["spec_truncated_from"] == 0, "no truncation")
check(meta["phase2_spec_reinjections"] == 0, "no re-injection")
check(meta["phase1_single_confirm"] is False, "single-confirm off")
check(wf_render.CONFIRM_CLAIM_DOUBLE.splitlines()[0] in rec.calls[0]["prompt"],
      "the phase-1 prompt carries the DOUBLE-confirm claim by default")

print("== single_confirm=True swaps the claim to the true one ==")
a, rec, tmp, meta = run_case(phase1_single_confirm=True)
p1p = rec.calls[0]["prompt"]
check(wf_render.CONFIRM_CLAIM_SINGLE.splitlines()[0] in p1p,
      "the phase-1 prompt carries the SINGLE-confirm claim")
check(wf_render.CONFIRM_CLAIM_DOUBLE.splitlines()[0] not in p1p,
      "and no longer promises a confirmation round that will not happen")
check(meta["phase1_single_confirm"] is True, "recorded in meta for the audit")

print("== _execute_commands: the nudge, and only in phase 1 ==")


def probe(agent: Terminus2Dynamic, phase: int, seconds_left: float, span: float, episodes: int) -> str:
    import time as _t
    agent._dynwf_phase = phase
    agent._dynwf_explore_span = span
    agent._dynwf_explore_deadline = _t.time() + seconds_left
    agent._n_episodes = episodes
    return asyncio.run(agent._execute_commands([], agent._session))[1]


tmpn = Path(tempfile.mkdtemp())
an = make_agent(tmpn, phase1_wrapup_frac=0.6, phase1_wrapup_turn_frac=0.0)
out = probe(an, 1, seconds_left=900.0, span=1000.0, episodes=3)
check(NUDGE_MARK not in out and REMIND_MARK not in out, "no nudge at 10 % of the slice")
out = probe(an, 1, seconds_left=300.0, span=1000.0, episodes=8)
check(NUDGE_MARK in out, "full nudge at 70 % of the slice (threshold 0.6)")
check(out.startswith("root@host:/app#"), "the nudge is APPENDED, so the terminal output still leads")
secs = [int(m) for m in re.findall(r"(\d+)\s*\n?\s*second\(s\)", out.split(NUDGE_MARK, 1)[1])]
check(bool(secs) and abs(secs[0] - 300) <= 2,
      "the nudge states the seconds actually remaining", f"parsed {secs}")
out = probe(an, 1, seconds_left=200.0, span=1000.0, episodes=9)
check(REMIND_MARK in out and NUDGE_MARK not in out,
      "the SECOND and later firings are the one-line reminder, not 20 lines again")
check(an._dynwf_wrapups == 2, f"nudge count recorded, got {an._dynwf_wrapups}")
out = probe(an, 2, seconds_left=1.0, span=1000.0, episodes=99)
check(NUDGE_MARK not in out and REMIND_MARK not in out,
      "INERT IN PHASE 2 — a solving agent must never be told to stop and write a plan")
out = probe(an, 0, seconds_left=1.0, span=1000.0, episodes=99)
check(NUDGE_MARK not in out and REMIND_MARK not in out, "inert outside a dynamic run (vanilla mode)")

tmpn2 = Path(tempfile.mkdtemp())
an2 = make_agent(tmpn2, phase1_wrapup_frac=0.0, phase1_wrapup_turn_frac=0.65, phase1_max_turns=40)
out = probe(an2, 1, seconds_left=900.0, span=1000.0, episodes=20)
check(NUDGE_MARK not in out, "turn axis: silent at 20/40 turns")
out = probe(an2, 1, seconds_left=900.0, span=1000.0, episodes=26)
check(NUDGE_MARK in out, "turn axis: fires at 26/40 turns (threshold 0.65)")
check("14" in out.split(NUDGE_MARK, 1)[1], "and states the turns actually remaining")

tmpn3 = Path(tempfile.mkdtemp())
an3 = make_agent(tmpn3)
out = probe(an3, 1, seconds_left=0.5, span=1000.0, episodes=39)
check(NUDGE_MARK not in out and REMIND_MARK not in out, "both thresholds 0 = the control is off entirely")

print("== _execute_commands: single-confirm forces the flag the loop is about to read ==")
tmps = Path(tempfile.mkdtemp())
asc = make_agent(tmps, phase1_single_confirm=True)
asc._dynwf_phase = 1
asc._pending_completion = False
asyncio.run(asc._execute_commands([], asc._session))
check(asc._pending_completion is True, "phase 1 + single_confirm -> flag forced True")
asc._dynwf_phase = 2
asc._pending_completion = False
asyncio.run(asc._execute_commands([], asc._session))
check(asc._pending_completion is False,
      "phase 2 -> NOT forced (forcing here would end the episode on the first task_complete)")
tmps2 = Path(tempfile.mkdtemp())
asc2 = make_agent(tmps2)
asc2._dynwf_phase = 1
asc2._pending_completion = False
asyncio.run(asc2._execute_commands([], asc2._session))
check(asc2._pending_completion is False, "single_confirm off -> stock double-confirm preserved")

print("== the reserve is carved OUT of the phase-1 slice, never added to it ==")
a, rec, tmp, meta = run_case(**FAST)
check(abs(meta["phase1_budget_s"] - 4.0) < 0.05, f"p1 budget 4.0 s, got {meta['phase1_budget_s']}")
check(abs(meta["phase1_reserve_s"] - 2.5) < 0.05, f"reserve 2.5 s, got {meta['phase1_reserve_s']}")
check(abs(meta["phase1_explore_budget_s"] + meta["phase1_reserve_s"] - meta["phase1_budget_s"]) < 0.05,
      "explore + reserve == the phase-1 budget exactly")
tmpc = Path(tempfile.mkdtemp())
ac = make_agent(tmpc, agent_budget_s=8.0, phase1_frac=0.5, phase1_min_s=3.5, phase1_spec_reserve_s=99.0)
check(True, "clamp case built")

print("== forced turn: exploration is cut off with NO plan, the reserved tail produces one ==")
a, rec, tmp, meta = run_case(p1_timeout=True, plan_at="forced", **FAST)
kinds = [c["kind"] for c in rec.calls]
check(kinds == ["explore", "forced", "solve"], f"three loops in order, got {kinds}")
check(meta["phase1_stopped_by"] == "phase_deadline", "exploration ended on its deadline")
check(meta["phase1_forced_spec"]["attempted"] is True, "forced turn attempted")
check(meta["phase1_forced_spec"]["gained_spec"] is True,
      "and it is what produced the plan (gained_spec proves the plan did not exist before it)")
check(meta["have_spec"] is True and meta["phase2_variant"] == "injected",
      f"phase 2 is INJECTED, not the vanilla fallback: {meta.get('phase2_variant')}")
check(FORCED_MARK in rec.calls[1]["prompt"], "the forced prompt is the asset's phase1_forced_spec step")
check(rec.calls[1]["pending_completion_at_entry"] is True,
      "the forced turn enters with the confirmation flag already True — one response ends it")
check(rec.calls[1]["max_episodes_at_entry"] == 2,
      f"forced turn capped at phase1_spec_reserve_turns, got {rec.calls[1]['max_episodes_at_entry']}")
check(meta["phase1_episodes"] == 3 + 1,
      f"forced-turn episodes counted into the phase boundary, got {meta['phase1_episodes']}")
check(meta["phase2_max_episodes"] == 250 - 4,
      f"phase 2 gets the true remainder, got {meta.get('phase2_max_episodes')}")
check(rec.calls[2]["pending_completion_at_entry"] is False,
      "phase 2 still enters with the flag reset (manipulation 2 survives the new path)")
check(a._dynwf_force_confirm is False, "the forced-turn confirmation override is cleared afterwards")
check(a._max_episodes == 250 - 4, "the phase-1 turn cap is not left behind on the agent")

print("== forced turn is NOT attempted when exploration already produced a plan ==")
a, rec, tmp, meta = run_case(plan_at="explore", **FAST)
check([c["kind"] for c in rec.calls] == ["explore", "solve"], "no forced turn")
check(meta["phase1_forced_spec"] == {"attempted": False}, str(meta["phase1_forced_spec"]))

print("== forced turn is skipped when the reserve is already gone ==")
a, rec, tmp, meta = run_case(p1_timeout=True, plan_at=None,
                             agent_budget_s=8.0, phase1_frac=0.5, phase1_min_s=0.5,
                             phase1_spec_reserve_s=2.5, phase1_forced_min_s=99.0)
check([c["kind"] for c in rec.calls] == ["explore", "solve"],
      "a call that cannot land is not made at all")
check(meta["phase1_forced_spec"]["attempted"] is False
      and "reserve left" in str(meta["phase1_forced_spec"].get("skipped_reason")),
      f"and the reason is recorded: {meta['phase1_forced_spec']}")
check(meta["phase2_variant"] == "fallback", "phase 2 degrades exactly as it did before — nothing is lost")

print("== forced turn that produces nothing usable still lets phase 2 run ==")
a, rec, tmp, meta = run_case(p1_timeout=True, plan_at=None, **FAST)
check([c["kind"] for c in rec.calls] == ["explore", "forced", "solve"], "phase 2 still ran")
check(meta["phase1_forced_spec"].get("gained_spec") is False, "gained_spec false")
check(meta["phase2_variant"] == "fallback", "and the metadata records the fallback")

print("== spec ceiling: clipped before rendering, writing and needle extraction ==")
BIG = SYNTHETIC_SPEC + "".join(f"   - filler line {i} of a runaway reasoning dump, padded out to be needle-eligible\n"
                          for i in range(400))
a, rec, tmp, meta = run_case(plan_at="explore", plan=BIG, spec_max_chars=2000)
check(meta["spec_truncated_from"] == len(BIG.strip()),
      "pre-clip length recorded", f"got {meta['spec_truncated_from']} want {len(BIG.strip())}")
check(meta["spec_chars"] <= 2200, f"injected spec is clipped, got {meta['spec_chars']}")
p2p = rec.calls[-1]["prompt"]
check("[... this workflow was clipped at 2000 characters" in p2p, "the clip is visible in the prompt")
check(len(meta["spec_needles"]) > 0 and all(n in p2p for n in meta["spec_needles"]),
      "EVERY recorded needle is still verbatim in the prompt — this is what the injection audit checks")
saved = (tmp / "agent" / "specified_workflow.md").read_text()
check("clipped at 2000 characters" in saved,
      "the archived copy is the bytes that were injected, not the pre-clip text")
a, rec, tmp, meta = run_case(plan_at="explore", plan=BIG)
check(meta["spec_truncated_from"] == 0 and meta["spec_chars"] == len(BIG.strip()) or
      meta["spec_truncated_from"] == 0,
      "cap 0 = no clipping at all")

print("== compaction re-injection , phase 2 only ==")
_real_parent = Terminus2._summarize


async def fake_parent(self, chat, original_instruction, session):   # noqa: ANN001
    return "Here are the answers the other agent provided.\n\nSUMMARY BODY", None


Terminus2._summarize = fake_parent       # type: ignore[assignment]
try:
    tmpr = Path(tempfile.mkdtemp())
    ar = make_agent(tmpr, phase2_reinject_on_summarize=True)
    ar._dynwf_phase = 2
    ar._dynwf_spec_for_reinject = SYNTHETIC_SPEC
    h, _ = asyncio.run(ar._summarize(None, "instr", None))
    check(wf_render.SPEC_SECTION_HEADER in h and "output.txt" in h,
          "the handoff prompt carries the Specified Workflow verbatim")
    check(h.startswith("Here are the answers"), "the stock handoff text is preserved and leads")
    check(ar._dynwf_reinjections == 1, "counted for the audit")
    ar._dynwf_phase = 1
    h1, _ = asyncio.run(ar._summarize(None, "instr", None))
    check(wf_render.SPEC_SECTION_HEADER not in h1,
          "INERT IN PHASE 1 — nothing has been injected yet, so there is nothing to restate")
    tmpr2 = Path(tempfile.mkdtemp())
    ar2 = make_agent(tmpr2)
    ar2._dynwf_phase = 2
    ar2._dynwf_spec_for_reinject = ""
    h2, _ = asyncio.run(ar2._summarize(None, "instr", None))
    check(wf_render.SPEC_SECTION_HEADER not in h2, "knob off -> stock handoff, byte for byte")
finally:
    Terminus2._summarize = _real_parent  # type: ignore[assignment]

print("== run() arms the re-injection only when there is a plan to re-inject ==")
a, rec, tmp, meta = run_case(plan_at="explore", phase2_reinject_on_summarize=True)
check(a._dynwf_spec_for_reinject.startswith("Prepare the synthetic fixture output"),
      "the spec is handed to the summarizer")
a, rec, tmp, meta = run_case(p1_timeout=True, plan_at=None, phase2_reinject_on_summarize=True,
                             agent_budget_s=8.0, phase1_frac=0.5, phase1_min_s=0.5)
check(a._dynwf_spec_for_reinject == "", "a fallback episode has nothing to re-inject and says so")

print("== _execute_commands: the PHASE-1 HARD STOP refuses exploration past the threshold ==")
HARD_MARK = "EXPLORATION IS CLOSED"
HARD_REPEAT_MARK = "[EXPLORATION IS CLOSED"
CMDS = [type("C", (), {"keystrokes": "ls -la\n", "duration_sec": 0.1})() for _ in range(3)]


def probe_cmds(agent: Terminus2Dynamic, phase: int, seconds_left: float, span: float, episodes: int, cmds):
    import time as _t
    agent._dynwf_phase = phase
    agent._dynwf_explore_span = span
    agent._dynwf_explore_deadline = _t.time() + seconds_left
    agent._n_episodes = episodes
    before = len(agent._session.sent)
    timeout_occurred, out = asyncio.run(agent._execute_commands(cmds, agent._session))
    return out, len(agent._session.sent) - before, timeout_occurred


tmph = Path(tempfile.mkdtemp())
ah = make_agent(tmph, phase1_hard_stop_frac=0.85, phase1_hard_stop_turn_frac=0.0, phase1_single_confirm=True,
                phase1_wrapup_frac=0.6)
out, sent, _ = probe_cmds(ah, 1, seconds_left=500.0, span=1000.0, episodes=5, cmds=CMDS)
check(HARD_MARK not in out and sent == 3, "below the threshold every command batch still reaches the terminal",
      f"sent={sent}")
check(NUDGE_MARK not in out, "at 50 % the wrap-up nudge (threshold 0.6) has not fired either")
out, sent, _ = probe_cmds(ah, 1, seconds_left=100.0, span=1000.0, episodes=6, cmds=CMDS)
check(HARD_MARK in out and sent == 0, "at 90 % of the slice (threshold 0.85) the batch is REFUSED: nothing is sent",
      f"sent={sent} out={out[:80]!r}")
check("3 COMMAND(S) WERE NOT RUN" in out, "the refusal names how many commands were dropped", out[:120])
check(ah._pending_completion is True, "single-confirm is still forced on a refused turn, so the plan response ends the phase")
check(ah._dynwf_hard_stops == 1 and ah._dynwf_hard_stop_cmds == 3, "refusals and dropped commands are counted",
      f"{ah._dynwf_hard_stops}/{ah._dynwf_hard_stop_cmds}")
secs = [int(m) for m in re.findall(r"about (\d+) second", out)]
check(bool(secs) and abs(secs[0] - 100) <= 2, "the refusal states the seconds actually remaining", f"parsed {secs}")
out, sent, _ = probe_cmds(ah, 1, seconds_left=80.0, span=1000.0, episodes=7, cmds=CMDS)
check(HARD_REPEAT_MARK in out and sent == 0 and "YOUR 3 COMMAND(S)" not in out,
      "the SECOND refusal is the short repeat, not the full text again")
out, sent, _ = probe_cmds(ah, 1, seconds_left=80.0, span=1000.0, episodes=8, cmds=[])
check(HARD_MARK not in out and HARD_REPEAT_MARK not in out,
      "an EMPTY batch (the deliverable response) passes through to the parent untouched", out[:80])
check(ah._pending_completion is True, "...with the confirmation flag forced, so task_complete ends the phase")
out, sent, _ = probe_cmds(ah, 2, seconds_left=1.0, span=1000.0, episodes=99, cmds=CMDS)
check(HARD_MARK not in out and sent == 3, "INERT IN PHASE 2 -- the solving agent's commands always run", f"sent={sent}")
out, sent, _ = probe_cmds(ah, 0, seconds_left=1.0, span=1000.0, episodes=99, cmds=CMDS)
check(HARD_MARK not in out and sent == 3, "inert outside a dynamic run (vanilla mode)", f"sent={sent}")

tmph2 = Path(tempfile.mkdtemp())
ah2 = make_agent(tmph2, phase1_hard_stop_frac=0.0, phase1_hard_stop_turn_frac=0.9, phase1_max_turns=40)
out, sent, _ = probe_cmds(ah2, 1, seconds_left=900.0, span=1000.0, episodes=35, cmds=CMDS)
check(HARD_MARK not in out and sent == 3, "turn axis: 35/40 still runs (threshold 0.9 = 36)", f"sent={sent}")
out, sent, _ = probe_cmds(ah2, 1, seconds_left=900.0, span=1000.0, episodes=36, cmds=CMDS)
check(HARD_MARK in out and sent == 0, "turn axis: refused at 36/40", f"sent={sent}")
check("4 turn(s)" in out, "and states the turns remaining", out[:200])

tmph3 = Path(tempfile.mkdtemp())
ah3 = make_agent(tmph3)
out, sent, _ = probe_cmds(ah3, 1, seconds_left=0.5, span=1000.0, episodes=39, cmds=CMDS)
check(HARD_MARK not in out and sent == 3, "both thresholds 0 = the hard stop is off entirely (default behavior unchanged)",
      f"sent={sent}")

print("== hard stop: counters reach the meta and reset per run ==")
a, rec, tmp, meta = run_case(phase1_hard_stop_frac=0.85)
check(meta["phase1_hard_stops"] == 0 and meta["phase1_hard_stop_cmds_refused"] == 0
      and meta["phase1_hard_stop_frac"] == 0.85,
      "meta records the threshold and zero refusals on a run that never hit it", str({k: meta.get(k) for k in
      ("phase1_hard_stops", "phase1_hard_stop_cmds_refused", "phase1_hard_stop_frac")}))
a._dynwf_hard_stops = 5
a._dynwf_reset_phase_state()
check(a._dynwf_hard_stops == 0 and a._dynwf_hard_stop_cmds == 0, "reset per run")

print("== per-run state does not leak between runs of one agent instance ==")
tmpl = Path(tempfile.mkdtemp())
al = make_agent(tmpl, phase1_wrapup_frac=0.6, phase2_reinject_on_summarize=True)
al._run_agent_loop = Recorder(al)        # type: ignore[assignment]
asyncio.run(al.run("Task A", None, AgentContext()))            # type: ignore[arg-type]
al._dynwf_wrapups = 7
al._run_agent_loop = Recorder(al)        # type: ignore[assignment]
asyncio.run(al.run("Task B", None, AgentContext()))            # type: ignore[arg-type]
m2 = json.loads((tmpl / "agent" / "dynwf_meta.json").read_text())
check(m2["phase1_wrapups"] == 0, f"nudge count reset by _reset_per_run_state, got {m2['phase1_wrapups']}")
check(al._dynwf_phase == 2, "phase flag left where the run ended, not leaked as phase 1")

print()
if FAILS:
    print(f"BUDGET-CONTROL GATE FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print("BUDGET-CONTROL GATE: ALL GREEN")

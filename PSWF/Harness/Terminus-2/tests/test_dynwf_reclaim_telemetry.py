#!/usr/bin/env python3
"""Synthetic regression tests for dynwf reclaim telemetry."""
from __future__ import annotations
import asyncio, json, sys, tempfile, types
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


STOCK_CALLS: list[str] = []


async def parent_summarize(self, chat, oi, session):
    """Upstream's three-call `_summarize`: it rebuilds history and charges the count."""
    STOCK_CALLS.append(oi)
    self._summarization_count += 1
    chat._messages = [chat.messages[0], {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    chat.reset_response_chain()
    return ("STOCK HANDOFF", ["r1", "r2", "r3"])


da.Terminus2._summarize = parent_summarize


class FakeSession:
    async def capture_pane(self, capture_entire=False):
        return "root@host:/app#"


class FakeChat:
    def __init__(self, n=41):
        self._messages = [{"role": "system", "content": "PHASE-2 PROMPT"}]
        self._messages += [{"role": "user" if i % 2 else "assistant", "content": f"m{i}"} for i in range(1, n)]
        self.resets = 0

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        self.resets += 1


MODEL_INFO = {"max_input_tokens": 212992, "max_output_tokens": 32768, "input_cost_per_token": 0.0,
              "output_cost_per_token": 0.0, "litellm_provider": "openai", "mode": "chat"}
BASE = dict(phase1_max_turns=30, phase1_frac=0.27, agent_budget_s=20000.0, phase1_max_s=1944.0,
            phase1_spec_reserve_s=600.0, phase1_forced_min_s=90.0, phase1_wrapup_frac=0.60,
            phase1_wrapup_turn_frac=0.60, phase1_hard_stop_frac=0.70, phase1_hard_stop_turn_frac=0.75,
            phase1_context_max_frac=0.25, phase1_single_confirm=True, phase2_same_chat=False,
            phase1_budget_hint=True, phase1_truncation_stop=True, phase1_deadline_grace_s=420.0,
            phase2_reinject_on_summarize=True, phase2_replan_before_compact=True,
            phase2_replan_prefer_structured=True, phase2_endgame_s=1800.0, replan_trusted_dump=True)


def build(telemetry, fast_threshold=0.0):
    tmp = Path(tempfile.mkdtemp())
    logs = tmp / "agent"
    logs.mkdir(parents=True, exist_ok=True)
    a = Terminus2Dynamic(logs_dir=logs, model_name="openai/x", max_turns=400,
                         api_base="http://127.0.0.1:1/v1", temperature=1.0, suppress_max_turns_warning=True,
                         model_info=MODEL_INFO, llm_kwargs={"api_key": "local"},
                         trusted_root=str(tmp / "trusted"),
                         compact_fast_below_s=fast_threshold, compact_reclaim_telemetry=telemetry, **BASE)
    a._count_total_tokens = lambda chat: 1000 * len(chat.messages)
    a._llm = types.SimpleNamespace(get_model_context_limit=lambda: 212992)
    a._dynwf_phase = 2
    a._dynwf_spec_for_reinject = "1. **Goal**\n2. **Env**\n3. **Plan**\n4. **Checkpoints**\n5. **Verification**"
    import time as _t
    a._dynwf_run_deadline = _t.time() + 5000
    return a


def summarize(a, chat):
    return asyncio.new_event_loop().run_until_complete(a._summarize(chat, "solve it", FakeSession()))


print("-- single-call compaction is provably OFF at threshold 0")
a = build(telemetry=True, fast_threshold=0.0)
check(a.compact_fast_below_s == 0.0, "threshold is 0")
check(a._dynwf_compact_fast_due()[0] is False, "clock test answers False at 5,000 s remaining")
import time as _time
a._dynwf_run_deadline = _time.time() + 10
check(a._dynwf_compact_fast_due()[0] is False, "and at 10 s remaining")
a._dynwf_run_deadline = _time.time() + 5000
STOCK_CALLS.clear()
chat = FakeChat(41)
handoff, refs = summarize(a, chat)
check(len(STOCK_CALLS) == 1, "the stock three-call path ran", len(STOCK_CALLS))
check(refs == ["r1", "r2", "r3"], "three subagent refs, from the stock summary path", refs)
row = a._dynwf_compactions[-1]
check(row["mode"] == "stock" and row["legs"] == 3, "the row says stock/3 legs", row)

print("\n-- the reclaim numbers are recorded and arithmetically right")
check(row.get("tokens_before") == 41000, "tokens_before = 41 messages x 1000", row.get("tokens_before"))
check(row.get("tokens_after") == 3000, "tokens_after = 3 messages x 1000 (history rebuilt)", row.get("tokens_after"))
check(row.get("reclaimed_tokens") == 38000, "reclaimed = before - after", row.get("reclaimed_tokens"))
check(isinstance(row.get("handoff_chars"), int) and row["handoff_chars"] > 0,
      "handoff_chars is present, so post-compaction context is computable", row.get("handoff_chars"))

print("\n-- OFF: no numbers recorded, and behaviour identical")
STOCK_CALLS.clear()
b = build(telemetry=False, fast_threshold=0.0)
chat_b = FakeChat(41)
handoff_b, refs_b = summarize(b, chat_b)
rb = b._dynwf_compactions[-1]
check(rb.get("tokens_before") is None, "OFF: tokens_before is None", rb.get("tokens_before"))
check("reclaimed_tokens" not in rb, "OFF: no reclaimed_tokens key", rb.get("reclaimed_tokens"))
check(handoff == handoff_b, "the handoff text is IDENTICAL with telemetry on and off")
check(chat.messages == chat_b.messages, "the rebuilt history is IDENTICAL", (chat.messages, chat_b.messages))
check(refs == refs_b, "the subagent refs are identical")
ignore = {"tokens_before", "tokens_after", "reclaimed_tokens"}
diff = {k: (row.get(k), rb.get(k)) for k in set(row) | set(rb) if k not in ignore and row.get(k) != rb.get(k)}
check(not diff, "every other recorded field is identical (the telemetry inertness property)", diff)
check(b._dynwf_reinjections == a._dynwf_reinjections, "the re-injection count is the same")

print("\n-- a token-count failure is survivable, not fatal")
c = build(telemetry=True, fast_threshold=0.0)


def boom(chat):
    raise RuntimeError("tokenizer unavailable")


c._count_total_tokens = boom
STOCK_CALLS.clear()
hc, _ = summarize(c, FakeChat(41))
rc = c._dynwf_compactions[-1]
check("STOCK HANDOFF" in hc, "the compaction still produced its handoff")
check(rc.get("tokens_before") is None and rc.get("tokens_after") is None,
      "both readings are None rather than raising", (rc.get("tokens_before"), rc.get("tokens_after")))
check("reclaimed_tokens" not in rc, "and no reclaimed_tokens is invented")

print("\n-- the meta carries the flag so a reader can tell telemetry was on")
check(hasattr(a, "compact_reclaim_telemetry") and a.compact_reclaim_telemetry is True, "flag is set on the agent")

print()
if FAILS:
    print(f"RECLAIM_TELEMETRY_GATE_FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("RECLAIM_TELEMETRY_GATE_OK")

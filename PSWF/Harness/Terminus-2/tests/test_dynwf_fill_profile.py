#!/usr/bin/env python3
"""Synthetic regression tests for dynwf fill profile."""
from __future__ import annotations
import asyncio, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from terminus2_dynwf import dynamic_agent as da  # noqa: E402
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic  # noqa: E402

FAILS: list[str] = []


def check(cond, label, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"  {extra}"))
    if not cond:
        FAILS.append(label)


STOCK_CALLS: list[str] = []


async def parent_summarize(self, chat, oi, session):
    STOCK_CALLS.append(oi)
    self._summarization_count += 1
    chat._messages = [chat.messages[0], {"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    chat.reset_response_chain()
    return ("STOCK HANDOFF", ["r1", "r2", "r3"])


da.Terminus2._summarize = parent_summarize


class FakeSession:
    async def capture_pane(self, capture_entire=False):
        return "root@host:/app#"


SIZES = [("system", 900), ("assistant", 800), ("user", 500), ("assistant", 50), ("user", 30), ("user", 250)]
CAP = 100
EXPECT_SHADOW = (500 - CAP) + (250 - CAP)
EXPECT_OVER_CAP = 2
EXPECT_SUM = sum(n for _, n in SIZES)
EXPECT_ASSISTANT = 800 + 50
EXPECT_OBSERVATION = EXPECT_SUM - EXPECT_ASSISTANT


class FakeChat:
    def __init__(self, sizes=SIZES):
        self._messages = [{"role": r, "content": "x" * n} for r, n in sizes]
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


def build(cap, telemetry=True, fast_threshold=0.0):
    tmp = Path(tempfile.mkdtemp())
    logs = tmp / "agent"
    logs.mkdir(parents=True, exist_ok=True)
    a = Terminus2Dynamic(logs_dir=logs, model_name="openai/x", max_turns=400,
                         api_base="http://127.0.0.1:1/v1", temperature=1.0, suppress_max_turns_warning=True,
                         model_info=MODEL_INFO, llm_kwargs={"api_key": "local"},
                         trusted_root=str(tmp / "trusted"),
                         compact_fast_below_s=fast_threshold, compact_reclaim_telemetry=telemetry,
                         fill_profile_cap_tokens=cap, **BASE)
    a._count_total_tokens = lambda chat: sum(len(m["content"]) for m in chat.messages)
    a._llm = types.SimpleNamespace(get_model_context_limit=lambda: 212992)
    a._dynwf_phase = 2
    a._dynwf_spec_for_reinject = "1. **Goal**"
    import time as _t
    a._dynwf_run_deadline = _t.time() + 5000
    return a


def summarize(a, chat):
    return asyncio.new_event_loop().run_until_complete(a._summarize(chat, "solve it", FakeSession()))


print("-- the knob defaults to 0 and is inert there")
off = build(cap=0)
check(off.fill_profile_cap_tokens == 0, "cap is 0", off.fill_profile_cap_tokens)
check(off._dynwf_fill_profile(FakeChat(), 2530) is None, "profile is None at cap 0")
STOCK_CALLS.clear()
chat_off = FakeChat()
handoff_off, refs_off = summarize(off, chat_off)
row_off = off._dynwf_compactions[-1]
check(row_off.get("fill_profile") is None, "OFF: no fill_profile on the row", row_off.get("fill_profile"))
check(row_off.get("tokens_before") == EXPECT_SUM, "reclaim telemetry still measures the whole chat", row_off.get("tokens_before"))

print("\n-- a negative cap is refused at construction")
try:
    build(cap=-1)
    check(False, "negative cap raises ValueError")
except ValueError as exc:
    check("must not be negative" in str(exc), "negative cap raises ValueError", exc)

print("\n-- the profile arithmetic, on a hand-computed history")
a = build(cap=CAP)
p = a._dynwf_fill_profile(FakeChat(), EXPECT_SUM)
check(p is not None, "profile is produced when the cap is set")
check(p["cap_tokens"] == CAP, "cap recorded", p["cap_tokens"])
check(p["messages"] == len(SIZES) and p["messages_counted"] == len(SIZES) and p["messages_skipped"] == 0,
      "every message was counted", (p["messages"], p["messages_counted"], p["messages_skipped"]))
check(p["sum_message_tokens"] == EXPECT_SUM, f"sum of parts = {EXPECT_SUM}", p["sum_message_tokens"])
check(p["chat_tokens"] == EXPECT_SUM, "whole-chat count recorded alongside the sum", p["chat_tokens"])
check(p["sum_over_whole_ratio"] == 1.0, "ratio of the two is recorded", p.get("sum_over_whole_ratio"))
check(p["by_role"] == {"system": 900, "assistant": 850, "user": 780}, "tokens by role", p["by_role"])
check(p["assistant_tokens"] == EXPECT_ASSISTANT, "assistant tokens", p["assistant_tokens"])
check(p["observation_tokens"] == EXPECT_OBSERVATION, "observation tokens = everything else", p["observation_tokens"])
check(p["assistant_tokens"] + p["observation_tokens"] == p["sum_message_tokens"],
      "the two partitions add back to the sum")
check([r["tokens"] for r in p["largest"]] == [900, 800, 500, 250, 50],
      "largest five, descending", [r["tokens"] for r in p["largest"]])
check(p["over_cap_messages"] == EXPECT_OVER_CAP, f"{EXPECT_OVER_CAP} observations over the cap",
      p["over_cap_messages"])
check(p["shadow_saved_tokens"] == EXPECT_SHADOW, f"shadow saving = {EXPECT_SHADOW}", p["shadow_saved_tokens"])
check(p["shadow_frac_of_chat"] == round(EXPECT_SHADOW / EXPECT_SUM, 4), "shadow as a fraction of the chat",
      p["shadow_frac_of_chat"])

print("\n-- the shadow EXCLUDES index 0 and assistant text, by construction")
huge = build(cap=CAP)
only_prompt = huge._dynwf_fill_profile(FakeChat([("system", 9000)]), 9000)
check(only_prompt["shadow_saved_tokens"] == 0, "a 9,000-token injected prompt at index 0 saves nothing",
      only_prompt["shadow_saved_tokens"])
only_model = huge._dynwf_fill_profile(FakeChat([("system", 10), ("assistant", 9000)]), 9010)
check(only_model["shadow_saved_tokens"] == 0, "a 9,000-token assistant message saves nothing",
      only_model["shadow_saved_tokens"])
one_obs = huge._dynwf_fill_profile(FakeChat([("system", 10), ("user", 9000)]), 9010)
check(one_obs["shadow_saved_tokens"] == 9000 - CAP, "a 9,000-token observation saves cap-complement",
      one_obs["shadow_saved_tokens"])
check(huge._dynwf_fill_profile(FakeChat([]), 0) is not None, "an empty chat profiles rather than raising")

print("\n-- THE SNAPSHOT: the profile describes the history that was REPLACED")
STOCK_CALLS.clear()
b = build(cap=CAP)
chat_b = FakeChat()
handoff_b, refs_b = summarize(b, chat_b)
row_b = b._dynwf_compactions[-1]
fp = row_b.get("fill_profile")
check(isinstance(fp, dict), "the row carries a fill_profile", type(fp).__name__)
check(len(chat_b.messages) == 3, "the chat really was rebuilt to three messages", len(chat_b.messages))
check(fp["messages"] == len(SIZES), "the profile still describes the PRE-compaction six", fp["messages"])
check(fp["sum_message_tokens"] == EXPECT_SUM, "and its token total is the pre-compaction one",
      fp["sum_message_tokens"])
check(fp["shadow_saved_tokens"] == EXPECT_SHADOW, "and its shadow is the pre-compaction one",
      fp["shadow_saved_tokens"])
EXPECT_AFTER = 900 + len("Q") + len("A")
check(row_b.get("tokens_after") == EXPECT_AFTER, "reclaim telemetry's post-compaction reading is unaffected",
      (row_b.get("tokens_after"), EXPECT_AFTER))

print("\n-- INERTNESS: on vs off, the agent behaves identically (the telemetry inertness property)")
check(handoff_b == handoff_off, "the handoff text is IDENTICAL with fill profiling on and off")
check(chat_b.messages == chat_off.messages, "the rebuilt history is IDENTICAL",
      (chat_b.messages, chat_off.messages))
check(refs_b == refs_off, "the subagent refs are identical", (refs_b, refs_off))
check(row_b["mode"] == "stock" and row_b["legs"] == 3, "single-call compaction is still off: stock, three legs", row_b)
check(len(STOCK_CALLS) == 1, "exactly one stock compaction ran", len(STOCK_CALLS))
diff = {k: (row_b.get(k), row_off.get(k)) for k in set(row_b) | set(row_off)
        if k != "fill_profile" and row_b.get(k) != row_off.get(k)}
check(not diff, "every other recorded field is identical", diff)
check(b._dynwf_reinjections == off._dynwf_reinjections, "the re-injection count is the same")

print("\n-- a counter failure degrades to a skipped message, never to an exception")
c = build(cap=CAP)


def selective(chat):
    if any(len(m["content"]) == 500 for m in chat.messages):
        raise RuntimeError("tokenizer unavailable for this one")
    return sum(len(m["content"]) for m in chat.messages)


c._count_total_tokens = selective
pc = c._dynwf_fill_profile(FakeChat(), None)
check(pc["messages_skipped"] == 1, "the one unmeasurable message is skipped", pc["messages_skipped"])
check(pc["messages_counted"] == len(SIZES) - 1, "the rest are still counted", pc["messages_counted"])
check(pc["sum_message_tokens"] == EXPECT_SUM - 500, "and the sum excludes it", pc["sum_message_tokens"])
check(pc["shadow_saved_tokens"] == 250 - CAP, "the shadow excludes it too", pc["shadow_saved_tokens"])
check("shadow_frac_of_chat" not in pc, "with chat_tokens None, no fraction is invented", pc.get("shadow_frac_of_chat"))

d = build(cap=CAP)


def boom(chat):
    raise RuntimeError("tokenizer unavailable")


d._count_total_tokens = boom
STOCK_CALLS.clear()
hd, _ = summarize(d, FakeChat())
rd = d._dynwf_compactions[-1]
check("STOCK HANDOFF" in hd, "a total counter failure still produces the handoff")
check(rd["fill_profile"]["messages_counted"] == 0 and rd["fill_profile"]["messages_skipped"] == len(SIZES),
      "and the profile records that nothing could be measured", rd.get("fill_profile"))
check(rd["fill_profile"]["shadow_saved_tokens"] == 0, "no shadow saving is invented")

print("\n-- the flag reaches the artifact so a reader can tell the profile was on")
check(int(getattr(a, "fill_profile_cap_tokens", 0)) == CAP, "the agent carries the cap", a.fill_profile_cap_tokens)

print()
if FAILS:
    print(f"FILL_PROFILE_GATE_FAILED: {len(FAILS)} check(s)")
    for f in FAILS:
        print("   -", f)
    raise SystemExit(1)
print("FILL_PROFILE_GATE_OK")

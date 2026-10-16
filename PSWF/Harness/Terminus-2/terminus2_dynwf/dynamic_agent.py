"""Explore a task, specify a workflow, and solve it using Harbor's Terminus-2 agent."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step

from terminus2_dynwf import wf_render
from terminus2_dynwf.spec_extract import (
    SPEC_MIN_CHARS_DEFAULT,
    extract_spec,
    marker_score,
    sanitize_spec,
)


_TRUNCATION_RETRY_SENTINEL = "ERROR!! NONE of the actions you just requested were performed"
_TRUNCATION_SWAP_MAX = 3

_PROACTIVE_CALL_SENTINEL = "_check_proactive_summarization("
_PROACTIVE_UNPACK_SENTINEL = "prompt, subagent_refs = proactive_summary_result"
_PHASE2_REPLAN_MAX = 3


class _ReplanStep:
    """Adapt a replan response for the workflow extraction interface."""

    source = "agent"
    message = ""

    def __init__(self, reasoning_content: str) -> None:
        self.reasoning_content = reasoning_content


class _PolicyStop(BaseException):
    """Stop a trial locally without entering the parent's retry handlers."""


class Terminus2Dynamic(Terminus2):
    """Run a two-phase workflow with configurable budget and context controls."""

    @staticmethod
    def name() -> str:
        return "terminus-2-dynwf"

    def version(self) -> str | None:
        return "dynwf-1.0.0/terminus-2.0.0"

    def __init__(
        self,
        *args: Any,
        tb_mode: str = "dynamic",
        wf_yaml: str | None = None,
        phase1_max_turns: int = 40,
        phase1_frac: float = 0.27,
        phase1_min_s: float = 60.0,
        phase1_max_s: float = 0.0,
        agent_budget_s: float = 7200.0,
        spec_min_chars: int = SPEC_MIN_CHARS_DEFAULT,
        spec_max_chars: int = 0,
        phase2_same_chat: bool = True,
        phase2_handoff: str = "",
        phase2_compact_min_frac: float = 0.0,
        phase2_restate_format: bool = True,
        phase1_wrapup_frac: float = 0.0,
        phase1_wrapup_turn_frac: float = 0.0,
        phase1_spec_reserve_s: float = 0.0,
        phase1_spec_reserve_turns: int = 2,
        phase1_forced_min_s: float = 90.0,
        phase1_single_confirm: bool = False,
        phase2_reinject_on_summarize: bool = False,
        phase1_hard_stop_frac: float = 0.0,
        phase1_hard_stop_turn_frac: float = 0.0,
        phase1_context_max_frac: float = 0.0,
        stop_on_policy_violation: bool = False,
        request_input_budget_guard: bool = False,
        phase1_budget_hint: bool = False,
        phase1_truncation_stop: bool = False,
        phase1_deadline_grace_s: float = 0.0,
        phase2_replan_before_compact: bool = False,
        phase2_replan_free_tokens: float = 32768.0,
        phase2_replan_min_free_tokens: float = 16384.0,
        phase2_replan_max_tokens: int = 8192,
        phase2_endgame_s: float = 0.0,
        phase2_replan_prefer_structured: bool = False,
        replan_trusted_dump: bool = False,
        trusted_root: str | None = None,
        compact_fast_below_s: float = 0.0,
        compact_reclaim_telemetry: bool = False,
        fill_profile_cap_tokens: int = 0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tb_mode = (tb_mode or "dynamic").strip().lower()
        self.stop_on_policy_violation = bool(stop_on_policy_violation)
        self.request_input_budget_guard = bool(request_input_budget_guard)
        self.phase1_budget_hint = bool(phase1_budget_hint)
        self.phase1_truncation_stop = bool(phase1_truncation_stop)
        self.phase1_deadline_grace_s = float(phase1_deadline_grace_s)
        if self.phase1_truncation_stop:
            source = inspect.getsource(Terminus2._query_llm)
            if _TRUNCATION_RETRY_SENTINEL not in source:
                raise RuntimeError(
                    "phase1_truncation_stop requires the pinned Terminus2 output-length recovery message"
                )
        self.phase2_replan_before_compact = bool(phase2_replan_before_compact)
        self.phase2_replan_free_tokens = float(phase2_replan_free_tokens)
        self.phase2_replan_min_free_tokens = float(phase2_replan_min_free_tokens)
        self.phase2_replan_max_tokens = int(phase2_replan_max_tokens)
        self.phase2_replan_prefer_structured = bool(phase2_replan_prefer_structured)
        self.replan_trusted_dump = bool(replan_trusted_dump)
        if self.phase2_replan_before_compact:
            if self.phase2_replan_min_free_tokens >= self.phase2_replan_free_tokens:
                raise ValueError(
                    "phase2_replan_min_free_tokens must be below phase2_replan_free_tokens"
                )
            if self.phase2_replan_max_tokens <= 0:
                raise ValueError("phase2_replan_max_tokens must be positive")
            source = inspect.getsource(Terminus2._run_agent_loop)
            if _PROACTIVE_CALL_SENTINEL not in source or _PROACTIVE_UNPACK_SENTINEL not in source:
                raise RuntimeError(
                    "phase2_replan_before_compact requires the pinned Terminus2 proactive-compaction contract"
                )
        self.wf_yaml = wf_yaml
        self.phase1_max_turns = int(phase1_max_turns)
        self.phase1_frac = float(phase1_frac)
        self.phase1_min_s = float(phase1_min_s)
        self.phase1_max_s = float(phase1_max_s)
        self.agent_budget_s = float(agent_budget_s)
        self.phase2_endgame_s = float(phase2_endgame_s)
        if self.phase2_endgame_s < 0:
            raise ValueError("phase2_endgame_s must not be negative")
        self.compact_fast_below_s = float(compact_fast_below_s)
        self.compact_reclaim_telemetry = bool(compact_reclaim_telemetry)
        self.fill_profile_cap_tokens = int(fill_profile_cap_tokens)
        if self.fill_profile_cap_tokens < 0:
            raise ValueError(
                f"fill_profile_cap_tokens ({self.fill_profile_cap_tokens}) must not be negative"
            )
        if self.compact_fast_below_s < 0:
            raise ValueError(f"compact_fast_below_s ({self.compact_fast_below_s}) must not be negative")
        if self.compact_fast_below_s >= self.agent_budget_s:
            raise ValueError(
                f"compact_fast_below_s ({self.compact_fast_below_s}) must be under agent_budget_s "
                f"({self.agent_budget_s}): above the budget it stops being a clock control"
            )
        if self.phase2_endgame_s >= self.agent_budget_s:
            raise ValueError(
                f"phase2_endgame_s ({self.phase2_endgame_s}) must be under agent_budget_s ({self.agent_budget_s})"
            )
        self.spec_min_chars = int(spec_min_chars)
        self.spec_max_chars = int(spec_max_chars)
        self.phase2_same_chat = bool(phase2_same_chat)
        self.phase2_handoff = (phase2_handoff or "").strip().lower()
        self.phase2_compact_min_frac = float(phase2_compact_min_frac)
        self.phase2_restate_format = bool(phase2_restate_format)
        self.phase1_wrapup_frac = float(phase1_wrapup_frac)
        self.phase1_wrapup_turn_frac = float(phase1_wrapup_turn_frac)
        self.phase1_spec_reserve_s = float(phase1_spec_reserve_s)
        self.phase1_spec_reserve_turns = int(phase1_spec_reserve_turns)
        self.phase1_forced_min_s = float(phase1_forced_min_s)
        self.phase1_single_confirm = bool(phase1_single_confirm)
        self.phase2_reinject_on_summarize = bool(phase2_reinject_on_summarize)
        self.phase1_hard_stop_frac = float(phase1_hard_stop_frac)
        self.phase1_hard_stop_turn_frac = float(phase1_hard_stop_turn_frac)
        self.phase1_context_max_frac = float(phase1_context_max_frac)
        if not 0 <= self.phase1_context_max_frac <= 0.5:
            raise ValueError('phase1_context_max_frac must be between 0 and 0.5')
        self.trusted_root = Path(trusted_root) if trusted_root else None
        self._dynwf_reset_phase_state()

        self._wf_doc = wf_render.load_wf(self.wf_yaml)
        self._max_turns_total = self._max_episodes

    def _reset_per_run_state(self) -> None:
        """Reset trial state and propagate the current session ID to the LLM client."""
        super()._reset_per_run_state()
        self._dynwf_reset_phase_state()
        llm = getattr(self, "_llm", None)
        if llm is not None and hasattr(llm, "_session_id"):
            llm._session_id = self._session_id

    def _install_policy_stop(self) -> None:
        """Stop the trial when the endpoint rejects a request with a policy violation."""
        if not self.stop_on_policy_violation:
            return
        llm = self._llm
        if getattr(llm, "_q38_policy_stop_installed", False):
            return
        original = getattr(llm, "_handle_litellm_error", None)
        if original is None:
            raise RuntimeError("the safety-stop profile requires the pinned LiteLLM error handler")

        def handle(error: Exception) -> Any:
            status = getattr(error, "status_code", None)
            response = getattr(error, "response", None)
            if status is None and response is not None:
                status = getattr(response, "status_code", None)
            detail = (str(error) + " " + repr(getattr(error, "body", None)))[:16000]
            refusal = re.search(
                r"model command batch refused before execution \(([a-z][a-z0-9_-]*)\)",
                detail,
            )
            if (status == 403
                    and ("guard_policy_violation" in detail or refusal is not None)
                    and "unparseable_command" not in detail):
                raise _PolicyStop(detail[:1000]) from error
            return original(error)

        llm._handle_litellm_error = handle
        llm._q38_policy_stop_installed = True


    def _install_request_input_budget_guard(self, meta: dict[str, Any]) -> None:
        """Count the complete request before transport and use the existing context fallback on overflow."""
        if not self.request_input_budget_guard:
            return
        from harbor.llms.base import ContextLengthExceededError

        state: dict[str, Any] = {
            "enabled": True, "checked_calls": 0, "blocked_calls": [],
            "input_limit_tokens": int(self._llm.get_model_context_limit()),
        }
        if state["input_limit_tokens"] <= 0:
            raise ValueError("request input budget must be positive")
        meta["request_input_budget_guard"] = state
        self._dynwf_request_budget_state = state
        llm = self._llm
        owner = getattr(llm, "_q38_request_budget_guard_owner", None)
        if owner is self:
            return
        if owner is not None:
            raise RuntimeError("request input budget guard belongs to another agent")
        original = llm.call

        async def checked_call(*args: Any, **kwargs: Any) -> Any:
            prompt = kwargs.get("prompt", args[0] if args else None)
            history = kwargs.get("message_history", args[1] if len(args) > 1 else [])
            if not isinstance(prompt, str) or not isinstance(history, (list, tuple)):
                raise TypeError("request budget guard requires text prompt and message history")
            if self._q38_custom_tokenizer() is None:
                raise RuntimeError("request budget guard requires the served model tokenizer")
            messages = list(history) + [{"role": "user", "content": prompt}]
            used = int(self._q38_count_messages(messages))
            current = self._dynwf_request_budget_state
            current["checked_calls"] += 1
            limit = current["input_limit_tokens"]
            if used > limit:
                current["blocked_calls"].append({
                    "check_index": current["checked_calls"],
                    "input_tokens": used, "input_limit_tokens": limit,
                    "message_count": len(messages),
                    "phase": getattr(self, "_dynwf_phase", 0),
                    "sent_to_model": False,
                })
                self.logger.info(
                    f"dynwf: request input budget exceeded locally ({used}>{limit}); "
                    "using the existing context recovery path before transport"
                )
                raise ContextLengthExceededError(
                    f"Request input budget exceeded before transport: {used} tokens, limit {limit}"
                )
            return await original(*args, **kwargs)

        llm.call = checked_call
        llm._q38_request_budget_guard_owner = self

    def _dynwf_reset_phase_state(self) -> None:
        """Reset phase-local counters and budget-control state."""
        self._dynwf_phase = 0
        self._dynwf_explore_deadline: float | None = None
        self._dynwf_explore_span = 0.0
        self._dynwf_wrapups = 0
        self._dynwf_force_confirm = False
        self._dynwf_spec_for_reinject = ""
        self._dynwf_reinjections = 0
        self._dynwf_hard_stops = 0
        self._dynwf_hard_stop_cmds = 0
        self._dynwf_phase1_truncations = 0
        self._dynwf_truncation_swaps = 0
        self._dynwf_exploration_closed = False
        self._dynwf_phase2_replans: list[dict[str, Any]] = []
        self._dynwf_run_deadline: float | None = None
        self._dynwf_phase2_endgames = 0
        self._dynwf_phase2_endgame_first_remaining: float | None = None
        self._dynwf_spec_is_refreshed = False
        self._dynwf_compactions: list[dict[str, Any]] = []

    def _dynwf_wrapup_text(self) -> str:
        """Render a wrap-up reminder when the wall-clock or turn budget reaches its threshold."""
        if self.phase1_wrapup_frac <= 0 and self.phase1_wrapup_turn_frac <= 0:
            return ""
        due = False
        seconds_left: float | None = None
        turns_left: int | None = None
        if self._dynwf_explore_deadline is not None and self._dynwf_explore_span > 0:
            seconds_left = self._dynwf_explore_deadline - time.time()
            spent = 1.0 - (max(0.0, seconds_left) / self._dynwf_explore_span)
            if self.phase1_wrapup_frac > 0 and spent >= self.phase1_wrapup_frac:
                due = True
        if self.phase1_max_turns > 0:
            turns_left = self.phase1_max_turns - self._n_episodes
            if (
                self.phase1_wrapup_turn_frac > 0
                and self._n_episodes >= self.phase1_wrapup_turn_frac * self.phase1_max_turns
            ):
                due = True
        if not due:
            return ""
        self._dynwf_wrapups += 1
        return wf_render.render_wrapup(
            self._wf_doc,
            turns_left=turns_left,
            seconds_left=seconds_left,
            short=self._dynwf_wrapups > 1,
        )

    def _dynwf_hard_stop_due(self) -> tuple[bool, float | None, int | None]:
        """Return whether the current exploration phase has reached a hard-stop threshold."""
        if getattr(self, "_dynwf_exploration_closed", False):
            seconds_left = (self._dynwf_explore_deadline - time.time()
                            if self._dynwf_explore_deadline is not None else None)
            turns_left = (self.phase1_max_turns - self._n_episodes
                          if self.phase1_max_turns > 0 else None)
            return True, seconds_left, turns_left
        if (self.phase1_hard_stop_frac <= 0 and self.phase1_hard_stop_turn_frac <= 0
                and self.phase1_context_max_frac <= 0):
            return False, None, None
        due = False
        seconds_left: float | None = None
        turns_left: int | None = None
        if self._dynwf_explore_deadline is not None and self._dynwf_explore_span > 0:
            seconds_left = self._dynwf_explore_deadline - time.time()
            spent = 1.0 - (max(0.0, seconds_left) / self._dynwf_explore_span)
            if self.phase1_hard_stop_frac > 0 and spent >= self.phase1_hard_stop_frac:
                due = True
        if self.phase1_max_turns > 0:
            turns_left = self.phase1_max_turns - self._n_episodes
            if (
                self.phase1_hard_stop_turn_frac > 0
                and self._n_episodes >= self.phase1_hard_stop_turn_frac * self.phase1_max_turns
            ):
                due = True
        if self.phase1_context_max_frac > 0:
            fraction = self._dynwf_phase1_context_frac()
            if fraction is None or fraction >= self.phase1_context_max_frac:
                due = True
        return due, seconds_left, turns_left

    async def _execute_commands(self, commands: Any, session: Any) -> tuple[bool, str]:
        """Apply phase-specific budget controls around the parent's command execution."""
        if getattr(self, "_dynwf_phase", 0) == 1 and commands:
            due, seconds_left, turns_left = self._dynwf_hard_stop_due()
            if due:
                self._dynwf_hard_stops += 1
                try:
                    self._dynwf_hard_stop_cmds += len(commands)
                except TypeError:
                    pass
                if self.phase1_single_confirm or getattr(self, "_dynwf_force_confirm", False):
                    self._pending_completion = True
                return False, wf_render.render_hard_stop(
                    self._wf_doc,
                    n_refused=len(commands) if hasattr(commands, "__len__") else 1,
                    turns_left=turns_left,
                    seconds_left=seconds_left,
                    short=self._dynwf_hard_stops > 1,
                )
        timeout_occurred, output = await super()._execute_commands(commands, session)
        if getattr(self, "_dynwf_phase", 0) == 2:
            endgame = self._dynwf_phase2_endgame_text()
            if endgame:
                output = f"{output}\n\n{endgame}"
            return timeout_occurred, output
        if getattr(self, "_dynwf_phase", 0) != 1:
            return timeout_occurred, output
        if self.phase1_single_confirm or getattr(self, "_dynwf_force_confirm", False):
            self._pending_completion = True
        nudge = self._dynwf_wrapup_text()
        if nudge:
            output = f"{output}\n\n{nudge}"
        return timeout_occurred, output

    async def _query_llm(self, chat: Any, prompt: str, original_instruction: str = "",
                         session: Any = None) -> Any:
        """Replace phase-1 truncation retries with a bounded request for the workflow."""
        if (self.phase1_truncation_stop
                and getattr(self, "_dynwf_phase", 0) == 1
                and isinstance(prompt, str)
                and prompt.startswith(_TRUNCATION_RETRY_SENTINEL)):
            self._dynwf_phase1_truncations += 1
            if self._dynwf_truncation_swaps < _TRUNCATION_SWAP_MAX:
                self._dynwf_truncation_swaps += 1
                self._dynwf_exploration_closed = True
                if self.phase1_single_confirm:
                    self._pending_completion = True
                prompt = wf_render.render_truncation_stop(
                    self._wf_doc, short=self._dynwf_truncation_swaps > 1
                )
                self.logger.warning(
                    "dynwf: phase-1 response hit the output cap; closing exploration and requesting the "
                    f"Specified Workflow (truncation {self._dynwf_phase1_truncations})"
                )
        return await super()._query_llm(chat, prompt, original_instruction, session)

    async def _check_proactive_summarization(
        self, chat: Any, original_instruction: str, session: Any
    ) -> tuple[str, Any] | None:
        """Refresh the plan before proactive compaction when the configured window permits it."""
        if not (self.phase2_replan_before_compact
                and getattr(self, "_dynwf_phase", 0) == 2
                and len(self._dynwf_phase2_replans) < _PHASE2_REPLAN_MAX):
            return await super()._check_proactive_summarization(chat, original_instruction, session)
        try:
            limit = int(self._llm.get_model_context_limit())
            used = int(self._count_total_tokens(chat))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"dynwf: could not measure the phase-2 context ({exc}); leaving compaction to the parent"
            )
            return await super()._check_proactive_summarization(chat, original_instruction, session)
        free = limit - used
        if free >= self.phase2_replan_free_tokens:
            return await super()._check_proactive_summarization(chat, original_instruction, session)
        row: dict[str, Any] = {
            "index": len(self._dynwf_phase2_replans) + 1, "at_used_tokens": used,
            "free_tokens": free, "limit_tokens": limit, "ok": False,
        }
        self._dynwf_phase2_replans.append(row)
        if free < self.phase2_replan_min_free_tokens:
            row["skipped_reason"] = (
                f"{free} free tokens is under the {int(self.phase2_replan_min_free_tokens)} floor"
            )
            return await super()._check_proactive_summarization(chat, original_instruction, session)
        t0 = time.time()
        summarizations_before = self._summarization_count
        prompt = wf_render.render_phase2_replan(
            self._wf_doc, free_tokens=free, index=row["index"],
            have_spec=bool(self._dynwf_spec_for_reinject),
        )
        self._append_user_step(prompt)
        saved = dict(self._llm_call_kwargs or {})
        capped = min(int(saved.get("max_tokens") or self.phase2_replan_max_tokens),
                     self.phase2_replan_max_tokens)
        row["max_tokens"] = capped
        try:
            self._llm_call_kwargs = {**saved, "max_tokens": capped}
            try:
                response = await self._query_llm(chat, prompt, original_instruction, session)
            finally:
                self._llm_call_kwargs = saved
        except Exception as exc:  # noqa: BLE001
            row.update(error=f"{type(exc).__name__}: {exc}", elapsed_s=round(time.time() - t0, 1))
            self.logger.warning(f"dynwf: phase-2 replan turn failed: {exc}")
            return await super()._check_proactive_summarization(chat, original_instruction, session)
        row["elapsed_s"] = round(time.time() - t0, 1)
        if self._summarization_count > summarizations_before:
            row["skipped_reason"] = "the parent's context recovery compacted during the replan turn"
            return None
        text = getattr(response, "content", "") or ""
        fresh, source, reason = self._dynwf_replan_candidates(response)
        rejected = "" if fresh else text
        fresh, truncated_from = wf_render.truncate_spec(fresh, self.spec_max_chars)
        row.update(response_chars=len(text), spec_source=source if fresh else "none",
                   spec_chars=len(fresh), spec_truncated_from=truncated_from, spec_reason=reason,
                   prefer_structured=bool(getattr(self, "phase2_replan_prefer_structured", False)))
        if fresh:
            row.update(ok=True, replaced_spec=True)
            row.update(self._dynwf_spec_witness(fresh))
            if self.replan_trusted_dump:
                name = f"specified_workflow_refreshed_{row['index']}.md"
                try:
                    (self._trusted_dir() / name).write_text(fresh)
                    row["spec_dump"] = name
                except Exception as exc:  # noqa: BLE001
                    row["spec_dump_error"] = f"{type(exc).__name__}: {exc}"
                    self.logger.warning(f"dynwf: could not dump the refreshed plan: {exc}")
            self._dynwf_spec_for_reinject = fresh
            self._dynwf_spec_is_refreshed = True
        else:
            row["rejected_chars"] = len(rejected)
        result = await self._summarize(chat, original_instruction, session)
        row["compacted_immediately"] = True
        return result

    def _dynwf_phase2_endgame_text(self) -> str:
        """Render an endgame reminder without refusing phase-2 commands."""
        try:
            if self.phase2_endgame_s <= 0 or getattr(self, "_dynwf_phase", 0) != 2:
                return ""
            deadline = getattr(self, "_dynwf_run_deadline", None)
            if deadline is None:
                return ""
            remaining = deadline - time.time()
            if remaining > self.phase2_endgame_s:
                return ""
            self._dynwf_phase2_endgames += 1
            if self._dynwf_phase2_endgame_first_remaining is None:
                self._dynwf_phase2_endgame_first_remaining = round(remaining, 1)
            return wf_render.render_phase2_endgame(
                self._wf_doc, seconds_left=remaining, short=self._dynwf_phase2_endgames > 1
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: phase-2 endgame nudge skipped ({exc})")
            return ""

    def _dynwf_replan_candidates(self, response: Any) -> tuple[str, str, str]:
        """Select a valid replan using the configured candidate-source ordering."""
        text = getattr(response, "content", "") or ""
        reasoning = getattr(response, "reasoning_content", "") or ""
        msgs = [{"role": "assistant", "content": text}]
        steps = [_ReplanStep(reasoning)]
        raw, source, cands = extract_spec(msgs, steps, min_chars=self.spec_min_chars)
        if not getattr(self, "phase2_replan_prefer_structured", False):
            fresh, _rejected, reason = sanitize_spec(raw, min_chars=self.spec_min_chars)
            return fresh, source, reason
        rung_rank = {"json_plan": 3, "trajectory_plan": 2, "assistant_text": 1, "reasoning_content": 0}
        ordered = sorted(
            cands,
            key=lambda c: (rung_rank.get(c.source, -1), marker_score(c.text), len(c.text)),
            reverse=True,
        )
        first_reason = ""
        for cand in ordered:
            fresh, _rejected, reason = sanitize_spec(cand.text, min_chars=self.spec_min_chars)
            if fresh:
                return fresh, cand.source, reason
            first_reason = first_reason or reason
        return "", "none", first_reason or "no candidate passed the predicate"

    @staticmethod
    def _dynwf_spec_witness(spec: str) -> dict[str, Any]:
        """Record a digest and representative lines of the exact plan being injected."""
        import hashlib
        lines = [ln.strip() for ln in (spec or "").splitlines()]
        usable = [ln for ln in lines if len(ln) >= 24]
        needles: list[str] = []
        if usable:
            for frac in (0.25, 0.5, 0.75):
                idx = min(len(usable) - 1, int(len(usable) * frac))
                cand = usable[idx][:160]
                if cand not in needles:
                    needles.append(cand)
        return {
            "spec_sha256": hashlib.sha256((spec or "").encode("utf-8", "replace")).hexdigest(),
            "spec_lines": len(lines),
            "spec_needles": needles,
        }

    def _dynwf_phase1_context_frac(self) -> float | None:
        """Return phase-1 context usage, or None when it cannot be counted."""
        try:
            limit = int(self._llm.get_model_context_limit())
            used = int(self._count_total_tokens(self._chat))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: could not measure phase-1 context ({exc}); compacting anyway")
            return None
        return round(used / limit, 4) if limit > 0 else None

    def _dynwf_chat_tokens(self, chat: Any) -> int | None:
        """Count the chat for telemetry without allowing counter errors to interrupt a trial."""
        if not getattr(self, "compact_reclaim_telemetry", False):
            return None
        try:
            return int(self._count_total_tokens(chat))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: compaction telemetry could not measure chat tokens ({exc})")
            return None

    def _dynwf_count_messages(self, messages: Any) -> int | None:
        """Count messages using the same tokenizer and accounting as the harness."""
        class _View:
            __slots__ = ("messages",)

            def __init__(self, msgs: Any) -> None:
                self.messages = msgs

        try:
            return int(self._count_total_tokens(_View(list(messages))))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: context profile could not count a message list ({exc})")
            return None

    @staticmethod
    def _dynwf_message_role(message: Any) -> str:
        """Return a message role for either mapping or object messages."""
        try:
            role = getattr(message, "role", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
            if hasattr(role, "value"):
                role = role.value
            return str(role) if role else "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    def _dynwf_fill_profile(self, chat: Any, chat_tokens: int | None) -> dict[str, Any] | None:
        """Describe context composition and hypothetical token savings without modifying messages."""
        cap = int(getattr(self, "fill_profile_cap_tokens", 0) or 0)
        if cap <= 0:
            return None
        try:
            messages = list(chat.messages)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: context profile could not read chat messages ({exc})")
            return None
        by_role: dict[str, int] = {}
        per_message: list[dict[str, Any]] = []
        total = 0
        skipped = 0
        shadow = 0
        over_cap = 0
        for index, message in enumerate(messages):
            count = self._dynwf_count_messages([message])
            if count is None:
                skipped += 1
                continue
            role = self._dynwf_message_role(message)
            by_role[role] = by_role.get(role, 0) + count
            total += count
            per_message.append({"index": index, "role": role, "tokens": count})
            if role != "assistant" and index > 0 and count > cap:
                over_cap += 1
                shadow += count - cap
        profile = {
            "cap_tokens": cap,
            "messages": len(messages),
            "messages_counted": len(per_message),
            "messages_skipped": skipped,
            "chat_tokens": chat_tokens,
            "sum_message_tokens": total,
            "by_role": by_role,
            "assistant_tokens": by_role.get("assistant", 0),
            "observation_tokens": sum(v for k, v in by_role.items() if k != "assistant"),
            "largest": sorted(per_message, key=lambda r: -r["tokens"])[:5],
            "over_cap_messages": over_cap,
            "shadow_saved_tokens": shadow,
        }
        if isinstance(chat_tokens, int) and chat_tokens > 0:
            profile["shadow_frac_of_chat"] = round(shadow / chat_tokens, 4)
            profile["sum_over_whole_ratio"] = round(total / chat_tokens, 4)
        return profile

    def _dynwf_compact_fast_due(self) -> tuple[bool, float | None]:
        """Select single-call compaction only inside the configured remaining-time window."""
        try:
            if self.compact_fast_below_s <= 0 or getattr(self, "_dynwf_phase", 0) != 2:
                return False, None
            deadline = getattr(self, "_dynwf_run_deadline", None)
            if deadline is None:
                return False, None
            remaining = deadline - time.time()
            if remaining <= 0:
                return False, round(remaining, 1)
            return remaining < self.compact_fast_below_s, round(remaining, 1)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: single-call compaction clock test failed ({exc}); compacting the stock way")
            return False, None

    async def _dynwf_fast_summarize(self, chat: Any, original_instruction: str, session: Any,
                                    remaining: float | None) -> Any:
        """Create a single-call summary and rebuild the handoff using the parent's format."""
        n = self._summarization_count + 1
        steps_to_include = 1 + (len(chat.messages) - 1) // 2
        summary_steps, _ = self._prepare_copied_trajectory_steps(steps_to_include)
        summary_prompt = f"""You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""
        self._summarization_count = n
        self._dynwf_compact_fast_called = True
        summary_response, summary_ref = await self._run_subagent(
            prompt=summary_prompt,
            message_history=chat.messages,
            steps=summary_steps,
            session_id=f"{self._session_id}-summarization-{n}-summary",
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=f"Context summarization {n}: Step 1 - Summary generation (single-call summary, "
                         f"{remaining if remaining is not None else '?'} s of budget left)",
            subagent_name_for_logging="summary generation LLM call",
        )
        summary_text = summary_response.content or ""
        try:
            current_screen = await session.capture_pane(capture_entire=False)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"dynwf: single-call compaction could not capture the pane ({exc})")
            current_screen = "(terminal screen unavailable)"
        chat._messages = [chat.messages[0]]
        chat.reset_response_chain()
        handoff_prompt = (
            "You are picking up your own work on this task after a context compaction. The conversation you "
            "were in has been replaced by the summary below, which you wrote a moment ago while that "
            "conversation was still in front of you.\n\n"
            f"**Original Task:** {original_instruction}\n\n"
            f"**Summary of the work so far:**\n{summary_text}\n\n"
            f"**Current Terminal Screen:**\n{current_screen}\n\n"
            "Continue working on this task from where you left off. You can no longer ask questions. Please "
            "follow the spec to interact with the terminal."
        )
        self._dynwf_compactions.append({
            "index": n, "mode": "fast", "legs": 1, "legs_saved": 2,
            "at_remaining_s": remaining, "threshold_s": self.compact_fast_below_s,
            "summary_chars": len(summary_text), "handoff_chars": len(handoff_prompt),
            "screen_chars": len(current_screen or ""),
        })
        return handoff_prompt, [summary_ref]

    async def _summarize(self, chat: Any, original_instruction: str, session: Any) -> Any:
        """Compact context, record optional telemetry, and re-inject the current plan."""
        fast, remaining = self._dynwf_compact_fast_due()
        tokens_before = self._dynwf_chat_tokens(chat)
        fill_snapshot = None
        if int(getattr(self, "fill_profile_cap_tokens", 0) or 0) > 0:
            try:
                fill_snapshot = type("_Snapshot", (), {"messages": list(chat.messages)})()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"dynwf: context profile could not snapshot the pre-compaction chat ({exc})")
        stock_start = self._summarization_count
        if fast:
            self._dynwf_compact_fast_called = False
            try:
                handoff, refs = await self._dynwf_fast_summarize(chat, original_instruction, session, remaining)
            except Exception as exc:  # noqa: BLE001
                if getattr(self, "_dynwf_compact_fast_called", False):
                    raise
                self.logger.warning(f"dynwf: single-call compaction fast compaction unusable before its call ({exc}); "
                                    f"falling back to the stock three-call path")
                self._summarization_count = stock_start
                fast = False
        if not fast:
            handoff, refs = await super()._summarize(chat, original_instruction, session)
            self._dynwf_compactions.append({
                "index": self._summarization_count, "mode": "stock", "legs": 3, "legs_saved": 0,
                "at_remaining_s": remaining, "threshold_s": self.compact_fast_below_s,
                "handoff_chars": len(handoff or ""),
            })
        if self._dynwf_compactions:
            row = self._dynwf_compactions[-1]
            row["tokens_before"] = tokens_before
            row["tokens_after"] = self._dynwf_chat_tokens(chat)
            if isinstance(row.get("tokens_before"), int) and isinstance(row.get("tokens_after"), int):
                row["reclaimed_tokens"] = row["tokens_before"] - row["tokens_after"]
            if fill_snapshot is not None:
                row["fill_profile"] = self._dynwf_fill_profile(fill_snapshot, tokens_before)
        spec = getattr(self, "_dynwf_spec_for_reinject", "")
        if getattr(self, "_dynwf_phase", 0) == 2 and spec:
            self._dynwf_reinjections += 1
            if getattr(self, "_dynwf_spec_is_refreshed", False):
                intro = (
                    "This is the UPDATED Specified Workflow you wrote a moment ago, after the work summarised "
                    "above and while that work was still in your context. The summary replaced the "
                    "conversation it was in, so the workflow is restated here verbatim. It is your current "
                    "plan: it already records what is done and verified, so continue from the first step it "
                    "leaves open, use its execution checkpoints, and before finishing verify every item of "
                    "its acceptance checklist. Where your own checks contradict it, the environment is the "
                    "source of truth."
                )
            else:
                intro = (
                    "This is the Specified Workflow you wrote for this task earlier in this session. The "
                    "summary above replaced the conversation it was in, so it is restated here verbatim. It "
                    "is still your plan: work through its steps in order, use its execution checkpoints, and "
                    "before finishing verify every item of its acceptance checklist. Where your own checks "
                    "contradict it, the environment is the source of truth."
                )
            handoff = (
                f"{handoff}\n\n{wf_render.SPEC_SECTION_HEADER}\n\n{intro}\n\n{spec}\n"
            )
        return handoff, refs

    def _append_user_step(self, message: str) -> None:
        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=message,
            )
        )

    def _trusted_dir(self) -> Path:
        """Return a host-side artifact directory outside the task container."""
        if self.trusted_root:
            base = self.trusted_root
        else:
            base = Path(self.logs_dir).parent.parent / "dynwf_trusted"
        d = base / (self._session_id or "unknown")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_artifacts(self, meta: dict[str, Any], files: dict[str, str]) -> None:
        """Write trial artifacts without turning an output failure into a failed trial."""
        for target, tag in ((Path(self.logs_dir), "logs"), (self._trusted_dir(), "trusted")):
            try:
                target.mkdir(parents=True, exist_ok=True)
                for fname, content in files.items():
                    if content:
                        (target / fname).write_text(content)
                (target / "dynwf_meta.json").write_text(json.dumps(meta, indent=2, default=str))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"dynwf: failed writing {tag} artifacts: {exc}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.tb_mode != "dynamic":
            return await super().run(instruction, environment, context)

        self._reset_per_run_state()
        self._chat = Chat(self._llm, interleaved_thinking=self._interleaved_thinking)
        self._context = context
        if self._session is None:
            raise RuntimeError("Session is not set")

        t_start = time.time()
        self._dynwf_run_deadline = t_start + self.agent_budget_s
        meta: dict[str, Any] = {
            "tb_mode": "dynamic",
            "session_id": self._session_id,
            "agent_budget_s": self.agent_budget_s,
            "phase1_max_turns": self.phase1_max_turns,
            "max_turns_total": self._max_turns_total,
            "wf_yaml": str(self.wf_yaml or wf_render.DEFAULT_WF_YAML),
        }
        spec = ""
        rejected = ""
        p1_prompt = p2_prompt = ""
        retired_usage = dict(input=0, output=0, cache=0, cost=0.0, details=[])

        try:
            self._install_policy_stop()
            self._install_request_input_budget_guard(meta)
            meta["stop_on_policy_violation"] = self.stop_on_policy_violation
            meta["phase1_budget_hint"] = self.phase1_budget_hint
            meta["phase1_truncation_stop"] = self.phase1_truncation_stop
            meta["phase1_deadline_grace_s"] = self.phase1_deadline_grace_s
            meta["phase2_replan_before_compact"] = self.phase2_replan_before_compact
            terminal_state = self._limit_output_length(await self._session.get_incremental_output())
            confirm_claim = (
                wf_render.CONFIRM_CLAIM_SINGLE
                if self.phase1_single_confirm
                else wf_render.CONFIRM_CLAIM_DOUBLE
            )
            p1_instr = wf_render.render_phase1(
                self._wf_doc, instruction, self.phase1_max_turns, confirm_claim
            )
            if not self.phase2_same_chat:
                p1_instr += (
                    "\n\nPhase-boundary handoff: after you emit your Specified Workflow, the raw "
                    "exploration history will be dropped to reserve context for execution. Your plan "
                    "must be self-contained: include the exact relevant paths, observed facts, current "
                    "working directory, required output, uncertainties, and concrete verification "
                    "commands within the existing five sections. Clearly separate observations from "
                    "assumptions. Aim for at most 8,000 characters. Stop exploring once the needed "
                    "facts are available; do not solve the task in this phase. End with the complete "
                    "plan, commands: [], and task_complete: true.\n"
                )
            p1_budget = max(self.phase1_min_s, self.phase1_frac * self.agent_budget_s)
            if self.phase1_max_s > 0:
                p1_budget = max(self.phase1_min_s, min(p1_budget, self.phase1_max_s))
            reserve = max(0.0, min(self.phase1_spec_reserve_s, max(0.0, p1_budget - self.phase1_min_s)))
            explore_budget = p1_budget - reserve
            if self.phase1_budget_hint:
                import math
                p1_instr += "\n\n" + wf_render.render_budget_hint(
                    explore_s=explore_budget,
                    reserve_s=reserve,
                    phase1_max_turns=self.phase1_max_turns,
                    hard_stop_s=(self.phase1_hard_stop_frac * explore_budget
                                 if self.phase1_hard_stop_frac > 0 else None),
                    hard_stop_turns=(math.ceil(self.phase1_hard_stop_turn_frac * self.phase1_max_turns)
                                     if self.phase1_hard_stop_turn_frac > 0 and self.phase1_max_turns > 0 else None),
                    nudge_s=(self.phase1_wrapup_frac * explore_budget if self.phase1_wrapup_frac > 0 else None),
                    nudge_turns=(math.ceil(self.phase1_wrapup_turn_frac * self.phase1_max_turns)
                                 if self.phase1_wrapup_turn_frac > 0 and self.phase1_max_turns > 0 else None),
                    context_max_frac=(self.phase1_context_max_frac if self.phase1_context_max_frac > 0 else None),
                    truncation_stop=self.phase1_truncation_stop,
                    grace_s=(max(0.0, min(self.phase1_deadline_grace_s,
                                          reserve - self.phase1_forced_min_s))
                             if self.phase1_deadline_grace_s > 0 else None),
                )
            p1_prompt = self._prompt_template.format(
                instruction=p1_instr, terminal_state=terminal_state
            )
            self._append_user_step(p1_prompt)

            self._max_episodes = self.phase1_max_turns
            self._dynwf_phase = 1
            self._dynwf_explore_span = explore_budget
            self._dynwf_explore_deadline = time.time() + explore_budget
            phase1_stopped_by = "completed"
            grace_budget = 0.0
            if self.phase1_deadline_grace_s > 0:
                grace_budget = max(0.0, min(self.phase1_deadline_grace_s,
                                            reserve - self.phase1_forced_min_s))
            grace: dict[str, Any] = {"configured_s": round(self.phase1_deadline_grace_s, 1),
                                     "available_s": round(grace_budget, 1), "granted": False}
            loop_task = asyncio.ensure_future(
                self._run_agent_loop(
                    initial_prompt=p1_prompt, chat=self._chat, original_instruction=p1_instr
                )
            )
            try:
                await asyncio.wait_for(asyncio.shield(loop_task), timeout=explore_budget)
            except asyncio.TimeoutError:
                phase1_stopped_by = "phase_deadline"
                told_to_deliver = bool(self._dynwf_hard_stops or self._dynwf_wrapups
                                       or self._dynwf_phase1_truncations)
                grace["already_instructed"] = told_to_deliver
                if grace_budget > 0 and told_to_deliver:
                    self._dynwf_exploration_closed = True
                    grace["granted"] = True
                    t_g = time.time()
                    try:
                        await asyncio.wait_for(asyncio.shield(loop_task), timeout=grace_budget)
                        phase1_stopped_by = "completed"
                        grace["landed"] = True
                    except asyncio.TimeoutError:
                        grace["landed"] = False
                    grace["waited_s"] = round(time.time() - t_g, 1)
                elif grace_budget <= 0:
                    grace["skipped_reason"] = "no grace configured or reserve too small"
                else:
                    grace["skipped_reason"] = "the model had not yet been told to deliver the workflow"
            finally:
                if not loop_task.done():
                    loop_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await loop_task
            meta["phase1_deadline_grace"] = grace

            p1_calls = len(self._api_request_times)
            p1_episodes = self._n_episodes

            raw, spec_source, cands = extract_spec(
                self._chat.messages, self._trajectory_steps, min_chars=self.spec_min_chars
            )
            spec, rejected, spec_reason = sanitize_spec(raw, min_chars=self.spec_min_chars)
            have_spec = bool(spec)

            forced: dict[str, Any] = {"attempted": False}
            if not have_spec and reserve > 0:
                left = (t_start + p1_budget) - time.time()
                forced.update(attempted=True, budget_s=round(left, 1), skipped_reason=None)
                if left < self.phase1_forced_min_s:
                    forced.update(attempted=False, skipped_reason=f"only {left:.0f}s of reserve left")
                else:
                    fs_prompt = wf_render.render_forced_spec(self._wf_doc)
                    self._append_user_step(fs_prompt)
                    self._max_episodes = max(1, self.phase1_spec_reserve_turns)
                    self._pending_completion = True
                    self._dynwf_force_confirm = True
                    try:
                        await asyncio.wait_for(
                            self._run_agent_loop(
                                initial_prompt=fs_prompt,
                                chat=self._chat,
                                original_instruction=p1_instr,
                            ),
                            timeout=left,
                        )
                        forced["stopped_by"] = "completed"
                    except asyncio.TimeoutError:
                        forced["stopped_by"] = "deadline"
                    except Exception as exc:  # noqa: BLE001
                        forced["stopped_by"] = f"error: {type(exc).__name__}"
                        self.logger.warning(f"dynwf: forced spec turn failed: {exc}")
                    finally:
                        self._dynwf_force_confirm = False
                        self._max_episodes = self.phase1_max_turns
                    forced["episodes"] = self._n_episodes
                    p1_episodes += self._n_episodes
                    p1_calls = len(self._api_request_times)
                    raw, spec_source, cands = extract_spec(
                        self._chat.messages, self._trajectory_steps, min_chars=self.spec_min_chars
                    )
                    spec, rejected, spec_reason = sanitize_spec(raw, min_chars=self.spec_min_chars)
                    forced["gained_spec"] = bool(spec) and not have_spec
                    have_spec = bool(spec)

            p1_msgs = len(self._chat.messages)
            p1_elapsed = time.time() - t_start

            spec, spec_truncated_from = wf_render.truncate_spec(spec, self.spec_max_chars)

            meta.update(
                phase1_stopped_by=phase1_stopped_by,
                phase1_explore_budget_s=round(explore_budget, 1),
                phase1_reserve_s=round(reserve, 1),
                phase1_wrapups=self._dynwf_wrapups,
                phase1_single_confirm=self.phase1_single_confirm,
                phase1_hard_stop_frac=self.phase1_hard_stop_frac,
                phase1_hard_stop_turn_frac=self.phase1_hard_stop_turn_frac,
                phase1_hard_stops=self._dynwf_hard_stops,
                phase1_hard_stop_cmds_refused=self._dynwf_hard_stop_cmds,
                phase1_truncations=self._dynwf_phase1_truncations,
                phase1_truncation_swaps=self._dynwf_truncation_swaps,
                phase1_exploration_closed_early=self._dynwf_exploration_closed,
                phase1_context_max_frac=self.phase1_context_max_frac,
                phase1_context_frac=self._dynwf_phase1_context_frac(),
                phase1_forced_spec=forced,
                spec_truncated_from=spec_truncated_from,
                spec_max_chars=self.spec_max_chars,
                phase1_calls=p1_calls,
                phase1_episodes=p1_episodes,
                phase1_msgs=p1_msgs,
                phase1_elapsed_s=round(p1_elapsed, 1),
                phase1_budget_s=round(p1_budget, 1),
                spec_source=spec_source if have_spec else "none",
                spec_source_raw=spec_source,
                spec_chars=len(spec),
                spec_rejected_chars=len(rejected),
                spec_reason=spec_reason,
                have_spec=have_spec,
                spec_candidates=[
                    {"source": c.source, "chars": len(c.text), "markers": c.score, **c.detail}
                    for c in cands
                ],
            )
            if not have_spec:
                self.logger.warning(
                    f"dynwf: no usable Specified Workflow ({spec_reason}) — phase 2 degrades to vanilla"
                )

            self._dynwf_phase = 2
            self._dynwf_explore_deadline = None
            self._dynwf_exploration_closed = False
            self._dynwf_spec_for_reinject = ""

            mode = self.phase2_handoff or ("same_chat" if self.phase2_same_chat else "fresh_chat")
            compaction: dict[str, Any] = {"requested": mode == "compact", "applied": False}
            if mode == "fresh_chat" and not have_spec:
                mode = "same_chat"
                compaction["skipped_reason"] = "no valid plan; retain observations"
            meta["plan_compaction"] = {
                "requested": not self.phase2_same_chat,
                "applied": mode == "fresh_chat",
                "additional_llm_calls": 0,
                "retained_plan_chars": len(spec) if have_spec else 0,
            }
            if mode == "compact":
                frac = self._dynwf_phase1_context_frac()
                compaction["phase1_context_frac"] = frac
                compaction["min_frac"] = self.phase2_compact_min_frac
                if frac is not None and frac < self.phase2_compact_min_frac:
                    mode = "same_chat"
                    compaction["skipped_reason"] = (
                        f"phase-1 history is {frac:.4f} of the context window, under "
                        f"{self.phase2_compact_min_frac}"
                    )
                else:
                    t_c = time.time()
                    try:
                        handoff_text, refs = await self._summarize(self._chat, instruction, self._session)
                        if not (handoff_text or "").strip():
                            raise RuntimeError("summarizer returned an empty handoff")
                        compaction.update(
                            applied=True, handoff_chars=len(handoff_text),
                            subagent_refs=len(refs or []), elapsed_s=round(time.time() - t_c, 1),
                        )
                        if refs:
                            self._pending_subagent_refs = refs
                    except Exception as exc:  # noqa: BLE001
                        mode = "same_chat"
                        compaction.update(
                            applied=False, error=f"{type(exc).__name__}: {exc}",
                            elapsed_s=round(time.time() - t_c, 1),
                        )
                        self.logger.warning(f"dynwf: boundary compaction failed, using same_chat: {exc}")

            same_chat = mode == "same_chat"
            p2_prompt, p2_variant = wf_render.render_phase2(
                self._wf_doc,
                problem_statement=instruction,
                spec=spec if have_spec else None,
                same_chat=same_chat,
                restate_format=self.phase2_restate_format,
                handoff=mode,
            )
            if mode == "compact" and compaction.get("applied"):
                p2_prompt = (
                    f"{p2_prompt}\n\n{wf_render.COMPACTED_NOTES_HEADER}\n\n{handoff_text}\n"
                )

            self._pending_completion = False
            remaining = self._max_turns_total - p1_episodes
            self._max_episodes = max(1, remaining)
            self._dynwf_spec_for_reinject = (
                spec if (have_spec and self.phase2_reinject_on_summarize) else ""
            )

            if not same_chat:
                retired_usage = dict(
                    input=self._chat.total_input_tokens, output=self._chat.total_output_tokens,
                    cache=self._chat.total_cache_tokens, cost=self._chat.total_cost,
                    details=list(self._chat.rollout_details),
                )
                meta['phase1_usage_before_chat_reset'] = {k: v for k, v in retired_usage.items() if k != 'details'}
                self._chat = Chat(self._llm, interleaved_thinking=self._interleaved_thinking)
                ts2 = self._limit_output_length(await self._session.get_incremental_output())
                p2_prompt = self._prompt_template.format(instruction=p2_prompt, terminal_state=ts2)

            self._append_user_step(p2_prompt)
            meta.update(
                phase2_handoff=mode,
                phase2_handoff_requested=(
                    self.phase2_handoff or ("same_chat" if self.phase2_same_chat else "fresh_chat")
                ),
                phase2_boundary_compaction=compaction,
                phase1_max_s=self.phase1_max_s,
                phase2_variant=p2_variant,
                phase2_spec_injected=bool(have_spec and wf_render.SPEC_SECTION_HEADER in p2_prompt),
                phase2_max_episodes=self._max_episodes,
                phase2_prompt_chars=len(p2_prompt),
                memory_claim={
                    "same_chat": wf_render.MEMORY_CLAIM_SAME,
                    "fresh_chat": wf_render.MEMORY_CLAIM_FRESH,
                    "compact": wf_render.MEMORY_CLAIM_COMPACT,
                }[mode],
                spec_needles=wf_render.spec_needles(spec) if have_spec else [],
            )

            await self._run_agent_loop(
                initial_prompt=p2_prompt, chat=self._chat, original_instruction=instruction
            )

        except _PolicyStop as exc:
            meta["safety_stop"] = {
                "kind": "guard_policy_violation", "phase": self._dynwf_phase,
                "immediate": True, "error": str(exc), "further_actions": False,
            }
            self.logger.warning("dynwf: safety guard stopped this trial before command execution")
        finally:
            try:
                context.rollout_details = (
                    retired_usage['details'] + (self._chat.rollout_details if self._chat else []) + self._subagent_rollout_details
                )
                if self._chat is not None:
                    context.n_input_tokens = (
                        retired_usage['input'] + self._chat.total_input_tokens + self._subagent_metrics.total_prompt_tokens
                    )
                    context.n_output_tokens = (
                        retired_usage['output'] + self._chat.total_output_tokens + self._subagent_metrics.total_completion_tokens
                    )
                    context.n_cache_tokens = (
                        retired_usage['cache'] + self._chat.total_cache_tokens + self._subagent_metrics.total_cached_tokens
                    )
                    total_cost = retired_usage['cost'] + self._chat.total_cost + self._subagent_metrics.total_cost_usd
                    context.cost_usd = total_cost if total_cost > 0 else None
                context.metadata = {
                    "n_episodes": self._n_episodes,
                    "api_request_times_msec": self._api_request_times,
                    "summarization_count": self._summarization_count,
                    "dynwf": meta,
                }
                meta["total_calls"] = len(self._api_request_times)
                meta["total_episodes_phase2"] = self._n_episodes
                meta["wall_s"] = round(time.time() - t_start, 1)
                meta["summarization_count"] = self._summarization_count
                meta["phase2_spec_reinjections"] = getattr(self, "_dynwf_reinjections", 0)
                meta["compact_fast"] = {
                    "reclaim_telemetry": bool(getattr(self, "compact_reclaim_telemetry", False)),
                    "fill_profile_cap_tokens": int(getattr(self, "fill_profile_cap_tokens", 0) or 0),
                    "enabled": self.compact_fast_below_s > 0,
                    "threshold_s": self.compact_fast_below_s,
                    "events": list(getattr(self, "_dynwf_compactions", [])),
                    "fast_events": sum(1 for c in getattr(self, "_dynwf_compactions", [])
                                       if c.get("mode") == "fast"),
                    "legs_saved": sum(int(c.get("legs_saved") or 0)
                                      for c in getattr(self, "_dynwf_compactions", [])),
                }
                meta["phase2_endgame"] = {
                    "enabled": self.phase2_endgame_s > 0,
                    "reserve_s": self.phase2_endgame_s,
                    "events": self._dynwf_phase2_endgames,
                    "first_fired_at_remaining_s": self._dynwf_phase2_endgame_first_remaining,
                }
                meta["phase2_replan"] = {
                    "enabled": self.phase2_replan_before_compact,
                    "trigger_free_tokens": self.phase2_replan_free_tokens,
                    "min_free_tokens": self.phase2_replan_min_free_tokens,
                    "max_response_tokens": self.phase2_replan_max_tokens,
                    "max_events": _PHASE2_REPLAN_MAX,
                    "spec_refreshed": bool(getattr(self, "_dynwf_spec_is_refreshed", False)),
                    "prefer_structured": bool(getattr(self, "phase2_replan_prefer_structured", False)),
                    "trusted_dump": bool(getattr(self, "replan_trusted_dump", False)),
                    "events": list(getattr(self, "_dynwf_phase2_replans", [])),
                }
                meta["phase1_wrapups"] = getattr(self, "_dynwf_wrapups", 0)
                meta["phase1_hard_stops"] = getattr(self, "_dynwf_hard_stops", 0)
                meta["phase1_truncations"] = getattr(self, "_dynwf_phase1_truncations", 0)
                self._write_artifacts(
                    meta,
                    {
                        "phase1_prompt.md": p1_prompt,
                        "phase2_prompt.md": p2_prompt,
                        "specified_workflow.md": spec,
                        "specified_workflow.rejected.md": (
                            f"<!-- rejected: {meta.get('spec_reason')} chars={len(rejected)} -->\n{rejected}"
                            if rejected
                            else ""
                        ),
                        "phase1_calls.txt": str(meta.get("phase1_calls", "")),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"dynwf: finally block error: {exc}")
            finally:
                self._dump_trajectory()

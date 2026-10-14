"""Synthetic regression tests for request input budget."""
import ast
import asyncio
import copy
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic, _PolicyStop


def count(messages):
    return sum(len(m.get('content') or '') + len(m.get('reasoning_content') or '') for m in messages)


class RecordingLLM:
    def __init__(self, limit=100):
        self.limit = limit
        self.calls = []
        self.failure = None
    def get_model_context_limit(self):
        return self.limit
    def get_model_output_limit(self):
        return 2000
    async def call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.failure is not None:
            raise self.failure
        return LLMResponse(content='fixture response')


def make_holder(limit=100, enabled=True):
    llm = RecordingLLM(limit)
    holder = SimpleNamespace(
        request_input_budget_guard=enabled, _llm=llm, _dynwf_phase=2,
        _q38_custom_tokenizer=lambda: {'type': 'fixture'},
        _q38_count_messages=count, logger=logging.getLogger('budget-test'),
    )
    meta = {}
    Terminus2Dynamic._install_request_input_budget_guard(holder, meta)
    return holder, meta


class RequestBudgetTests(unittest.TestCase):
    def test_default_off_is_inert(self):
        holder, meta = make_holder(enabled=False)
        response = asyncio.run(holder._llm.call('x' * 1000))
        self.assertEqual(response.content, 'fixture response')
        self.assertEqual(len(holder._llm.calls), 1)
        self.assertEqual(meta, {})

    def test_fitting_request_preserves_arguments_and_history(self):
        holder, meta = make_holder()
        history = [{'role': 'assistant', 'content': 'answer', 'reasoning_content': 'thought'}]
        before = copy.deepcopy(history)
        kwargs = {'prompt': 'next', 'message_history': history, 'temperature': 1.0,
                  'max_tokens': 32768, 'extra_body': {'top_k': 20}}
        response = asyncio.run(holder._llm.call(**kwargs))
        self.assertEqual(response.content, 'fixture response')
        self.assertEqual(holder._llm.calls[0], ((), kwargs))
        self.assertIs(holder._llm.calls[0][1]['message_history'], history)
        self.assertEqual(history, before)
        self.assertEqual(meta['request_input_budget_guard']['checked_calls'], 1)
        self.assertEqual(meta['request_input_budget_guard']['blocked_calls'], [])

    def test_pending_prompt_is_counted_and_overflow_never_delegates(self):
        holder, meta = make_holder()
        history = [{'role': 'user', 'content': 'h' * 90}]
        asyncio.run(holder._llm.call('p' * 10, history))
        self.assertEqual(len(holder._llm.calls), 1)
        before = copy.deepcopy(history)
        with self.assertRaises(ContextLengthExceededError):
            asyncio.run(holder._llm.call('p' * 11, history))
        self.assertEqual(len(holder._llm.calls), 1)
        self.assertEqual(history, before)
        event = meta['request_input_budget_guard']['blocked_calls'][0]
        self.assertEqual(event['input_tokens'], 101)
        self.assertFalse(event['sent_to_model'])

    def test_reasoning_is_included(self):
        holder, meta = make_holder()
        history = [{'role': 'assistant', 'content': 'a', 'reasoning_content': 'r' * 100}]
        with self.assertRaises(ContextLengthExceededError):
            asyncio.run(holder._llm.call('next', message_history=history))
        self.assertEqual(holder._llm.calls, [])
        self.assertGreater(meta['request_input_budget_guard']['blocked_calls'][0]['input_tokens'], 100)

    def test_tokenizer_or_counter_failure_prevents_transport(self):
        holder, meta = make_holder()
        holder._q38_custom_tokenizer = lambda: None
        with self.assertRaisesRegex(RuntimeError, 'served model tokenizer'):
            asyncio.run(holder._llm.call('next'))
        self.assertEqual(holder._llm.calls, [])
        holder._q38_custom_tokenizer = lambda: {'type': 'fixture'}
        def fail(_):
            raise ValueError('counter failed')
        holder._q38_count_messages = fail
        with self.assertRaisesRegex(ValueError, 'counter failed'):
            asyncio.run(holder._llm.call('next'))
        self.assertEqual(holder._llm.calls, [])

    def test_new_run_rebinds_metrics_without_stacking_wrappers(self):
        holder, first = make_holder()
        wrapped = holder._llm.call
        asyncio.run(holder._llm.call('first'))
        second = {}
        Terminus2Dynamic._install_request_input_budget_guard(holder, second)
        self.assertIs(holder._llm.call, wrapped)
        asyncio.run(holder._llm.call('second'))
        self.assertEqual(first['request_input_budget_guard']['checked_calls'], 1)
        self.assertEqual(second['request_input_budget_guard']['checked_calls'], 1)
        self.assertEqual(len(holder._llm.calls), 2)

    def test_policy_stop_passes_through_unchanged(self):
        holder, meta = make_holder()
        failure = _PolicyStop('fixture policy refusal')
        holder._llm.failure = failure
        with self.assertRaises(_PolicyStop) as caught:
            asyncio.run(holder._llm.call('next'))
        self.assertIs(caught.exception, failure)
        self.assertEqual(len(holder._llm.calls), 1)

    def test_existing_summary_fit_is_accepted_and_retains_summary_tail(self):
        holder, meta = make_holder(limit=6000)
        history = [{'role': 'user', 'content': 'task'}]
        history += [{'role': 'user', 'content': 'u' * 2000}, {'role': 'assistant', 'content': 'a' * 2000}] * 2
        history += [{'role': 'user', 'content': 'summary request'}, {'role': 'assistant', 'content': 'summary' * 100}]
        tail = copy.deepcopy(history[-2:])
        fitted = Terminus2._q38_fit_subagent_history(holder, history, 'questions', 'fixture')
        self.assertEqual(fitted[-2:], tail)
        self.assertLessEqual(count(fitted + [{'role': 'user', 'content': 'questions'}]), 6000 - 2048)
        asyncio.run(holder._llm.call('questions', message_history=fitted))
        self.assertEqual(len(holder._llm.calls), 1)
        self.assertEqual(meta['request_input_budget_guard']['blocked_calls'], [])

    def test_actual_parent_truncation_recursion_uses_local_context_fallback(self):
        fixture_tree = ast.parse((Path(__file__).parent / 'test_run_orchestration.py').read_text())
        selected = [n for n in fixture_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and n.name in {'FakeSession', 'make_agent'}]
        namespace = {'Path': Path, 'Terminus2Dynamic': Terminus2Dynamic}
        exec(compile(ast.Module(body=selected, type_ignores=[]), 'fixtures', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as tmp:
            agent = namespace['make_agent'](Path(tmp), request_input_budget_guard=True)
            llm = RecordingLLM(6000)
            transports = []
            async def transport(*args, **kwargs):
                prompt = kwargs.get('prompt', args[0] if args else '')
                history = kwargs.get('message_history', args[1] if len(args) > 1 else [])
                used = count(list(history) + [{'role': 'user', 'content': prompt}])
                transports.append(used)
                if used > 6000:
                    raise AssertionError('oversized request reached transport')
                if len(transports) == 1:
                    raise OutputLengthExceededError('fixture truncated output', 'x' * 2000)
                return LLMResponse(content='valid continuation')
            llm.call = transport
            agent._llm = llm
            agent._q38_custom_tokenizer = lambda: {'type': 'fixture'}
            agent._q38_count_messages = count
            agent._count_total_tokens = lambda chat: count(chat.messages)
            agent._llm_call_kwargs = {'max_tokens': 2000}
            agent._enable_summarize = True
            chat = Chat(llm)
            chat._messages = [{'role': 'user', 'content': 'h' * 5200}]
            summaries = []
            async def summarize(current, instruction, session):
                summaries.append(instruction)
                current._messages = [{'role': 'user', 'content': 'task'}]
                current.reset_response_chain()
                return 'continue safely', []
            agent._summarize = summarize
            meta = {}
            agent._install_request_input_budget_guard(meta)
            response = asyncio.run(agent._query_llm(chat, 'p' * 100, 'task', object()))
            self.assertEqual(response.content, 'valid continuation')
            self.assertEqual(len(transports), 2)
            self.assertTrue(all(n <= 6000 for n in transports))
            self.assertEqual(summaries, ['task'])
            self.assertEqual(len(meta['request_input_budget_guard']['blocked_calls']), 1)
            self.assertEqual(agent._pending_handoff_prompt, 'continue safely')


if __name__ == '__main__':
    unittest.main(verbosity=2)

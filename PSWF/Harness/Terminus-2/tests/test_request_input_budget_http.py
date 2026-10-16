"""Synthetic regression tests for request input budget http."""
import ast
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic

fixture_tree = ast.parse((Path(__file__).parent / 'test_run_orchestration.py').read_text())
selected = [n for n in fixture_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in {'FakeSession', 'make_agent'}]
namespace = {'Path': Path, 'Terminus2Dynamic': Terminus2Dynamic}
exec(compile(ast.Module(body=selected, type_ignores=[]), 'orchestration-fixtures', 'exec'), namespace)


class NativeCounter:
    def __init__(self):
        path = Path(os.environ['Q38_TOKENIZER_JSON'])
        self.tokenizer = Tokenizer.from_file(str(path))
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                           extensions=['jinja2.ext.loopcontrols'])
        def fail(message):
            raise ValueError(message)
        env.globals['raise_exception'] = fail
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, **kwargs)
        self.template = env.from_string((path.parent / 'chat_template.jinja').read_text())
    def __call__(self, messages):
        rendered = self.template.render(messages=messages, tools=None,
            add_generation_prompt=True, enable_thinking=True,
            preserve_thinking=True, reasoning_effort='xhigh')
        return len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        tokens = self.server.counter(body['messages'])
        completion = body['max_tokens']
        self.server.inputs.append(tokens)
        if tokens + completion > 262144:
            self.server.overflows += 1
            status = 400
            reply = {'object': 'error', 'message':
                f"Requested token count exceeds the model's maximum context length of 262144 tokens. You requested a total of {tokens + completion} tokens: {tokens} tokens from the input messages and {completion} tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit.",
                'type': 'BadRequestError', 'param': None, 'code': 400}
        else:
            status = 200
            first = len(self.server.inputs) == 1
            text = 'x ' * 32766 if first else 'valid continuation'
            count = 32768 if first else 3
            reply = {'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': 'amazon/LAR-27B',
                'choices': [{'index': 0, 'message': {'role': 'assistant',
                    'content': text, 'reasoning_content': 'r'},
                    'finish_reason': 'length' if first else 'stop'}],
                'usage': {'prompt_tokens': tokens, 'completion_tokens': count,
                    'total_tokens': tokens + count,
                    'completion_tokens_details': {'reasoning_tokens': 1}}}
        raw = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class RequestBudgetHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = NativeCounter()
    def run_case(self, enabled):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.inputs = []
        server.overflows = 0
        server.counter = self.counter
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            llm = LiteLLM('openai/amazon/LAR-27B', temperature=1.0,
                api_base=f'http://127.0.0.1:{server.server_port}/v1', api_key='local',
                timeout=10, num_retries=10,
                model_info={'max_input_tokens': 212992, 'max_output_tokens': 32768,
                    'input_cost_per_token': 0, 'output_cost_per_token': 0,
                    'cache_creation_input_token_cost': 0, 'cache_read_input_token_cost': 0,
                    'mode': 'chat', 'supports_reasoning': True})
            with tempfile.TemporaryDirectory() as tmp:
                agent = namespace['make_agent'](Path(tmp), request_input_budget_guard=enabled)
                agent._llm = llm
                agent._model_name = 'openai/amazon/LAR-27B'
                agent._llm_call_kwargs = {'max_tokens': 32768, 'top_p': 0.95,
                    'extra_body': {'chat_template_kwargs': {'enable_thinking': True,
                        'preserve_thinking': True, 'reasoning_effort': 'xhigh'}}}
                agent._enable_summarize = True
                chat = Chat(llm, interleaved_thinking=True)
                chat._messages = [{'role': 'user', 'content': 'history ' * 203900}]
                prompt = 'pending ' * 1500
                self.assertEqual(llm.get_model_context_limit(), 212992)
                self.assertGreaterEqual(212992 - agent._count_total_tokens(chat), 8000)
                summaries = []
                async def summarize(current, instruction, session):
                    summaries.append(instruction)
                    current._messages = [{'role': 'user', 'content': 'task'}]
                    current.reset_response_chain()
                    return 'continue safely', []
                agent._summarize = summarize
                meta = {}
                agent._install_request_input_budget_guard(meta)
                response = asyncio.run(agent._query_llm(chat, prompt, 'task', object()))
                self.assertEqual(response.content, 'valid continuation')
                self.assertEqual(summaries, ['task'])
                return {'enabled': enabled, 'http_calls': len(server.inputs),
                    'wire_input_tokens': list(server.inputs), 'oversized_http_calls': server.overflows,
                    'local_budget': meta.get('request_input_budget_guard')}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    def test_disabled_guard_allows_oversized_http_requests(self):
        report = self.run_case(False)
        self.assertGreaterEqual(report['oversized_http_calls'], 1)
        self.assertTrue(any(n + 32768 > 262144 for n in report['wire_input_tokens']))
        print(json.dumps(report), flush=True)
    def test_enabled_guard_uses_context_fallback_without_oversized_http(self):
        report = self.run_case(True)
        self.assertEqual(report['http_calls'], 2)
        self.assertEqual(report['oversized_http_calls'], 0)
        self.assertTrue(all(n + 32768 <= 262144 for n in report['wire_input_tokens']))
        self.assertEqual(len(report['local_budget']['blocked_calls']), 1)
        self.assertFalse(report['local_budget']['blocked_calls'][0]['sent_to_model'])
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)

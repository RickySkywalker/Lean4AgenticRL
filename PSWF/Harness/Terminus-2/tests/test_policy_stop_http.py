"""Synthetic regression tests for policy stop http."""
import ast
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic, _PolicyStop

fixtures=ast.parse((Path(__file__).parent/'test_run_orchestration.py').read_text())
selected=[n for n in fixtures.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in {'FakeSession','make_agent'}]
namespace={'Path':Path,'Terminus2Dynamic':Terminus2Dynamic}
exec(compile(ast.Module(body=selected,type_ignores=[]),'orchestration-fixtures','exec'),namespace)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length','0')))
        self.server.calls+=1
        if self.server.calls==1:
            status=403
            body={'error':{'type':self.server.error_type,'message':self.server.message}}
        else:
            status=200
            body={'id':'fixture','object':'chat.completion','created':1,'model':'Qwen/Qwen3.8-27B','choices':[{'index':0,'message':{'role':'assistant','content':'fixture ok','reasoning_content':'fixture reasoning'},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3,'completion_tokens_details':{'reasoning_tokens':1}}}
        raw=json.dumps(body).encode()
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

class PolicyHTTPTests(unittest.TestCase):
    def setUp(self):
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.calls=0
        self.server.error_type='guard_policy_violation'
        self.server.message='model command batch refused before execution (scan)'
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.llm=LiteLLM('openai/Qwen/Qwen3.8-27B',temperature=1.0,api_base=f'http://127.0.0.1:{self.server.server_port}/v1',api_key='local',timeout=3,model_info={'max_input_tokens':262144,'max_output_tokens':32768,'input_cost_per_token':0,'output_cost_per_token':0,'cache_creation_input_token_cost':0,'cache_read_input_token_cost':0,'mode':'chat'})
    def tearDown(self):
        self.server.shutdown();self.server.server_close()
    def test_real_client_denial_stops_before_retry_or_phase2(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent=namespace['make_agent'](Path(tmp),stop_on_policy_violation=True)
            agent._llm=self.llm
            agent._llm_call_kwargs={'max_tokens':16}
            context=AgentContext();phases=[];continued=[]
            async def loop(**kwargs):
                phases.append(agent._dynwf_phase)
                await Terminus2._query_llm(agent,Chat(self.llm),'offline policy fixture')
                continued.append(True)
            agent._run_agent_loop=loop
            asyncio.run(agent.run('Inspect a local file.',None,context))
            self.assertEqual(self.server.calls,1)
            self.assertEqual(phases,[1])
            self.assertEqual(continued,[])
            stop=context.metadata['dynwf']['safety_stop']
            self.assertTrue(stop['immediate']);self.assertFalse(stop['further_actions'])
            saved=json.loads((Path(tmp)/'agent/dynwf_meta.json').read_text())
            self.assertEqual(saved['safety_stop']['phase'],1)
    def ordinary_denial(self,message,error_type):
        self.server.message=message;self.server.error_type=error_type
        Terminus2Dynamic._install_policy_stop(SimpleNamespace(stop_on_policy_violation=True,_llm=self.llm))
        response=asyncio.run(self.llm.call('offline ordinary-error fixture',max_tokens=16))
        self.assertEqual(self.server.calls,2)
        self.assertEqual(response.content,'fixture ok')
    def test_generic_403_keeps_ordinary_error_path(self):
        self.ordinary_denial('ordinary access denied','forbidden')
    def test_framing_refusal_keeps_ordinary_error_path(self):
        self.ordinary_denial('model command batch refused before execution (unparseable_command)','guard_policy_violation')

if __name__=='__main__':unittest.main(verbosity=2)

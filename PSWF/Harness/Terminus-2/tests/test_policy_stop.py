"""Synthetic regression tests for policy stop."""
import ast
import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from harbor.models.agent.context import AgentContext
from harbor.agents.terminus_2.terminus_2 import Terminus2
from terminus2_dynwf.dynamic_agent import Terminus2Dynamic, _PolicyStop

fixtures=ast.parse((Path(__file__).parent/'test_run_orchestration.py').read_text())
selected=[n for n in fixtures.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in {'FakeSession','make_agent'}]
namespace={'Path':Path,'Terminus2Dynamic':Terminus2Dynamic}
exec(compile(ast.Module(body=selected,type_ignores=[]),'orchestration-fixtures','exec'),namespace)

class Denied(RuntimeError):
    status_code=403
    body={'error':{'type':'guard_policy_violation'}}

with tempfile.TemporaryDirectory() as tmp:
    agent=namespace['make_agent'](Path(tmp),stop_on_policy_violation=True)
    context=AgentContext(); phases=[]
    async def blocked_loop(**kwargs):
        phases.append(agent._dynwf_phase)
        agent._llm._handle_litellm_error(Denied('guard_policy_violation: tor'))
    agent._run_agent_loop=blocked_loop
    asyncio.run(agent.run('Inspect a local file.',None,context))
    assert phases==[1],phases
    assert context.metadata['dynwf']['safety_stop']['immediate'] is True
    assert context.metadata['dynwf']['safety_stop']['further_actions'] is False
    assert (Path(tmp)/'agent/dynwf_meta.json').is_file()
    assert json.loads((Path(tmp)/'agent/dynwf_meta.json').read_text())['safety_stop']['phase']==1
    for message in ['authentication failed','guard_policy_violation unparseable_command']:
        error=Denied(message);error.body={}
        try:agent._llm._handle_litellm_error(error)
        except _PolicyStop:raise AssertionError('non-policy/framing error was classified as safety stop')
        except Exception:pass
        else:raise AssertionError('ordinary error was swallowed')
    calls=[]
    class Chat:
        async def chat(self,*args,**kwargs):
            calls.append(1);raise _PolicyStop('guard_policy_violation')
    try:asyncio.run(Terminus2._query_llm(agent,Chat(),'prompt'))
    except _PolicyStop:pass
    else:raise AssertionError('stop was swallowed by query retry layer')
    assert len(calls)==1,calls
with tempfile.TemporaryDirectory() as tmp:
    agent=namespace['make_agent'](Path(tmp),stop_on_policy_violation=True)
    context=AgentContext(); phases=[]
    async def stripped_refusal(**kwargs):
        phases.append(agent._dynwf_phase)
        error=Denied('litellm.APIError: OpenAIException - model command batch refused before execution (scan)')
        error.body=None
        agent._llm._handle_litellm_error(error)
    agent._run_agent_loop=stripped_refusal
    asyncio.run(agent.run('Inspect a local file.',None,context))
    assert phases==[1]
    assert context.metadata['dynwf']['safety_stop']['immediate'] is True
    for message in ['generic access denied', 'model command batch refused before execution (unparseable_command)']:
        error=Denied(message);error.body=None
        try:agent._llm._handle_litellm_error(error)
        except _PolicyStop:raise AssertionError('generic/framing denial incorrectly stopped the trial')
        except Exception:pass
        else:raise AssertionError('ordinary error was swallowed')
print('POLICY_STOP_OFFLINE_OK: one refusal, no phase2, retry bypass, metadata preserved')

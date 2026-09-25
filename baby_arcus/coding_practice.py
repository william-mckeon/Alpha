"""ReAct episodes with immutable action intents and replayable outcomes."""
import json
from langchain_core.runnables import RunnableLambda
from baby_arcus.contracts import digest
from baby_arcus.transport import RemoteError
from baby_arcus.coding_policy import system_prompt


def run_episode(episode, tools, policy, store, max_steps=8, cancelled=lambda: False):
    if type(max_steps) is not int or not 1 <= max_steps <= 32:
        raise ValueError('Invalid episode budget')
    # A crash between intent and receipt is ambiguous. Never repeat a side effect.
    if store.read(episode):
        raise ValueError('Episode already exists; inspect receipts and start a new episode')
    messages = [{'role':'system','content':system_prompt([tools.catalog.tools['tool_search']])},
                {'role':'user','content':tools.environment.task['instruction']}]
    actor = RunnableLambda(policy)
    store.append(episode,0,{'kind':'start','task':tools.environment.task_name,
                          'catalog':tools.catalog.version, 'messages':messages.copy()})
    solved = False
    for index in range(max_steps):
        if cancelled():
            return {'episode':episode,'solved':False,'steps':index,'status':'cancelled',
                    'messages':messages,'automatically_approved':False}
        definitions = tools.context.definitions()
        decision = actor.invoke({'messages':messages.copy(), 'definitions':definitions})
        store.append(episode,index*2+1,{'kind':'intent','decision':decision,'context_hash':digest(messages)})
        call = decision.get('call')
        if cancelled():
            result = {'status':'cancelled','error':'Human interaction interrupted execution'}
        elif call is None:
            result = {'status':decision.get('status','invalid_call'),'error':'No executable tool call'}
        else:
            try:
                result = tools.execute(call)
            except (ValueError, OSError, TimeoutError, RemoteError) as exc:
                result = {'status':'tool_error','error':str(exc)[:1000]}
        store.append(episode,index*2+2,{'kind':'outcome','result':result})
        messages.append({'role':'assistant','content':json.dumps(call) if call else (decision.get('text') or 'No valid call'),
                         'train':bool(call and result.get('status') not in ('tool_error','cancelled'))})
        messages.append({'role':'tool','content':json.dumps(result,sort_keys=True)})
        solved = bool(call and call.get('name') == 'run_tests' and result.get('passed') is True)
        if solved or result.get('status') == 'cancelled' or decision.get('status') in ('context_exhausted','cancelled'):
            break
    return {'episode':episode,'solved':solved,'steps':index+1,'messages':messages,
            'automatically_approved':False}

"""Minimal local Runnable/StateGraph bridge; model and tool policies stay external."""
from typing import Any, TypedDict

class InferenceState(TypedDict):
    messages: list[dict[str, str]]
    result: Any

def invoke_messages(generate, messages):
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
    chain = RunnableLambda(generate)
    graph = StateGraph(InferenceState)
    graph.add_node('donor', lambda state: {'result':chain.invoke(state['messages'])})
    graph.add_edge(START, 'donor')
    graph.add_edge('donor', END)
    return graph.compile().invoke({'messages':messages})['result']


class ApplicationState(TypedDict):
    messages: list[Any]
    model_calls: int
    tool_calls: int
    status: str


def application_turn(chat_model, messages, check_live, max_model_calls=3, max_tool_calls=2):
    """Actual LangGraph loop; each model/tool node checks cancellation and budgets."""
    import json
    from langchain_core.messages import ToolMessage
    from langgraph.graph import StateGraph, START, END
    from arcus3.tools import dispatch
    if not 1<=max_model_calls<=4 or not 0<=max_tool_calls<=4: raise ValueError('Invalid loop budget')
    graph=StateGraph(ApplicationState)
    def model_node(state):
        check_live()
        answer=chat_model.invoke(state['messages'])
        status=('invalid_tool_call' if answer.invalid_tool_calls else 'running' if answer.tool_calls
                else 'response_truncated' if answer.response_metadata.get('truncated') else 'complete')
        return {'messages':state['messages']+[answer],'model_calls':state['model_calls']+1,'status':status}
    def tool_node(state):
        history=list(state['messages']);count=state['tool_calls']
        for call in history[-1].tool_calls:
            check_live()
            if count>=max_tool_calls: return {'messages':history,'status':'tool_budget_exhausted','tool_calls':count}
            receipt=dispatch(call['name'],call['args']);count+=1
            history.append(ToolMessage(content=json.dumps(receipt),tool_call_id=call['id'],name=call['name']))
        return {'messages':history,'tool_calls':count}
    def after_model(state):
        if state['status']!='running': return END
        if state['model_calls']>=max_model_calls:
            return 'exhausted'
        return 'tools'
    graph.add_node('model',model_node);graph.add_node('tools',tool_node)
    graph.add_node('exhausted',lambda state:{'status':'model_budget_exhausted'})
    graph.add_edge(START,'model');graph.add_conditional_edges('model',after_model)
    graph.add_conditional_edges('tools',lambda state:END if state['status']=='tool_budget_exhausted' else 'model')
    graph.add_edge('exhausted',END)
    return graph.compile().invoke({'messages':list(messages),'model_calls':0,'tool_calls':0,'status':'running'},
                                  config={'recursion_limit':16})

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

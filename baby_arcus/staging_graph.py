"""LangGraph validates and stages examples; it cannot approve them."""
from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from baby_arcus.sft_validation import validate


class State(TypedDict, total=False):
    records: list
    batch_id: str


def graph(store):
    validator = RunnableLambda(lambda rows: [validate(row) for row in rows])
    flow = StateGraph(State)
    flow.add_node('validate', lambda state: {'records':validator.invoke(state['records'])})
    flow.add_node('stage', lambda state: {'batch_id':store.stage(state['records'])})
    flow.add_edge(START,'validate'); flow.add_edge('validate','stage'); flow.add_edge('stage',END)
    return flow.compile()

"""
subgraph_graph.py — demonstrates composing a compiled graph as a node.

A compiled StateGraph can be added directly as a node in another graph via
add_node(name, compiled_app). When the parent and child share overlapping
state keys, LangGraph passes the parent's state in as the child's input and
merges the child's output back into the parent's state — no manual
translation needed.

This wraps router_graph's classify/respond graph (its own compiled app) as
a single "qa" node inside a parent graph that logs before and after it run.

Run with: ./venv/bin/python subgraph_graph.py
"""

from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from router_graph import classify, respond

load_dotenv()


class QAState(TypedDict):
    question: str
    provider: str
    answer: str


# The child graph: identical to router_graph's graph, compiled on its own.
child_graph = StateGraph(QAState)
child_graph.add_node("classify", classify)
child_graph.add_node("respond", respond)
child_graph.set_entry_point("classify")
child_graph.add_edge("classify", "respond")
child_graph.add_edge("respond", END)
qa_app = child_graph.compile()


class ParentState(TypedDict):
    question: str
    provider: str
    answer: str
    log: list[str]


def before(state: ParentState) -> ParentState:
    return {"log": state["log"] + [f"received question: {state['question']}"]}


def after(state: ParentState) -> ParentState:
    return {"log": state["log"] + [f"got answer via provider={state['provider']}"]}


graph = StateGraph(ParentState)
graph.add_node("before", before)
# The compiled child graph is added directly as a node. Its QAState fields
# (question, provider, answer) overlap with ParentState, so the parent's
# values flow in as input and the child's output flows back out.
graph.add_node("qa", qa_app)
graph.add_node("after", after)
graph.set_entry_point("before")
graph.add_edge("before", "qa")
graph.add_edge("qa", "after")
graph.add_edge("after", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke(
        {"question": "Write a Python function to check if a number is prime.",
         "provider": "", "answer": "", "log": []}
    )
    for line in result["log"]:
        print(f"  [log] {line}")
    print(f"\nanswer: {result['answer']}")

"""
fan_out_fan_in_graph.py — demonstrates parallel branches merged by a reducer.

Adding multiple edges from one node fans out to several nodes that run
concurrently (LangGraph executes nodes with no dependency between them in
the same "superstep"). Each branch writes to the same state key; since
plain fields are overwritten on write, concurrent writes to one would
conflict. Annotated[list, operator.add] instead tells LangGraph to combine
concurrent writes with that reducer, so fan-in is just a normal node that
reads the merged list once every branch has completed.

Run with: ./venv/bin/python fan_out_fan_in_graph.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END


class State(TypedDict):
    topic: str
    # Reducer-backed: every branch's write is appended rather than
    # overwriting the others, so all three results survive the fan-in.
    findings: Annotated[list[str], operator.add]
    summary: str


def start(state: State) -> State:
    return {}


def research_history(state: State) -> State:
    return {"findings": [f"[history] background on {state['topic']}"]}


def research_stats(state: State) -> State:
    return {"findings": [f"[stats] numbers related to {state['topic']}"]}


def research_opinions(state: State) -> State:
    return {"findings": [f"[opinions] takes on {state['topic']}"]}


def summarize(state: State) -> State:
    # Runs only after all three branches have completed and merged.
    return {"summary": " | ".join(sorted(state["findings"]))}


graph = StateGraph(State)
graph.add_node("start", start)
graph.add_node("research_history", research_history)
graph.add_node("research_stats", research_stats)
graph.add_node("research_opinions", research_opinions)
graph.add_node("summarize", summarize)

graph.set_entry_point("start")
# Fan-out: three edges from "start" run all three branches concurrently.
graph.add_edge("start", "research_history")
graph.add_edge("start", "research_stats")
graph.add_edge("start", "research_opinions")
# Fan-in: "summarize" waits for all three branches before running.
graph.add_edge("research_history", "summarize")
graph.add_edge("research_stats", "summarize")
graph.add_edge("research_opinions", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"topic": "electric vehicles", "findings": [], "summary": ""})
    print("findings:")
    for f in result["findings"]:
        print(f"  {f}")
    print(f"\nsummary: {result['summary']}")

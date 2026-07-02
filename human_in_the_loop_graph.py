"""
human_in_the_loop_graph.py — demonstrates pausing a graph for human approval.

Calling interrupt() inside a node halts execution and surfaces a payload to
the caller; app.invoke() returns immediately with that payload instead of
running to completion. Resuming happens by invoking again on the same
thread_id with Command(resume=<value>) — the interrupted node re-runs and
interrupt() returns that value instead of pausing again.

A checkpointer is required: without persisted state, there'd be nothing to
resume from.

Run with: ./venv/bin/python human_in_the_loop_graph.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt


class State(TypedDict):
    request: str
    approved: bool
    result: str


def draft(state: State) -> State:
    return {"result": f"Draft action: {state['request']}"}


def request_approval(state: State) -> State:
    # Pauses the graph here. The dict passed to interrupt() is surfaced to
    # the caller as the __interrupt__ payload; whatever value is later
    # passed via Command(resume=...) becomes this call's return value.
    decision = interrupt({"question": f"Approve? -> {state['result']}"})
    return {"approved": decision}


def apply_or_reject(state: State) -> str:
    return "apply" if state["approved"] else "reject"


def apply(state: State) -> State:
    return {"result": state["result"] + " [APPLIED]"}


def reject(state: State) -> State:
    return {"result": state["result"] + " [REJECTED]"}


graph = StateGraph(State)
graph.add_node("draft", draft)
graph.add_node("request_approval", request_approval)
graph.add_node("apply", apply)
graph.add_node("reject", reject)
graph.set_entry_point("draft")
graph.add_edge("draft", "request_approval")
graph.add_conditional_edges(
    "request_approval", apply_or_reject, {"apply": "apply", "reject": "reject"}
)
graph.add_edge("apply", END)
graph.add_edge("reject", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


def run(thread_id: str, request: str, approve: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n=== thread_id={thread_id!r} request={request!r} ===")

    result = app.invoke({"request": request, "approved": False, "result": ""}, config)
    interrupts = result.get("__interrupt__")
    if interrupts:
        print(f"  paused: {interrupts[0].value}")
    else:
        print(f"  finished without pausing: {result}")
        return

    resumed = app.invoke(Command(resume=approve), config)
    print(f"  resumed with approve={approve}: {resumed['result']}")


if __name__ == "__main__":
    run("thread-1", "delete staging database", approve=False)
    run("thread-2", "deploy hotfix", approve=True)

"""
persistence_graph.py — demonstrates checkpointing state across invocations.

A MemorySaver checkpointer persists graph state keyed by "thread_id". Each
app.invoke() call with the same thread_id resumes from where that thread
left off, instead of starting from the input state fresh. Different
thread_ids are fully isolated from each other.

Run with: ./venv/bin/python persistence_graph.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END


class State(TypedDict):
    count: int
    history: list[str]


def increment(state: State) -> State:
    new_count = state["count"] + 1
    return {"count": new_count, "history": state["history"] + [f"tick {new_count}"]}


graph = StateGraph(State)
graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_edge("increment", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


def run_thread(thread_id: str, invocations: int) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n=== thread_id={thread_id!r} ===")
    for i in range(invocations):
        # Fields without a reducer are overwritten by whatever's passed in,
        # so only the first call (no checkpoint yet) seeds count/history.
        # Later calls pass {} so the persisted state carries forward
        # untouched instead of being reset to the input.
        input_state = {"count": 0, "history": []} if i == 0 else {}
        result = app.invoke(input_state, config)
        print(f"  count={result['count']} history={result['history']}")


if __name__ == "__main__":
    # Two independent threads, interleaved, to show isolation.
    run_thread("thread-a", invocations=3)
    run_thread("thread-b", invocations=2)

    # Inspecting a thread's persisted state without invoking the graph.
    snapshot = app.get_state({"configurable": {"thread_id": "thread-a"}})
    print(f"\nthread-a final snapshot: {snapshot.values}")

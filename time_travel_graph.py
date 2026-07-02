"""
time_travel_graph.py — demonstrates rewinding and replaying from checkpoints.

Every superstep of a checkpointed graph is saved as its own checkpoint.
app.get_state_history() walks a thread's checkpoints newest-first. Passing
a config with a specific "checkpoint_id" to invoke() resumes execution from
that exact point rather than the thread's latest state — this is a fork:
new checkpoints from that call branch off from the chosen point, leaving
the original history after it untouched.

Run with: ./venv/bin/python time_travel_graph.py
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


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "time-travel-demo"}}

    # Run the graph forward three times, building up checkpoints.
    for _ in range(3):
        app.invoke({"count": 0, "history": []} if _ == 0 else {}, config)
    latest = app.get_state(config)
    print(f"latest state: {latest.values}")

    # Walk the checkpoint history, newest first.
    print("\ncheckpoint history:")
    checkpoints = list(app.get_state_history(config))
    for snap in checkpoints:
        count = snap.values.get("count", "<empty>")
        print(f"  count={count} checkpoint_id={snap.config['configurable']['checkpoint_id'][:8]}...")

    # Pick the checkpoint right after the first increment (count == 1) and
    # fork from there: resuming with {} replays "increment" once more on
    # top of that earlier state instead of the latest one.
    fork_point = next(s for s in checkpoints if s.values.get("count") == 1)
    fork_config = fork_point.config
    forked = app.invoke({}, fork_config)
    print(f"\nforked from count=1, after one more increment: {forked}")

    # get_state(config) without a checkpoint_id returns the most recently
    # written checkpoint on the thread. The fork just wrote a newer one
    # (count=2, forked off count=1), so it now shadows the original
    # count=3 checkpoint as "latest" — even though its count is lower.
    # The original checkpoints (count=2, count=3 from the first run) still
    # exist in history untouched; only what "latest" resolves to changed.
    print(f"thread 'latest' after fork: {app.get_state(config).values}")

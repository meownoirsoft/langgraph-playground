"""
streaming_graph.py — demonstrates LangGraph's app.stream() API.

Two stream modes are shown:
  "updates" — yields {node_name: state_delta} for each node as it completes.
              Only the keys that node actually changed are included.
  "values"  — yields the full accumulated state after each node completes.

Run with: ./venv/bin/python streaming_graph.py
"""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from router_graph import State, classify, respond

load_dotenv()

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.set_entry_point("classify")
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)

app = graph.compile()


def stream_updates(question: str) -> None:
    """
    stream_mode='updates' — one chunk per node, containing only the keys
    that node wrote to state. Useful for monitoring individual node outputs.
    """
    print("=== stream_mode='updates' ===")
    for chunk in app.stream(
        {"question": question, "provider": "", "answer": ""},
        stream_mode="updates",
    ):
        # chunk is {node_name: {key: new_value, ...}}
        for node_name, delta in chunk.items():
            keys = ", ".join(f"{k}={repr(v)[:60]}" for k, v in delta.items())
            print(f"  [{node_name}] {keys}")


def stream_values(question: str) -> None:
    """
    stream_mode='values' — yields the full accumulated state at every step.
    The first snapshot is the initial state (before any node runs), followed
    by one snapshot per node. For a 2-node graph: [initial, after-classify,
    after-respond]. Useful when downstream code needs the complete picture
    at every step rather than just the delta.
    """
    print("=== stream_mode='values' ===")
    for i, state in enumerate(
        app.stream(
            {"question": question, "provider": "", "answer": ""},
            stream_mode="values",
        )
    ):
        # Truncate answer for display
        display = {
            k: (v[:80] + "…" if isinstance(v, str) and len(v) > 80 else v)
            for k, v in state.items()
        }
        print(f"  step {i}: {display}")


if __name__ == "__main__":
    questions = [
        "Write a Python function to reverse a string.",
        "What is the capital of France?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        stream_updates(q)
        stream_values(q)

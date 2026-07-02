from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from router_graph import classify, respond

load_dotenv()

# Amount added to the running total on each loop, and the total to reach
# before handing off to the LLM router.
STEP = 1
MAX_COUNT = 3


class State(TypedDict):
    count: int
    question: str
    provider: str
    answer: str


def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


def add_step(state: State) -> State:
    return {"count": add(state["count"], STEP)}


def should_continue(state: State) -> str:
    return "classify" if state["count"] >= MAX_COUNT else "add_step"


graph = StateGraph(State)
graph.add_node("add_step", add_step)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.set_entry_point("add_step")
graph.add_conditional_edges("add_step", should_continue)
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "count": 0,
        "question": "Write a Python function to add two numbers.",
        "provider": "",
        "answer": "",
    })
    print(f"Loops completed: {result['count']}")
    print(result["answer"])

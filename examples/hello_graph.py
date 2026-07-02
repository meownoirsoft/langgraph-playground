from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from router_graph import classify, respond

load_dotenv()

# Number of increments before handing off to the LLM router.
MAX_COUNT = 3


class State(TypedDict):
    count: int
    question: str
    provider: str
    answer: str


def increment(state: State) -> State:
    return {"count": state["count"] + 1}


def should_continue(state: State) -> str:
    return "classify" if state["count"] >= MAX_COUNT else "increment"


graph = StateGraph(State)
graph.add_node("increment", increment)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.set_entry_point("increment")
graph.add_conditional_edges("increment", should_continue)
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

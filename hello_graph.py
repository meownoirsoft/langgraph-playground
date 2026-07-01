from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END

load_dotenv()


class State(TypedDict):
    count: int


def increment(state: State) -> State:
    return {"count": state["count"] + 1}


def should_continue(state: State) -> str:
    return END if state["count"] >= 3 else "increment"


graph = StateGraph(State)
graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_conditional_edges("increment", should_continue)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"count": 0})
    print(result)

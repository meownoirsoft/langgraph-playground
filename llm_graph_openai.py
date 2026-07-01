from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    question: str
    answer: str


def ask(state: State) -> State:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


graph = StateGraph(State)
graph.add_node("ask", ask)
graph.set_entry_point("ask")
graph.add_edge("ask", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"question": "Say hello in exactly 5 words."})
    print(result["answer"])

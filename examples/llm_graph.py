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

if __name__ == "__main__":
    questions = [
        "Say hello in exactly 5 words.",
        "Write a Python function to add two numbers.",
    ]
    for q in questions:
        result = app.invoke({"question": q, "provider": "", "answer": ""})
        print(f"Q: {q}\n{result['answer']}\n")

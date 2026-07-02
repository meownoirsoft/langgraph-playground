from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Registry mapping a provider name to its chat model.
PROVIDERS = {
    "openai": ChatOpenAI(model="gpt-4o-mini"),
    "anthropic": ChatAnthropic(model="claude-haiku-4-5-20251001"),
}

# Provider used for classification and as the fallback when routing is unclear.
DEFAULT_PROVIDER = "openai"

# Maps a classification label to the provider that should answer it.
LABEL_TO_PROVIDER = {
    "code": "anthropic",
    "general": "openai",
}

CLASSIFY_PROMPT = (
    "Classify the following question as either 'code' (programming, debugging, "
    "technical implementation) or 'general' (everything else). "
    "Reply with exactly one word: code or general.\n\nQuestion: {question}"
)


class State(TypedDict):
    question: str
    provider: str
    answer: str


def classify(state: State) -> State:
    response = PROVIDERS[DEFAULT_PROVIDER].invoke(
        CLASSIFY_PROMPT.format(question=state["question"])
    )
    label = response.content.strip().lower()
    provider = next(
        (p for keyword, p in LABEL_TO_PROVIDER.items() if keyword in label),
        DEFAULT_PROVIDER,
    )
    return {"provider": provider}


def respond(state: State) -> State:
    provider = state.get("provider", DEFAULT_PROVIDER)
    model = PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])
    response = model.invoke(state["question"])
    return {"answer": f"[{provider}] {response.content}"}


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

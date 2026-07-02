"""
tool_agent_graph.py — demonstrates a ReAct-style tool-calling agent loop.

The "agent" node calls an LLM bound with tools. If the LLM's response
includes tool calls, tools_condition routes to the "tools" node (a
ToolNode), which executes them and appends ToolMessages to the message
history. Control returns to "agent" so the LLM can read the tool results
and either call another tool or produce a final answer — looping until no
more tool calls are made.

Run with: ./venv/bin/python tool_agent_graph.py
"""

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

load_dotenv()


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


TOOLS = [add, multiply, word_count]


class State(TypedDict):
    # add_messages appends new messages instead of overwriting the list,
    # which is what lets tool call/result pairs accumulate across the loop.
    messages: Annotated[list[AnyMessage], add_messages]


model = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)


def agent(state: State) -> State:
    return {"messages": [model.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(TOOLS))
graph.set_entry_point("agent")
# tools_condition inspects the last message: routes to "tools" if it has
# tool calls, otherwise to END.
graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()


def ask(question: str) -> None:
    print(f"\nQ: {question}")
    result = app.invoke({"messages": [("user", question)]})
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    ask("What is (12 + 8) multiplied by 3?")
    ask("How many words are in the sentence 'the quick brown fox jumps'?")

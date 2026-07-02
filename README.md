# langgraph-playground

A small LangGraph project that routes questions to different LLM providers based on their content.

## How it works

1. **Classify** – the default provider (`gpt-4o-mini`) reads the question and returns a single-word label (`code` or `general`).
2. **Respond** – the label is mapped to a provider via `LABEL_TO_PROVIDER`. The chosen model answers the question.
3. The final answer is tagged with the provider name, e.g. `[claude] ...` or `[gpt] ...`.

```
classify → respond → END
```

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Secrets are managed with [phase.dev](https://phase.dev). Export `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` before running, or add them to your Phase environment.

## Running

```bash
./venv/bin/python router_graph.py
```

## Testing

```bash
./venv/bin/pytest test_router_graph.py -v
```

All tests stub out the `PROVIDERS` registry so no real API calls are made.

## Adding a new provider

### 1. Install the LangChain integration

Add the package to `requirements.txt` and install it, e.g.:

```
langchain-google-genai
```

### 2. Register the model in `PROVIDERS`

Open `router_graph.py` and add an entry to the `PROVIDERS` dict:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

PROVIDERS = {
    "claude": ChatAnthropic(model="claude-haiku-4-5-20251001"),
    "gpt":    ChatOpenAI(model="gpt-4o-mini"),
    "gemini": ChatGoogleGenerativeAI(model="gemini-2.0-flash"),  # new
}
```

### 3. Map a classification label to it (optional)

If you want the classifier to route certain questions to the new provider, add an entry to `LABEL_TO_PROVIDER`:

```python
LABEL_TO_PROVIDER = {
    "code":    "claude",
    "general": "gpt",
    "math":    "gemini",   # new label → new provider
}
```

You will also need to update `CLASSIFY_PROMPT` to include the new label in its instructions.

### 4. Update the `DEFAULT_PROVIDER` (optional)

`DEFAULT_PROVIDER` is used both for classification and as the fallback when a label is unrecognised. Change it if needed:

```python
DEFAULT_PROVIDER = "gemini"
```

No graph wiring changes are required — `respond()` looks up the provider from the registry at runtime.

## Examples

Beyond the router graph above, this repo has standalone scripts covering
core LangGraph concepts. Each is runnable on its own with
`./venv/bin/python <file>.py`.

| Script | Concept | What it shows |
|---|---|---|
| `hello_graph.py` | Loops & conditional edges | A counter node loops on itself via `add_conditional_edges` until a threshold, then hands off to the router graph. |
| `streaming_graph.py` | Streaming | `app.stream()` in `"updates"` mode (per-node deltas) vs `"values"` mode (full state snapshot after each step). |
| `persistence_graph.py` | Checkpointing | A `MemorySaver` checkpointer keyed by `thread_id` lets state (a counter + history) accumulate across separate `invoke()` calls. Different threads stay isolated. |
| `human_in_the_loop_graph.py` | Human-in-the-loop | `interrupt()` pauses a node mid-graph and surfaces a payload to the caller; `Command(resume=...)` on a later `invoke()` resumes that node with the supplied value and the graph continues (approve/reject branching). |
| `tool_agent_graph.py` | Tool-calling agent (ReAct) | An `agent` node calls an LLM bound to tools; `tools_condition` routes to a `ToolNode` when the response has tool calls, looping back to `agent` until the model returns a final answer. |
| `subgraph_graph.py` | Subgraphs | A compiled graph (the router graph) is added as a single node inside a parent graph — state flows in/out automatically via overlapping keys. |
| `fan_out_fan_in_graph.py` | Parallel fan-out/fan-in | Multiple edges from one node run several nodes concurrently; a `Annotated[list, operator.add]` reducer merges their writes so a downstream node can read all results at once. |
| `time_travel_graph.py` | Time travel | `get_state_history()` walks a thread's checkpoints; invoking with an earlier checkpoint's config forks a new branch from that point without disturbing the original history. |

## Project structure

```
router_graph.py              # graph definition: classify + respond nodes
hello_graph.py                # loop + conditional-edge example
streaming_graph.py            # app.stream() updates/values example
persistence_graph.py          # checkpointing / thread_id example
human_in_the_loop_graph.py    # interrupt() / Command(resume=...) example
tool_agent_graph.py           # ToolNode / tools_condition ReAct agent
subgraph_graph.py             # compiled graph as a node example
fan_out_fan_in_graph.py       # concurrent branches + reducer example
time_travel_graph.py          # checkpoint history / forking example
test_router_graph.py          # pytest suite (no live API calls)
requirements.txt              # Python dependencies
```

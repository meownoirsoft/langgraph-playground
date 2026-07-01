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

## Project structure

```
router_graph.py        # graph definition: classify + respond nodes
test_router_graph.py   # pytest suite (no live API calls)
requirements.txt       # Python dependencies
```

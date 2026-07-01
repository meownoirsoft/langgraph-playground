"""
Tests for router_graph.py
Run with: ./venv/bin/pytest test_router_graph.py -v
All tests stub out the PROVIDERS registry so no real API calls are made.
"""

import types
from unittest.mock import MagicMock, patch
import pytest

import router_graph
import llm_graph
import hello_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def msg(content: str) -> types.SimpleNamespace:
    """Lightweight stand-in for a LangChain AIMessage — only .content is needed."""
    return types.SimpleNamespace(content=content)


def make_model(content: str) -> MagicMock:
    """Return a fake LangChain chat model whose .invoke() returns a message."""
    model = MagicMock()
    model.invoke.return_value = msg(content)
    return model


def fake_providers(classify_response: str, answer_responses: dict[str, str]) -> dict:
    """
    Build a PROVIDERS dict where the DEFAULT_PROVIDER returns classify_response
    for classification invocations, and each provider returns its mapped answer.
    """
    providers = {}
    for name, answer in answer_responses.items():
        providers[name] = make_model(answer)
    # Give the default provider a secondary classify-response first so the
    # classification call consumes it, then further calls return the answer.
    default = router_graph.DEFAULT_PROVIDER
    providers[default].invoke.side_effect = [
        msg(classify_response),
        msg(answer_responses.get(default, "default answer")),
    ]
    return providers


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------

class TestClassify:
    def test_routes_code_to_claude(self):
        fake = fake_providers("code", {"claude": "...", "gpt": "..."})
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.classify({"question": "Write a sort function.", "provider": "", "answer": ""})
        assert result["provider"] == "claude"

    def test_routes_general_to_gpt(self):
        fake = fake_providers("general", {"claude": "...", "gpt": "..."})
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.classify({"question": "Best pizza topping?", "provider": "", "answer": ""})
        assert result["provider"] == "gpt"

    def test_falls_back_to_default_on_unknown_label(self):
        fake = fake_providers("mathematics", {"claude": "...", "gpt": "..."})
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.classify({"question": "What is 2+2?", "provider": "", "answer": ""})
        assert result["provider"] == router_graph.DEFAULT_PROVIDER

    def test_label_matching_is_case_insensitive(self):
        """Classifier output is lowercased before matching."""
        fake = fake_providers("CODE", {"claude": "...", "gpt": "..."})
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.classify({"question": "Sort algorithm?", "provider": "", "answer": ""})
        # "code" is a substring of "code" after .lower()
        assert result["provider"] == "claude"

    def test_label_with_extra_whitespace(self):
        fake = fake_providers("  code  ", {"claude": "...", "gpt": "..."})
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.classify({"question": "Reverse a string.", "provider": "", "answer": ""})
        assert result["provider"] == "claude"

    def test_uses_default_provider_for_classification(self):
        """Ensures classification always uses DEFAULT_PROVIDER, not the routed one."""
        fake = {
            "claude": make_model("irrelevant"),
            "gpt": make_model("general"),
        }
        # Re-patch side_effect for the default classifier
        fake[router_graph.DEFAULT_PROVIDER].invoke.side_effect = [
            msg("general"), msg("gpt answer")
        ]

        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            router_graph.classify({"question": "Tell me a joke.", "provider": "", "answer": ""})

        fake["claude"].invoke.assert_not_called()


# ---------------------------------------------------------------------------
# respond()
# ---------------------------------------------------------------------------

class TestRespond:
    def test_uses_routed_provider(self):
        fake = {"claude": make_model("Here is some code."), "gpt": make_model("gpt answer")}
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.respond({"question": "Write a loop.", "provider": "claude", "answer": ""})
        assert "[claude]" in result["answer"]
        assert "Here is some code." in result["answer"]
        fake["gpt"].invoke.assert_not_called()

    def test_answer_tagged_with_provider_name(self):
        fake = {"gpt": make_model("Banana bread recipe...")}
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.respond({"question": "Banana bread?", "provider": "gpt", "answer": ""})
        assert result["answer"].startswith("[gpt]")

    def test_falls_back_to_default_when_provider_missing(self):
        default_model = make_model("fallback answer")
        fake = {router_graph.DEFAULT_PROVIDER: default_model}
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            result = router_graph.respond({"question": "?", "provider": "unknown", "answer": ""})
        assert "fallback answer" in result["answer"]
        default_model.invoke.assert_called_once()

    def test_model_receives_the_question(self):
        fake = {"gpt": make_model("answer")}
        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            router_graph.respond({"question": "What is gravity?", "provider": "gpt", "answer": ""})
        fake["gpt"].invoke.assert_called_once_with("What is gravity?")


# ---------------------------------------------------------------------------
# Full graph (end-to-end with stubbed providers)
# ---------------------------------------------------------------------------

class TestGraphEndToEnd:
    def _run(self, classify_label: str, provider: str, answer_text: str, question: str) -> dict:
        fake = {
            "claude": make_model("claude fallback"),
            "gpt": make_model("gpt fallback"),
        }
        fake[router_graph.DEFAULT_PROVIDER].invoke.side_effect = [
            msg(classify_label), msg(answer_text)
        ]
        # If routing lands on non-default provider, wire its answer too
        if provider != router_graph.DEFAULT_PROVIDER:
            fake[provider] = make_model(answer_text)

        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            return router_graph.app.invoke({"question": question, "provider": "", "answer": ""})

    def test_code_question_answered_by_claude(self):
        result = self._run("code", "claude", "def reverse(lst): ...", "Reverse a list.")
        assert "[claude]" in result["answer"]

    def test_general_question_answered_by_gpt(self):
        result = self._run("general", "gpt", "Preheat oven to 350°F", "Banana bread recipe?")
        assert "[gpt]" in result["answer"]
        assert "Preheat oven" in result["answer"]

    def test_state_contains_expected_keys(self):
        result = self._run("general", "gpt", "some answer", "Any question?")
        assert "question" in result
        assert "provider" in result
        assert "answer" in result

    def test_provider_persisted_in_final_state(self):
        result = self._run("code", "claude", "code answer", "Write a class.")
        assert result["provider"] == "claude"


# ---------------------------------------------------------------------------
# llm_graph integration (reuses router_graph classify + respond via import)
# ---------------------------------------------------------------------------

class TestLlmGraph:
    """Verifies that llm_graph.app delegates routing to the shared nodes."""

    def _run(self, classify_label: str, provider: str, answer_text: str, question: str) -> dict:
        fake = {
            "claude": make_model("claude fallback"),
            "gpt":    make_model("gpt fallback"),
        }
        fake[router_graph.DEFAULT_PROVIDER].invoke.side_effect = [
            msg(classify_label), msg(answer_text)
        ]
        if provider != router_graph.DEFAULT_PROVIDER:
            fake[provider] = make_model(answer_text)

        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            return llm_graph.app.invoke({"question": question, "provider": "", "answer": ""})

    def test_uses_router_graph_app_is_distinct_object(self):
        """llm_graph compiles its own graph rather than re-using router_graph.app."""
        assert llm_graph.app is not router_graph.app

    def test_general_question_routed_to_gpt(self):
        result = self._run("general", "gpt", "Hello there!", "Say hello in exactly 5 words.")
        assert "[gpt]" in result["answer"]
        assert "Hello there!" in result["answer"]

    def test_code_question_routed_to_claude(self):
        result = self._run("code", "claude", "def add(a, b): return a + b", "Write a function to add two numbers.")
        assert "[claude]" in result["answer"]
        assert "def add" in result["answer"]

    def test_state_keys_present(self):
        result = self._run("general", "gpt", "any answer", "Any question?")
        assert {"question", "provider", "answer"} <= result.keys()

    def test_provider_set_in_final_state(self):
        result = self._run("code", "claude", "code answer", "Write a class.")
        assert result["provider"] == "claude"

    def test_original_question_preserved(self):
        q = "Say hello in exactly 5 words."
        result = self._run("general", "gpt", "Hi there, how are you?", q)
        assert result["question"] == q


# ---------------------------------------------------------------------------
# hello_graph — counter loop chained into router classify + respond
# ---------------------------------------------------------------------------

class TestHelloGraph:
    """Verifies the counter loop runs to MAX_COUNT then hands off to the router."""

    def _run(self, classify_label: str, provider: str, answer_text: str,
             question: str, start_count: int = 0) -> dict:
        fake = {
            "claude": make_model("claude fallback"),
            "gpt":    make_model("gpt fallback"),
        }
        fake[router_graph.DEFAULT_PROVIDER].invoke.side_effect = [
            msg(classify_label), msg(answer_text)
        ]
        if provider != router_graph.DEFAULT_PROVIDER:
            fake[provider] = make_model(answer_text)

        with patch.dict(router_graph.PROVIDERS, fake, clear=True):
            return hello_graph.app.invoke({
                "count": start_count,
                "question": question,
                "provider": "",
                "answer": "",
            })

    def test_count_reaches_max(self):
        result = self._run("general", "gpt", "any answer", "Any question?")
        assert result["count"] == hello_graph.MAX_COUNT

    def test_count_increments_from_zero(self):
        result = self._run("general", "gpt", "any answer", "Any question?", start_count=0)
        assert result["count"] == hello_graph.MAX_COUNT

    def test_loop_runs_once_when_starting_at_max(self):
        """increment is always the entry point, so count increments once even when
        starting at MAX_COUNT before should_continue routes to classify."""
        result = self._run("general", "gpt", "any answer", "Any question?",
                           start_count=hello_graph.MAX_COUNT)
        assert result["count"] == hello_graph.MAX_COUNT + 1

    def test_code_question_routed_to_claude(self):
        result = self._run("code", "claude", "def add(a, b): return a + b",
                           "Write a function to add two numbers.")
        assert "[claude]" in result["answer"]

    def test_general_question_routed_to_gpt(self):
        result = self._run("general", "gpt", "Preheat oven to 350°F", "Banana bread recipe?")
        assert "[gpt]" in result["answer"]
        assert "Preheat oven" in result["answer"]

    def test_is_distinct_compiled_graph(self):
        assert hello_graph.app is not router_graph.app
        assert hello_graph.app is not llm_graph.app

    def test_all_state_keys_present(self):
        result = self._run("general", "gpt", "some answer", "Any question?")
        assert {"count", "question", "provider", "answer"} <= result.keys()

    def test_question_preserved_through_loop(self):
        q = "Write a sort function."
        result = self._run("code", "claude", "sorted code", q)
        assert result["question"] == q

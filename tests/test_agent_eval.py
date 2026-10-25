"""Evaluation suite.

Two layers:

1. Retrieval evaluation - deterministic, runs with no API key. Asserts the
   RAG layer returns the correct source document for a set of questions and
   reports precision.
2. Tool-selection evaluation - requires OPENAI_API_KEY. Asserts the agent
   chooses the right tool for a given question. Skipped without a key so the
   suite stays runnable in CI.
"""
import os

import pytest

from app.db import init_db
from app.rag import search_policies

RETRIEVAL_CASES = [
    ("What are the standard payment terms?", "payment_terms"),
    ("How long do I have to raise a refund?", "refund_policy"),
    ("What happens when an invoice is 30 days overdue?", "escalation"),
    ("Is there a late payment charge?", "payment_terms"),
    ("Do I need approval to offer a discount?", "escalation"),
    ("Are setup fees refundable?", "refund_policy"),
]

TOOL_CASES = [
    ("Find the contact details for Kumar", "find_customer"),
    ("Which invoices are overdue?", "list_invoices"),
    ("What are our payment terms?", "lookup_policy"),
]


def setup_module(_) -> None:
    init_db(force=True)


@pytest.mark.parametrize("question,expected_source", RETRIEVAL_CASES)
def test_retrieval_returns_correct_document(question, expected_source):
    assert expected_source in search_policies(question)


def test_retrieval_precision_at_1():
    """Report precision@1: how often the top passage comes from the right doc."""
    hits = 0
    for question, expected in RETRIEVAL_CASES:
        top = search_policies(question, k=1)
        if expected in top:
            hits += 1
    precision = hits / len(RETRIEVAL_CASES)
    print(f"\nRetrieval precision@1: {precision:.0%} ({hits}/{len(RETRIEVAL_CASES)})")
    assert precision >= 0.8


needs_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - tool-selection evaluation skipped",
)


@needs_key
@pytest.mark.parametrize("question,expected_tool", TOOL_CASES)
def test_agent_selects_expected_tool(question, expected_tool):
    from app.agent import trace_tool_calls

    called = trace_tool_calls(question, thread_id=f"eval-{expected_tool}")
    assert expected_tool in called, f"expected {expected_tool}, agent called {called}"


@needs_key
def test_agent_remembers_across_turns():
    """A follow-up question should resolve against the previous turn."""
    from app.agent import ask, history

    ask("Find the contact details for Kumar", thread_id="memory-test")
    ask("Which city is he in?", thread_id="memory-test")
    assert len(history("memory-test")) >= 4

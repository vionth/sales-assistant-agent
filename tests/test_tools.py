"""Tool tests. These run without an API key - the tools are plain functions,
which is what makes the agent testable."""
from app.db import init_db
from app.rag import search_policies
from app.tools import find_customer, list_invoices, lookup_policy


def setup_module(_) -> None:
    init_db(force=True)


def test_find_customer_by_name():
    out = find_customer.invoke({"query": "Kumar"})
    assert "Raja Traders" in out
    assert "kumar@rajatraders.in" in out


def test_find_customer_no_match():
    assert "No customer found" in find_customer.invoke({"query": "Nonexistent"})


def test_list_overdue_invoices():
    out = list_invoices.invoke({"status": "overdue"})
    assert "INV-102" in out and "INV-106" in out
    assert "INV-101" not in out
    assert "TOTAL" in out


def test_list_all_invoices_has_total():
    out = list_invoices.invoke({"status": "all"})
    assert out.count("INV-") == 8
    assert "TOTAL: INR" in out


def test_policy_retrieval_finds_payment_terms():
    out = lookup_policy.invoke({"question": "What are the standard payment terms?"})
    assert "Net 30" in out
    assert "source:" in out


def test_policy_retrieval_finds_refund_window():
    assert "30 days" in search_policies("how long do I have to request a refund")

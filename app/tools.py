"""Tools the agent may call. Each is a plain function wrapped with @tool so
LangGraph can expose it to the model for tool calling."""
from typing import Literal

from langchain_core.tools import tool

from app.db import connect
from app.rag import search_policies


@tool
def find_customer(query: str) -> str:
    """Look up a customer by name or company. Returns their contact details
    and the sales rep who owns the account."""
    sql = """
        SELECT id, name, company, email, phone, city, owner
        FROM customers
        WHERE name LIKE ? OR company LIKE ?
        LIMIT 5
    """
    like = f"%{query}%"
    with connect() as conn:
        rows = conn.execute(sql, (like, like)).fetchall()
    if not rows:
        return f"No customer found matching '{query}'."
    return "\n".join(
        f"#{r['id']} {r['name']} ({r['company']}) | {r['email']} | {r['phone']} "
        f"| {r['city']} | owner: {r['owner']}"
        for r in rows
    )


@tool
def list_invoices(status: Literal["paid", "unpaid", "overdue", "all"] = "all") -> str:
    """List invoices filtered by status. Use 'overdue' for invoices past their
    due date, 'unpaid' for outstanding ones, or 'all' for everything."""
    base = """
        SELECT i.id, c.name, c.company, i.amount, i.currency, i.due_on, i.status
        FROM invoices i JOIN customers c ON c.id = i.customer_id
    """
    params: tuple = ()
    if status != "all":
        base += " WHERE i.status = ?"
        params = (status,)
    base += " ORDER BY i.due_on"

    with connect() as conn:
        rows = conn.execute(base, params).fetchall()
    if not rows:
        return f"No invoices with status '{status}'."

    total = sum(r["amount"] for r in rows)
    lines = [
        f"INV-{r['id']} | {r['company']} ({r['name']}) | "
        f"{r['currency']} {r['amount']:,.2f} | due {r['due_on']} | {r['status']}"
        for r in rows
    ]
    lines.append(f"TOTAL: INR {total:,.2f} across {len(rows)} invoice(s).")
    return "\n".join(lines)


@tool
def lookup_policy(question: str) -> str:
    """Answer a question about company policy (refunds, payment terms,
    escalation) by retrieving the relevant policy documents."""
    return search_policies(question)


ALL_TOOLS = [find_customer, list_invoices, lookup_policy]

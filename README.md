# Sales Assistant Agent

A tool-calling AI agent over a B2B sales CRM. Sales reps ask questions in
plain language — *"which invoices are overdue?"*, *"what are our payment
terms?"* — and the agent decides which tools to call, retrieves real data, and
answers from it.

Built with **Python, LangGraph, LangChain, FastAPI and SQLite**, with
retrieval-augmented generation over policy documents.

## Why this exists

Sales reps ask the same operational questions repeatedly, and the answers live
in three different places: the customer table, the invoice table, and a folder
of policy documents. This agent puts one natural-language interface over all
three.

## Architecture

```
  Client (HTTP / Flutter)
          │
          ▼
  FastAPI  ──  POST /ask   {"question": "..."}
          │
          ▼
  LangGraph agent
          │
    ┌─────┴─────┐
    │ assistant │ ◄──────────────┐   model decides: answer, or call a tool
    └─────┬─────┘                │
          │ tool_calls?          │
          ▼                      │
    ┌───────────┐                │
    │   tools   │ ───────────────┘   results appended to state, loop back
    └───────────┘
          │
  ┌───────┼────────────────┐
  ▼       ▼                ▼
find_   list_          lookup_policy
customer invoices           │
  │       │                 ▼
  └───┬───┘           RAG over data/policies/*.md
      ▼                (TF-IDF scored chunks)
   SQLite
```

The loop is the important part: the model may call a tool, read the result,
then call another before answering. `route()` in `app/agent.py` keeps cycling
back to the assistant until no further tool call is requested.

## Tools the agent can call

| Tool | Purpose | Backed by |
|---|---|---|
| `find_customer(query)` | Look up a customer by name or company | SQLite |
| `list_invoices(status)` | Filter invoices: paid / unpaid / overdue / all | SQLite |
| `lookup_policy(question)` | Answer policy questions from source documents | RAG |

## Retrieval

`app/rag.py` splits the policy markdown into paragraph chunks and scores them
against the question using term frequency weighted by inverse document
frequency. Retrieved passages are returned with their source file so the model
can cite them.

A vector store was deliberately not used: with three short documents, keyword
scoring returns the same passages without the dependency and start-up cost.
The function signature matches what a vector retriever would expose, so
swapping in embeddings later changes only `rag.py`.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

copy .env.example .env         # then add your OPENAI_API_KEY

python -m app.db               # create and seed sales.db
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Which invoices are overdue and what is the total?\"}"
```

## Tests

```bash
pytest -v
```

The tools are plain functions wrapped with `@tool`, so they are tested
directly without an API key or any model call. Only the agent loop needs a
key.

## Design decisions

- **Tools return formatted strings, not raw rows.** The model reasons better
  over readable text, and it keeps the SQL schema out of the prompt.
- **Temperature is 0.** These are factual lookups; creativity is a defect here.
- **The system prompt forbids inventing data.** Any customer, invoice number
  or amount must come from a tool result.
- **Drafted customer messages are marked as needing human approval**, matching
  the escalation policy in `data/policies/escalation.md`.
- **The model is bound lazily** in `_model()` so the package imports and the
  test suite runs without credentials.

## AI tools used to build this

Claude was used for scaffolding, code review and refactoring during
development. All generated code was read, run and tested before being
committed; the architecture decisions above are mine and I can explain each.

## Status and next steps

Working: the agent loop, all three tools, retrieval, the API and the tests.

Next: swap keyword retrieval for embeddings, add conversation memory across
turns, add an evaluation set that asserts the agent picks the right tool for a
given question, and connect the Flutter client.

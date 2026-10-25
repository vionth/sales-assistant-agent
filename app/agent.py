"""LangGraph agent: a tool-calling loop over the sales CRM.

Graph shape:

    START -> assistant -> (tools -> assistant)* -> END

The assistant node calls the model. If the model requests tools, the graph
routes to the tool node, appends the results, and loops back so the model can
reason over them. When no tool call is returned, the graph ends.

Conversation state is persisted by a checkpointer keyed on a thread id, so a
follow-up question like "and what about their other invoices?" resolves
against the earlier turns in that thread.
"""
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a sales assistant for a B2B sales CRM.

You help sales representatives answer questions about customers, invoices and
company policy.

Rules:
- Use the tools to fetch real data. Never invent a customer, an invoice
  number or an amount. If a tool returns nothing, say so plainly.
- When asked about policy, call lookup_policy and quote the retrieved text.
- Amounts are in INR. Keep answers short and factual.
- If you draft a follow-up message, keep it polite and under 120 words, and
  state clearly that it needs human approval before being sent.
"""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _model():
    """Bind the tools to the chat model. Imported lazily so the module can be
    imported (and tools tested) without an API key present."""
    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    return model.bind_tools(ALL_TOOLS)


def assistant(state: AgentState) -> dict:
    response = _model().invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [response]}


def route(state: AgentState) -> str:
    """Continue to the tool node while the model keeps requesting tools."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph(checkpointer=None):
    """Compile the graph. Pass a checkpointer to retain conversation state
    across turns; omit it for one-shot use."""
    graph = StateGraph(AgentState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=checkpointer)


AGENT = None
MEMORY = MemorySaver()


def _agent():
    global AGENT
    if AGENT is None:
        AGENT = build_graph(checkpointer=MEMORY)
    return AGENT


def ask(question: str, thread_id: str = "default") -> str:
    """Run one turn and return the final text answer.

    Turns sharing a thread_id share conversation history, so follow-up
    questions can refer back to earlier ones.
    """
    result = _agent().invoke(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def history(thread_id: str = "default") -> list[AnyMessage]:
    """Return the stored messages for a thread, oldest first."""
    state = _agent().get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state else []


def trace_tool_calls(question: str, thread_id: str = "trace") -> list[str]:
    """Run a question and report which tools the agent chose. Used by the
    evaluation suite to check tool selection."""
    result = _agent().invoke(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    names: list[str] = []
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            names.append(call["name"])
    return names

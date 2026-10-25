"""FastAPI service exposing the sales assistant agent."""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.db import DB_PATH, init_db
from app.schemas import AskRequest, AskResponse, HealthResponse
from app.tools import ALL_TOOLS

load_dotenv()

app = FastAPI(
    title="Sales Assistant Agent",
    description="A LangGraph tool-calling agent over a B2B sales CRM, with "
                "retrieval over policy documents.",
    version="0.1.0",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    policy_dir = Path(__file__).resolve().parent.parent / "data" / "policies"
    return HealthResponse(
        status="ok",
        database="ready" if DB_PATH.exists() else "missing",
        policies_loaded=len(list(policy_dir.glob("*.md"))),
        model_configured=bool(os.getenv("OPENAI_API_KEY")),
    )


@app.post("/ask", response_model=AskResponse)
def ask_agent(payload: AskRequest) -> AskResponse:
    from app.agent import ask

    try:
        answer = ask(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc

    return AskResponse(
        question=payload.question,
        answer=answer,
        tools_available=[t.name for t in ALL_TOOLS],
    )

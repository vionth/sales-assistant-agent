"""Lightweight retrieval over the policy documents.

Uses TF-IDF-style keyword scoring over paragraph chunks rather than a vector
database. The retrieval interface is deliberately the same shape a vector
store would expose, so swapping in embeddings later touches only this file.
"""
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"
STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "for", "in", "on",
    "we", "our", "that", "this", "it", "be", "as", "with", "from", "what",
    "how", "do", "does", "can", "i", "you",
}


def _tokenise(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@lru_cache(maxsize=1)
def _chunks() -> list[tuple[str, str, Counter]]:
    """Return (source, text, term_counts) for every paragraph in every doc."""
    out: list[tuple[str, str, Counter]] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        for para in [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n")]:
            if len(para) > 40:
                out.append((path.stem, para, Counter(_tokenise(para))))
    return out


def search_policies(question: str, k: int = 2) -> str:
    """Retrieve the k most relevant policy passages for a question."""
    chunks = _chunks()
    if not chunks:
        return "No policy documents are loaded."

    q_terms = _tokenise(question)
    if not q_terms:
        return "Could not interpret the policy question."

    # Inverse document frequency across chunks.
    n = len(chunks)
    idf = {
        t: math.log(n / (1 + sum(1 for _, _, c in chunks if t in c))) + 1.0
        for t in set(q_terms)
    }

    scored = []
    for source, text, counts in chunks:
        score = sum(counts.get(t, 0) * idf[t] for t in set(q_terms))
        if score > 0:
            scored.append((score, source, text))

    if not scored:
        return "No policy passage matched that question."

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join(f"[source: {src}.md]\n{text}" for _, src, text in scored[:k])

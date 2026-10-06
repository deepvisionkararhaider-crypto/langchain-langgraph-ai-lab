"""Project 05 — Customer Support Agent: retrieval + intent utilities.

Retrieval is a lightweight hybrid search over the curated knowledge base:
a keyword/tag score blended with an optional semantic (fastembed) score. It is
dependency-light and, unlike a vector DB, needs no index build — ideal for the
size of a help center.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from knowledge_base import Article, all_articles

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be",
    "can", "i", "you", "my", "your", "we", "it", "this", "that", "with", "how", "do",
    "does", "what", "when", "where", "which", "why", "please", "help", "need", "want",
}


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


# ---------------------------------------------------------------------------
# Semantic layer (optional)
# ---------------------------------------------------------------------------
_SEM = None
_SEM_TRIED = False


def _embedder():
    global _SEM, _SEM_TRIED
    if _SEM_TRIED:
        return _SEM
    _SEM_TRIED = True
    try:
        from fastembed import TextEmbedding

        _SEM = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    except Exception:
        _SEM = None
    return _SEM


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


_KB_VECTORS = None


def _kb_vectors(articles: list[Article]):
    global _KB_VECTORS
    if _KB_VECTORS is not None:
        return _KB_VECTORS
    emb = _embedder()
    if emb is None:
        _KB_VECTORS = False
        return False
    try:
        docs = [f"{a.title}. {a.category}. {' '.join(a.tags)}. {a.content}" for a in articles]
        _KB_VECTORS = [list(map(float, v)) for v in emb.embed(docs)]
    except Exception:
        _KB_VECTORS = False
    return _KB_VECTORS


# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = 3) -> list[tuple[Article, float]]:
    """Return the top-k articles with a blended relevance score in [0,1]."""
    articles = all_articles()
    q_tokens = _tokens(query)
    if not q_tokens:
        return []
    q_set = set(q_tokens)

    # keyword score (tag hits weigh more than body hits)
    scored: list[tuple[Article, float]] = []
    for a in articles:
        a_tags = set(t.lower() for t in a.tags)
        a_terms = set(_tokens(f"{a.title} {a.category} {a.content}"))
        tag_hits = len(q_set & a_tags)
        term_hits = len(q_set & a_terms)
        score = tag_hits * 1.5 + term_hits * 0.5
        # exact multi-word tag match bonus
        for t in a.tags:
            if t in query.lower():
                score += 1.0
        if score > 0:
            scored.append((a, score))

    # semantic score
    vectors = _kb_vectors(articles)
    if vectors:
        emb = _embedder()
        try:
            qv = list(map(float, next(iter(emb.query_embed([query])))))
            for i, a in enumerate(articles):
                sim = _cosine(qv, vectors[i])
                # merge into existing or seed
                found = next((idx for idx, (art, _) in enumerate(scored) if art.id == a.id), None)
                if found is not None:
                    art, kw = scored[found]
                    scored[found] = (art, kw + sim * 2.0)
                elif sim > 0.45:
                    scored.append((a, sim * 2.0))
        except Exception:
            pass

    if not scored:
        return []
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]
    mx = top[0][1] or 1.0
    return [(a, round(s / mx, 3)) for a, s in top]


def format_context(matches: list[tuple[Article, float]]) -> str:
    return "\n\n".join(
        f"[{a.id}] {a.title} (category: {a.category})\n{a.content}" for a, _ in matches
    )


INTENTS = [
    "billing", "refund", "technical_issue", "account_access", "plans_pricing",
    "shipping", "security", "integration", "how_to", "other",
]

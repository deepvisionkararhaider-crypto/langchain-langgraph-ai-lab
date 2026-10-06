"""Project 02 — shared research tools (robust real web search + extraction).

Search backends are tried in order, each with retries, and the first backend
that yields results wins — no backend is trusted to always be available:

1. ``ddgs`` / ``duckduckgo_search`` library (clean result URLs)
2. DuckDuckGo no-JS HTML endpoint (parsed locally)
3. Bing HTML endpoint (redirect URLs are resolved when the page is fetched)

Page fetching follows redirects, strips scripts/styles, and returns clean,
length-capped text that is safe to feed to an LLM. Sources are always real —
nothing is ever fabricated.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


@dataclass
class FetchedPage:
    url: str
    requested_url: str = ""
    title: str = ""
    text: str = ""
    ok: bool = True
    error: str = ""


def _clean(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    for a, b in (("&amp;", "&"), ("&quot;", '"'), ("&#x27;", "'"), ("&#39;", "'"),
                 ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " ")):
        html = html.replace(a, b)
    return re.sub(r"\s+", " ", html).strip()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _search_lib(q: str, n: int) -> list[SearchHit]:
    DDGS = None
    try:
        from ddgs import DDGS  # type: ignore
    except Exception:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except Exception:
            return []
    for attempt in range(3):
        try:
            with DDGS() as d:
                rows = list(d.text(q, max_results=n))
            hits = [
                SearchHit(
                    title=r.get("title", ""),
                    url=r.get("href") or r.get("url") or "",
                    snippet=r.get("body") or r.get("snippet") or "",
                )
                for r in rows
            ]
            hits = [h for h in hits if h.url]
            if hits:
                return hits
        except Exception:
            pass
        time.sleep(1.0 + attempt)  # backoff: 1s, 2s, 3s
    return []


def _search_ddg_html(q: str, n: int) -> list[SearchHit]:
    for attempt in range(2):
        try:
            r = requests.post(
                "https://html.duckduckgo.com/html/", data={"q": q}, headers=UA, timeout=25
            )
            if r.status_code != 200:
                time.sleep(1.0 + attempt)
                continue
            blocks = re.findall(
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S
            )
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
            hits: list[SearchHit] = []
            for i, (href, title) in enumerate(blocks[:n]):
                m = re.search(r"[?&]uddg=([^&]+)", href)
                if m:
                    from urllib.parse import unquote

                    href = unquote(m.group(1))
                hits.append(
                    SearchHit(
                        title=_clean(title),
                        url=href,
                        snippet=_clean(snippets[i]) if i < len(snippets) else "",
                    )
                )
            if hits:
                return hits
        except Exception:
            pass
        time.sleep(1.0 + attempt)
    return []


def _search_bing(q: str, n: int) -> list[SearchHit]:
    try:
        r = requests.get(
            "https://www.bing.com/search", params={"q": q, "count": n}, headers=UA, timeout=25
        )
        if r.status_code != 200:
            return []
        hits: list[SearchHit] = []
        for block in re.findall(r'<li class="b_algo".*?</li>', r.text, re.S)[: n + 2]:
            m = re.search(r'<a[^>]*href="(https?://[^"]+)"', block)
            if not m:
                continue
            title = re.search(r"<h2[^>]*>(.*?)</h2>", block, re.S)
            snip = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
            hits.append(
                SearchHit(
                    title=_clean(title.group(1)) if title else "",
                    url=m.group(1).replace("&amp;", "&"),
                    snippet=_clean(snip.group(1)) if snip else "",
                )
            )
        return hits[:n]
    except Exception:
        return []


def web_search(query: str, max_results: int = 5) -> list[SearchHit]:
    """Search the web across multiple backends. Never fabricates results."""
    query = (query or "").strip()
    if not query:
        return []
    for fn in (_search_lib, _search_ddg_html, _search_bing):
        try:
            hits = [h for h in fn(query, max_results) if h.url]
        except Exception:
            hits = []
        if hits:
            return hits[:max_results]
    return []


# ---------------------------------------------------------------------------
# Fetch + extract
# ---------------------------------------------------------------------------
def fetch_page(url: str, max_chars: int = 6000) -> FetchedPage:
    """Fetch a URL (following redirects) and return cleaned, capped text."""
    try:
        r = requests.get(url, headers=UA, timeout=25, allow_redirects=True)
        final_url = str(r.url)
        if r.status_code >= 400:
            return FetchedPage(url=final_url, requested_url=url, ok=False, error=f"HTTP {r.status_code}")
        if "pdf" in r.headers.get("content-type", ""):
            return FetchedPage(url=final_url, requested_url=url, ok=False, error="pdf not supported")
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
        return FetchedPage(
            url=final_url,
            requested_url=url,
            title=_clean(m.group(1)) if m else "",
            text=_clean(r.text)[:max_chars],
        )
    except Exception as exc:
        return FetchedPage(url=url, requested_url=url, ok=False, error=f"{type(exc).__name__}: {exc}")


def gather_evidence(query: str, max_sources: int = 4, per_page_chars: int = 2500) -> dict:
    """Search + fetch top pages → evidence bundle with resolved final URLs."""
    hits = web_search(query, max_results=max_sources + 3)
    pages: list[FetchedPage] = []
    for h in hits:
        if len(pages) >= max_sources:
            break
        page = fetch_page(h.url, max_chars=per_page_chars)
        if page.ok and len(page.text) > 200:
            page.title = page.title or h.title
            pages.append(page)
    return {"query": query, "hits": hits, "pages": pages}

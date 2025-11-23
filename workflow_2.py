"""
Script version of `workflow_2.ipynb`.

This module builds and runs the agentic RAG workflow that combines document
retrieval with Serper-powered web results, summarizes both, merges them, and
rates the final answer.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
import operator
import os
import requests
from pydantic import BaseModel, Field

from retrieval import hybrid_search


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------
llm_1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyBd1C67JVZRLJHpo6mZlx-QVYxBNLhDsb4",
)

llm_2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyBfwR0CdEOFjneKfQ7rrCqDoQnwi15U9_M",
)

llm_3 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key="AIzaSyDrWNuFUOy7ugN-u5C8sLHktc3gE9fF3Z4",
)


# ---------------------------------------------------------------------------
# Shared state definition
# ---------------------------------------------------------------------------
RatingLiteral = Literal["approved", "rejected"]


class AgenticRagState(TypedDict, total=False):
    query: str
    is_smalltalk: bool
    rag_answer: str
    google_answer: str
    r_summary: str
    g_summary: str
    r_g_summary: str
    rating: RatingLiteral


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SMALLTALK_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good evening",
    "good afternoon",
    "what's up",
    "how are you",
    "yo",
    "hola",
    "sup",
    "greetings",
}


def is_smalltalk_query(text: str | None) -> bool:
    if not text:
        return False
    normalized = " ".join(text.lower().split())
    if len(normalized) <= 2:
        return True
    return any(k in normalized for k in SMALLTALK_KEYWORDS)


def build_smalltalk_reply(original_query: str | None) -> str:
    base = "Hey! I'm ready whenever you want to dive into your documents."
    if not original_query:
        return base
    return f"Hey! You said “{original_query.strip()}”. I'm here when you want to explore your PDFs."


# ---------------------------------------------------------------------------
# Workflow node implementations
# ---------------------------------------------------------------------------
def query_enh(state: AgenticRagState) -> dict:
    incoming_query = state.get("query", "")
    smalltalk = is_smalltalk_query(incoming_query)
    if smalltalk:
        # No need to spend LLM calls; just echo the query.
        return {"query": incoming_query, "is_smalltalk": True}

    prompt = f"""You are a Query Enhancement module.
    here is the query {state['query']}
Your task is to take the query and convert it into an expanded, well-structured, information-rich search query suitable for Retrieval-Augmented Generation (RAG).

Goals:
1. Clarify ambiguous references (“it”, “that”, “this”, “here”, “there”, etc.).
2. Expand shorthand or incomplete questions into explicit, full queries.
3. Add related keywords, synonyms, and domain terms that improve retrieval.
4. Preserve the user’s intent without changing meaning.
5. Do NOT answer the question. Only rewrite and enhance it.
6. Keep the final output concise, factual, and optimized for retrieval engines.

Output Format:
- Provide a single enhanced query.
- No explanations, no bullets, no meta commentary."""
    result = llm_1.invoke(prompt)
    return {"query": result.content, "is_smalltalk": False}


def retriv(state: AgenticRagState) -> dict:
    if state.get("is_smalltalk"):
        return {
            "rag_answer": "Smalltalk query detected; skipping retrieval.",
        }

    query = state.get("query")
    if not query:
        return {"rag_answer": "No query provided for retrieval."}

    try:
        hits = hybrid_search(query, top_k=5)
    except Exception as exc:  # pragma: no cover - passthrough error message
        return {"rag_answer": f"Hybrid search failed: {exc}"}

    if not hits:
        return {"rag_answer": "No supporting documents were found for the query."}

    # Basic relevance filter: drop chunks that clearly don't mention
    # anything related to the query keywords. This helps when the index
    # contains very different document types (CVs, random PDFs, etc.).
    keywords = [
        w.strip(".,!?;:").lower()
        for w in query.split()
        if len(w.strip(".,!?;:")) > 3
    ]

    filtered_hits = []
    for hit in hits:
        source = hit.get("_source") if isinstance(hit, dict) else None
        content = ""
        if isinstance(source, dict):
            content = source.get("content") or ""
        else:
            content = str(hit)

        text = content.lower()
        if not keywords or any(k in text for k in keywords):
            filtered_hits.append(hit)

    if not filtered_hits:
        return {
            "rag_answer": (
                "No supporting documents in the vector store were clearly "
                "related to this query."
            )
        }

    formatted_chunks = []
    for idx, hit in enumerate(filtered_hits, start=1):
        source = hit.get("_source") if isinstance(hit, dict) else None
        content = ""
        meta_bits = []
        if isinstance(source, dict):
            content = source.get("content") or ""
            if source.get("content_type"):
                meta_bits.append(source["content_type"])
            if source.get("token_count") is not None:
                meta_bits.append(f"tokens={source['token_count']}")
        else:
            content = str(hit)

        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        formatted_chunks.append(f"[{idx}]{meta}\n{content}")

    rag_answer = "\n\n".join(formatted_chunks)
    return {"rag_answer": rag_answer}


def l_summa(state: AgenticRagState) -> dict:
    if state.get("is_smalltalk"):
        reply = build_smalltalk_reply(state.get("query"))
        return {"r_summary": reply}

    rag_answer = state.get("rag_answer", "")
    if not rag_answer:
        return {"r_summary": "No documents retrieved for summarization."}

    prompt = f"""You are a Summarization module.
retrieved documents : {rag_answer}
Your task is to read the retrieved documents and produce a single, compact summary that captures only the core information relevant to the user query.

Requirements:
1. Maximum length: 250 characters.
2. Preserve factual accuracy.
3. No filler, no opinions, no instructions, no disclaimers.
4. Do not mention the documents or the process.
5. Output only the summary text.
"""
    result = llm_2.invoke(prompt)
    return {"r_summary": result.content}


def google_s(state: AgenticRagState) -> dict:
    if state.get("is_smalltalk"):
        return {"google_answer": ""}

    query = state.get("query")
    if not query:
        return {"google_answer": "No query provided for Google search."}

    api_key = os.getenv("SERPER_API_KEY") or "ffa0120b601f768440f6f2bb82289fee7d239d9b"
    if not api_key:
        return {
            "google_answer": "Serper API key missing. Set SERPER_API_KEY to enable web search."
        }

    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": query},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # pragma: no cover - network failure passthrough
        return {"google_answer": f"Serper web search failed: {exc}"}

    organic = []
    if isinstance(payload, dict):
        organic = payload.get("organic", []) or []

    if not organic:
        return {"google_answer": "Serper search returned no organic results."}

    lines = []
    for item in organic[:5]:
        title = item.get("title", "Untitled result")
        snippet = item.get("snippet") or item.get("link", "")
        link = item.get("link", "")
        lines.append(f"- {title}\n  {snippet}\n  Source: {link}")

    google_answer = "Serper web search results:\n" + "\n".join(lines)
    return {"google_answer": google_answer}


def google_summa(state: AgenticRagState) -> dict:
    if state.get("is_smalltalk"):
        return {"g_summary": ""}

    google_answer = state.get("google_answer", "")
    if not google_answer:
        return {"g_summary": "No web search results available for summarization."}

    prompt = f"""
    You summarize noisy web search results from Serper.
google result - {google_answer}
Your job:
• Extract only factual, query-relevant information.
• Ignore SEO filler, unrelated text, ads, navigation elements, and promotional content.
• Remove duplicates and merge overlapping facts into a single coherent summary.
• Do not answer the question; only summarize the retrieved content.
• No meta-commentary. Output only the summary.

Length: maximum 250 characters.
Style: concise, factual, neutral.
    """
    result = llm_3.invoke(prompt)
    return {"g_summary": result.content}


def all_summa(state: AgenticRagState) -> dict:
    if state.get("is_smalltalk"):
        reply = build_smalltalk_reply(state.get("query"))
        return {"r_g_summary": reply}

    g_summary = state.get("g_summary", "")
    r_summary = state.get("r_summary", "")

    if not g_summary and not r_summary:
        return {"r_g_summary": "No summaries available to combine."}
    if not g_summary:
        return {"r_g_summary": r_summary}
    if not r_summary:
        return {"r_g_summary": g_summary}

    prompt = f"""
    You combine two inputs:
1. A Google-style summary - {g_summary}
2. A summary generated from retrieved documents. - {r_summary}

Your task:
• Identify factual overlaps between the two sources.
• Merge consistent information into one unified summary.
• Include unique details only if they logically fit the user query.
• Remove contradictions, speculation, SEO filler, and non-factual text.
• Do not answer the query; only produce a consolidated summary.
• No meta-comments about the sources.

Output:
• One coherent summary.
• Maximum length: 250 characters.
• Style: concise, factual, neutral.

    """
    result = llm_1.invoke(prompt)
    return {"r_g_summary": result.content}


class RatingModel(BaseModel):
    raty: RatingLiteral = Field(
        description="rate the summary as rejected or approved"
    )


Rater = llm_2.with_structured_output(RatingModel)


def rat(state: AgenticRagState) -> dict:
    r_g_summary = state.get("r_g_summary", "")
    if not r_g_summary:
        return {"rating": "rejected"}

    prompt = f"""rate this - {r_g_summary} approved, rejected"""
    try:
        result = Rater.invoke(prompt)
        rating = getattr(result, "raty", None)
        if rating not in ("approved", "rejected"):
            rating = "rejected"
    except Exception:  # pragma: no cover - defensive default
        rating = "rejected"

    return {"rating": rating}


def check_rating(state: AgenticRagState) -> str:
    rating = state.get("rating", "rejected")
    if rating == "approved":
        return "approved"
    return "rejected"


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------
graph = StateGraph(AgenticRagState)
graph.add_node("query_enh", query_enh)
graph.add_node("retriv", retriv)
graph.add_node("l_summa", l_summa)
graph.add_node("google_s", google_s)
graph.add_node("google_summa", google_summa)
graph.add_node("all_summa", all_summa)
graph.add_node("rat", rat)

graph.add_edge(START, "query_enh")
graph.add_edge("query_enh", "google_s")
graph.add_edge("query_enh", "retriv")
graph.add_edge("retriv", "l_summa")
graph.add_edge("google_s", "google_summa")
graph.add_edge("google_summa", "all_summa")
graph.add_edge("l_summa", "all_summa")
graph.add_edge("all_summa", "rat")
graph.add_conditional_edges("rat", check_rating, {"approved": END, "rejected": "retriv"})

workflow = graph.compile()


def run_workflow(initial_state: AgenticRagState) -> AgenticRagState:
    """
    Execute the compiled workflow with the provided initial state.
    """
    return workflow.invoke(initial_state)


if __name__ == "__main__":
    import json
    import sys

    query = " ".join(sys.argv[1:]).strip() or "tell me about one piece"
    final_state = run_workflow({"query": query})
    print(json.dumps(final_state, indent=2))


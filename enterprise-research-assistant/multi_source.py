"""
multi_source.py

Module 5 - Multi-Source Research

Combines the internet search tool, Wikipedia tool and the document (RAG)
retriever into a single research step. The three sources are fetched
concurrently using LangChain's RunnableParallel, then merged by the LLM
into one coherent, source-attributed answer.
"""

from typing import TypedDict

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config
from tools import search_tool, wiki_tool
from rag import get_retriever, has_documents

llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2)


class SourceResult(TypedDict):
    web_result: str
    wiki_result: str
    doc_result: str


def _run_web(topic: str) -> str:
    try:
        return search_tool.run(topic)
    except Exception as exc:  # noqa: BLE001 - surface network/tool errors to the UI
        return f"Web search unavailable: {exc}"


def _run_wiki(topic: str) -> str:
    try:
        return wiki_tool.run(topic)
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia lookup unavailable: {exc}"


def _run_docs(topic: str) -> str:
    if not has_documents():
        return "No documents have been uploaded to the knowledge base yet."
    try:
        retriever = get_retriever()
        docs = retriever.invoke(topic)
        if not docs:
            return "No relevant information found in the uploaded documents."
        return "\n\n".join(
            f"[{d.metadata.get('source_file', 'document')}] {d.page_content}" for d in docs
        )
    except Exception as exc:  # noqa: BLE001
        return f"Document search unavailable: {exc}"


# RunnableParallel fetches all three sources concurrently (in worker threads).
_parallel_fetch = RunnableParallel(
    web_result=RunnableLambda(_run_web),
    wiki_result=RunnableLambda(_run_wiki),
    doc_result=RunnableLambda(_run_docs),
)

_merge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research analyst. Combine the information from the "
            "web, Wikipedia and internal documents below into one clear, "
            "well-organized answer about the topic. Note which source each "
            "key fact came from. If a source has no useful information, "
            "ignore it silently rather than mentioning the omission.",
        ),
        (
            "human",
            "Topic: {topic}\n\n"
            "WEB RESULTS:\n{web_result}\n\n"
            "WIKIPEDIA RESULTS:\n{wiki_result}\n\n"
            "DOCUMENT RESULTS:\n{doc_result}",
        ),
    ]
)

_merge_chain = _merge_prompt | llm


def multi_source_research(topic: str) -> dict:
    """
    Runs web, Wikipedia and document research in parallel for a single
    topic and merges the results into one synthesized answer.

    Returns a dict with the raw per-source results plus 'merged_summary'.
    """
    sources: SourceResult = _parallel_fetch.invoke(topic)
    merged = _merge_chain.invoke({"topic": topic, **sources})

    return {
        "topic": topic,
        "web_result": sources["web_result"],
        "wiki_result": sources["wiki_result"],
        "doc_result": sources["doc_result"],
        "merged_summary": merged.content,
    }

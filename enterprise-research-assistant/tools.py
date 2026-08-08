"""
tools.py

Module 2 - Internet Research
Module 3 - Wikipedia Research
Module 4 - Document Research (RAG)
Module 10 - Python / Data-Analysis Tool

All research tools are free/no-key services (DuckDuckGo search, Wikipedia,
a local Chroma vector store, and the Python REPL), so the assistant works
out of the box with only a Groq API key configured.
"""

from typing import Dict, List

import wikipedia
import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool

from rag import get_retriever, has_documents

# ---------------------------------------------------------------------------
# Module 2 - Internet Research
# ---------------------------------------------------------------------------
search_tool = DuckDuckGoSearchRun(
    description=(
        "Search the internet for current information such as recent news, "
        "stock prices, company updates and market trends. Input should be a "
        "concise search query."
    )
)

# ---------------------------------------------------------------------------
# Module 3 - Wikipedia Research
# ---------------------------------------------------------------------------
wikipedia.set_user_agent("EnterpriseResearchAssistant/1.0")

wiki_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)

wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_api,
    description="Look up background/encyclopedic knowledge on a topic, company, technology or concept from Wikipedia.",
)

# ---------------------------------------------------------------------------
# Module 4 - Document Research (RAG)
# ---------------------------------------------------------------------------


@tool
def document_search(question: str) -> str:
    """
    Search the uploaded PDF/TXT knowledge base for information relevant to
    the question. Use this whenever the user refers to an uploaded document,
    annual report, policy or manual.
    """
    if not has_documents():
        return "No documents have been uploaded yet. Ask the user to upload a PDF or TXT file first."

    retriever = get_retriever()
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information was found in the uploaded documents."

    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown document")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# Module 10 - Python / Data-Analysis Tool
# ---------------------------------------------------------------------------
python_tool = PythonREPLTool(
    description=(
        "Execute Python code for calculations, statistics or quick data "
        "transformations. Input must be valid Python code. Use print() to "
        "see output."
    )
)


class ComparisonInput(BaseModel):
    entities: List[str] = Field(description="Names being compared, e.g. company names or years.")
    metrics: Dict[str, List[float]] = Field(
        description=(
            "Mapping of metric name to a list of numeric values, one per entity, "
            "in the same order as 'entities'. Example: {'Revenue (USD B)': [96.8, 61.9]}"
        )
    )


@tool("comparison_table", args_schema=ComparisonInput)
def comparison_table(entities: List[str], metrics: Dict[str, List[float]]) -> str:
    """
    Build a comparison table and summary statistics (min, max, mean, and
    percentage change) across a set of entities (e.g. companies or years)
    for one or more numeric metrics. Use this for requests like
    'compare revenue growth' or 'compare these companies'.
    """
    try:
        df = pd.DataFrame(metrics, index=entities)
    except ValueError as exc:
        return f"Could not build comparison table: {exc}. Every metric list must have one value per entity."

    lines = [df.to_markdown(), ""]

    for metric_name in df.columns:
        series = df[metric_name]
        lines.append(f"**{metric_name}**")
        lines.append(f"- Min: {series.min():.2f} ({series.idxmin()})")
        lines.append(f"- Max: {series.max():.2f} ({series.idxmax()})")
        lines.append(f"- Mean: {series.mean():.2f}")
        if len(series) >= 2 and series.iloc[0] != 0:
            pct_change = (series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0]) * 100
            lines.append(f"- Change (first → last): {pct_change:+.2f}%")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Combined toolkit used by the conversational agent (agent.py)
# ---------------------------------------------------------------------------
toolkit = [
    search_tool,
    wiki_tool,
    document_search,
    python_tool,
    comparison_table,
]

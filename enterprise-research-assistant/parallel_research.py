"""
parallel_research.py

Module 7 - Parallel Agent

Researches several topics (e.g. multiple companies) at the same time. Each
topic is researched fully independently (its own web/Wikipedia/document
multi-source lookup) via RunnableParallel, and the results are only combined
afterwards - matching the handbook example:
"Research Google, Microsoft, Amazon, and OpenAI."
"""

from typing import Dict, List

from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config
from multi_source import multi_source_research

llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2)

_compare_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research analyst. You are given independent research "
            "summaries for several entities. Write a short comparative "
            "overview (3-6 sentences) highlighting the most important "
            "similarities and differences between them.",
        ),
        ("human", "{research_dump}"),
    ]
)
_compare_chain = _compare_prompt | llm


def parse_topics(raw_input: str) -> List[str]:
    """Split a comma/newline separated string of topics into a clean list."""
    parts = [p.strip() for chunk in raw_input.splitlines() for p in chunk.split(",")]
    return [p for p in parts if p]


def parallel_topic_research(topics: List[str]) -> Dict[str, dict]:
    """
    Researches each topic independently and concurrently, then returns a
    dict keyed by topic with each topic's multi-source research result.
    """
    if not topics:
        raise ValueError("At least one topic is required.")

    parallel_chain = RunnableParallel(
        **{f"topic_{i}": RunnableLambda(lambda t=topic: multi_source_research(t)) for i, topic in enumerate(topics)}
    )

    raw_results = parallel_chain.invoke({})
    return {topics[i]: raw_results[f"topic_{i}"] for i in range(len(topics))}


def combine_parallel_results(results: Dict[str, dict]) -> str:
    """Produce a short comparative overview across all researched topics."""
    dump = "\n\n".join(
        f"### {topic}\n{data['merged_summary']}" for topic, data in results.items()
    )
    comparison = _compare_chain.invoke({"research_dump": dump})
    return comparison.content

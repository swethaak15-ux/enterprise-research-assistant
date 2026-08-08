"""
report_pipeline.py

Module 6 - Structured Output
Module 8 - Sequential Chain

Defines the fixed report schema (Pydantic) and the sequential pipeline:

    Research -> Summarize -> Generate Report -> Generate Executive Summary
    -> Prepare Email Draft

Each step is a RunnableLambda chained with LCEL's ``|`` operator; a single
dict ("context") is threaded through and enriched at every step so the
final result carries everything the Streamlit UI needs to display.
"""

from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config
import memory
from multi_source import multi_source_research

llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.3)


# ---------------------------------------------------------------------------
# Module 6 - Structured Output
# ---------------------------------------------------------------------------
class ResearchReport(BaseModel):
    title: str = Field(description="A clear, professional report title.")
    executive_summary: str = Field(description="A 3-5 sentence high-level summary of the findings.")
    key_findings: List[str] = Field(description="The most important factual findings, as bullet points.")
    strengths: List[str] = Field(description="Notable strengths, advantages or positive signals.")
    weaknesses: List[str] = Field(description="Notable weaknesses, risks or challenges.")
    future_opportunities: List[str] = Field(description="Forward-looking opportunities or trends to watch.")
    conclusion: str = Field(description="A concise closing assessment.")
    references: List[str] = Field(description="Sources used (e.g. 'Web search', 'Wikipedia', document names).")


structured_llm = llm.with_structured_output(ResearchReport)

_report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior research analyst producing a client-ready report. "
            "Use only the research material provided. Be specific and avoid "
            "vague filler statements.",
        ),
        (
            "human",
            "Topic: {topic}\n\nResearch summary:\n{summary}\n\n"
            "Sources consulted: web={has_web}, wikipedia={has_wiki}, documents={has_docs}",
        ),
    ]
)
_report_chain = _report_prompt | structured_llm


_exec_summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a 3-4 sentence executive summary suitable for a busy "
            "manager reading only an email, based on the full report below. "
            "No headers, no bullet points, plain prose.",
        ),
        ("human", "{report_text}"),
    ]
)
_exec_summary_chain = _exec_summary_prompt | llm


_email_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Draft a short, professional email delivering a research report. "
            "Reply with the email body only (no subject line, no markdown).",
        ),
        (
            "human",
            "Topic: {topic}\nExecutive summary: {executive_summary}\n"
            "Mention that the full structured report is attached.",
        ),
    ]
)
_email_chain = _email_prompt | llm


# ---------------------------------------------------------------------------
# Module 8 - Sequential Chain steps
# ---------------------------------------------------------------------------
def _step_research(context: dict) -> dict:
    result = multi_source_research(context["topic"])
    context.update(result)
    return context


def _step_summarize(context: dict) -> dict:
    # multi_source_research already produced 'merged_summary'; this step is kept
    # explicit (per the handbook's Research -> Summarize -> ... pipeline) so it
    # can be swapped for a different summarization strategy independently.
    context["summary"] = context["merged_summary"]
    return context


def _step_generate_report(context: dict) -> dict:
    report: ResearchReport = _report_chain.invoke(
        {
            "topic": context["topic"],
            "summary": context["summary"],
            "has_web": bool(context.get("web_result")),
            "has_wiki": bool(context.get("wiki_result")),
            "has_docs": "No documents" not in context.get("doc_result", "No documents"),
        }
    )
    context["report"] = report
    return context


def _step_generate_executive_summary(context: dict) -> dict:
    report: ResearchReport = context["report"]
    report_text = (
        f"Title: {report.title}\n"
        f"Executive Summary: {report.executive_summary}\n"
        f"Key Findings: {'; '.join(report.key_findings)}\n"
        f"Conclusion: {report.conclusion}"
    )
    exec_summary = _exec_summary_chain.invoke({"report_text": report_text})
    context["email_executive_summary"] = exec_summary.content
    return context


def _step_prepare_email_draft(context: dict) -> dict:
    body = _email_chain.invoke(
        {"topic": context["topic"], "executive_summary": context["email_executive_summary"]}
    )
    context["email_draft"] = {
        "subject": f"Research Report: {context['report'].title}",
        "body": body.content,
    }
    return context


sequential_report_pipeline = (
    RunnableLambda(_step_research)
    | RunnableLambda(_step_summarize)
    | RunnableLambda(_step_generate_report)
    | RunnableLambda(_step_generate_executive_summary)
    | RunnableLambda(_step_prepare_email_draft)
)


def run_sequential_report_pipeline(topic: str, thread_id: str = "default") -> dict:
    """
    Runs the full Research -> Summarize -> Report -> Executive Summary ->
    Email Draft pipeline for one topic, then saves the report to persistent
    client memory (Module 9).
    """
    context = sequential_report_pipeline.invoke({"topic": topic})

    memory.save_report(
        thread_id=thread_id,
        topic=topic,
        report_json=context["report"].model_dump_json(),
    )

    return context

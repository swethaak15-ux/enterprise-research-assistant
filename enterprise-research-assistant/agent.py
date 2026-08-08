"""
agent.py

Module 1 - AI Chat Assistant

Builds the conversational LangGraph agent used by the "Chat Assistant" tab.
The agent has access to the full research toolkit (internet search,
Wikipedia, document RAG, Python/data-analysis) plus Gmail tools when that
integration is configured. Short-term/persistent memory is provided by a
LangGraph SqliteSaver checkpointer keyed by thread_id.
"""

from typing import Dict, List, Tuple

from langchain_groq import ChatGroq
from langchain.agents import create_agent

import config
from memory import get_checkpointer, touch_session
from tools import toolkit
from gmail_tools import get_gmail_tools

config.require_groq_key()
config.ensure_directories()

llm = ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY)

checkpointer = get_checkpointer()

_all_tools = toolkit + get_gmail_tools()

SYSTEM_PROMPT = (
    "You are an Enterprise Research Assistant for a business consulting firm. "
    "You help consultants research industries, competitors, market trends and "
    "companies. Use the internet search and Wikipedia tools for current and "
    "background information, and the document_search tool whenever the user "
    "refers to an uploaded document or report. Use the comparison_table tool "
    "for any request to compare numeric data across companies, years or "
    "metrics. Be concise, cite where information came from (web, Wikipedia or "
    "uploaded documents), and offer to prepare a structured report or send it "
    "by email when appropriate."
)

agent = create_agent(
    model=llm,
    tools=_all_tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


def ask_agent(question: str, thread_id: str = "default") -> Tuple[str, List[Dict]]:
    """
    Sends a question to the agent and returns (answer_text, tool_calls),
    where tool_calls is a list of {"tool": name, "input": ..., "output": ...}
    used by the UI to populate the "Internet Sources" / "PDF Sources"
    expandable sections.
    """
    touch_session(thread_id)

    tool_calls: List[Dict] = []
    pending_inputs: Dict[str, str] = {}

    events = agent.stream(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    )

    answer = ""
    for event in events:
        messages = event["messages"]
        last_message = messages[-1]
        answer = getattr(last_message, "content", "") or answer

        # AIMessages with tool_calls tell us which tool is about to run.
        tool_calls_meta = getattr(last_message, "tool_calls", None)
        if tool_calls_meta:
            for call in tool_calls_meta:
                pending_inputs[call["id"]] = call.get("args", {})

        # ToolMessages carry the tool's name + output; pair with the input above.
        if getattr(last_message, "type", None) == "tool":
            tool_call_id = getattr(last_message, "tool_call_id", None)
            tool_calls.append(
                {
                    "tool": getattr(last_message, "name", "unknown"),
                    "input": pending_inputs.get(tool_call_id, {}),
                    "output": str(last_message.content)[:2000],
                }
            )

    return answer, tool_calls

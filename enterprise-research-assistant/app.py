"""
app.py

Streamlit UI for the Enterprise Research & Report Generation AI Assistant.

Two tabs:
  - Chat Assistant   : Module 1 conversational agent with all tools.
  - Report Generator : Modules 5-8 multi-source / parallel research and the
                        structured, sequential report pipeline, with TXT/PDF
                        export (Streamlit deliverable) and optional Gmail
                        delivery (Module 11).
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import config

config.ensure_directories()

st.set_page_config(page_title="Enterprise Research Assistant", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Enterprise Research Assistant")
    st.caption("AI-powered research, RAG and report generation")
    st.write("---")

    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file, then restart the app.")
        st.stop()

    import memory as memory_store

    thread_id = st.text_input("Client / Session ID", value=st.session_state.get("thread_id", "default"))
    st.session_state.thread_id = thread_id

    previous_sessions = memory_store.list_sessions()
    if previous_sessions:
        chosen = st.selectbox(
            "Previous Sessions",
            options=["(current)"] + [s for s in previous_sessions if s != thread_id],
        )
        if chosen != "(current)":
            st.session_state.thread_id = chosen
            st.rerun()

    profile = memory_store.get_profile(thread_id)
    with st.expander("Client Profile", expanded=False):
        client_name = st.text_input("Client name", value=profile.get("client_name") or "")
        report_style = st.selectbox(
            "Preferred report style",
            ["Concise", "Detailed", "Executive"],
            index=["Concise", "Detailed", "Executive"].index(profile.get("preferred_report_style") or "Detailed")
            if profile.get("preferred_report_style") in ["Concise", "Detailed", "Executive"]
            else 1,
        )
        if st.button("Save Profile"):
            memory_store.save_profile(thread_id, client_name=client_name, preferred_report_style=report_style)
            st.success("Profile saved.")

        top_industries = memory_store.top_industries(thread_id)
        if top_industries:
            st.caption("Frequently researched: " + ", ".join(top_industries))

    st.write("---")
    st.subheader("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Create / Update Knowledge Base"):
        if not uploaded_files:
            st.warning("Upload at least one PDF or TXT file first.")
        else:
            from rag import create_vector_database

            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
            with st.spinner("Ingesting documents..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(config.UPLOAD_FOLDER, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    create_vector_database(file_path)
            st.success(f"Knowledge base updated with {len(uploaded_files)} document(s).")

    from rag import list_ingested_sources

    ingested = list_ingested_sources()
    if ingested:
        st.caption("In knowledge base: " + ", ".join(ingested))

    st.write("---")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.write("---")
    st.caption(f"LLM: Groq · {config.GROQ_MODEL}")
    st.caption("Gmail: " + ("✅ configured" if config.is_gmail_configured() else "⚠️ not configured"))

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
tab_chat, tab_report = st.tabs(["💬 Chat Assistant", "📊 Report Generator"])

# ----- Chat Assistant tab ---------------------------------------------------
with tab_chat:
    from agent import ask_agent

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_calls"):
                _render_tool_calls = message["tool_calls"]
                web_calls = [t for t in _render_tool_calls if t["tool"] in ("duckduckgo_search",)]
                wiki_calls = [t for t in _render_tool_calls if "wikipedia" in t["tool"]]
                doc_calls = [t for t in _render_tool_calls if t["tool"] == "document_search"]
                other_calls = [t for t in _render_tool_calls if t not in web_calls + wiki_calls + doc_calls]

                if web_calls:
                    with st.expander("🌐 Internet Sources"):
                        for c in web_calls:
                            st.markdown(f"**Query:** {c['input']}")
                            st.text(c["output"])
                if wiki_calls:
                    with st.expander("📚 Wikipedia Sources"):
                        for c in wiki_calls:
                            st.markdown(f"**Query:** {c['input']}")
                            st.text(c["output"])
                if doc_calls:
                    with st.expander("📄 PDF / Document Sources"):
                        for c in doc_calls:
                            st.markdown(f"**Query:** {c['input']}")
                            st.text(c["output"])
                if other_calls:
                    with st.expander("🛠️ Other Tool Calls"):
                        for c in other_calls:
                            st.markdown(f"**{c['tool']}** — {c['input']}")
                            st.text(c["output"])

    prompt = st.chat_input("Ask anything - e.g. 'Research Tesla's AI strategy'")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    answer, tool_calls = ask_agent(prompt, thread_id=thread_id)
                except Exception as exc:  # noqa: BLE001
                    answer, tool_calls = f"Sorry, something went wrong: {exc}", []
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer, "tool_calls": tool_calls})
        st.rerun()

# ----- Report Generator tab -------------------------------------------------
with tab_report:
    st.subheader("Structured Report Generator")
    st.caption(
        "Runs the full pipeline: Research → Summarize → Structured Report → "
        "Executive Summary → Email Draft. Enter multiple topics (comma-separated) "
        "to research them in parallel, e.g. 'Google, Microsoft, Amazon, OpenAI'."
    )

    topics_input = st.text_input("Topic(s) to research", placeholder="e.g. Tesla's AI strategy")
    recipient_email = st.text_input("Report recipient email", value=config.DEFAULT_REPORT_RECIPIENT)
    generate_clicked = st.button("Generate Report(s)", type="primary")

    if generate_clicked and topics_input.strip():
        from parallel_research import parse_topics, parallel_topic_research, combine_parallel_results
        from report_pipeline import run_sequential_report_pipeline
        from export import export_report_txt, export_report_pdf

        topics = parse_topics(topics_input)

        if len(topics) == 1:
            with st.spinner(f"Researching '{topics[0]}'..."):
                try:
                    result = run_sequential_report_pipeline(topics[0], thread_id=thread_id)
                    st.session_state.last_report_results = {topics[0]: result}
                    st.session_state.last_comparison = None
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Report generation failed: {exc}")
                    st.session_state.last_report_results = None
        else:
            with st.spinner(f"Researching {len(topics)} topics in parallel..."):
                try:
                    parallel_raw = parallel_topic_research(topics)
                    results = {}
                    for topic in topics:
                        results[topic] = run_sequential_report_pipeline(topic, thread_id=thread_id)
                    comparison = combine_parallel_results(parallel_raw)
                    st.session_state.last_report_results = results
                    st.session_state.last_comparison = comparison
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Report generation failed: {exc}")
                    st.session_state.last_report_results = None

    results = st.session_state.get("last_report_results")

    if results:
        comparison = st.session_state.get("last_comparison")
        if comparison:
            st.markdown("### 🔍 Comparative Overview")
            st.info(comparison)

        for topic, context in results.items():
            report = context["report"]
            st.markdown(f"## 📑 {report.title}")

            with st.expander("🌐 Internet Sources"):
                st.text(context.get("web_result", "N/A"))
            with st.expander("📚 Wikipedia Sources"):
                st.text(context.get("wiki_result", "N/A"))
            with st.expander("📄 PDF / Document Sources"):
                st.text(context.get("doc_result", "N/A"))
            with st.expander("📝 Final Summary", expanded=True):
                st.write(context["summary"])

            st.markdown("#### Executive Summary")
            st.write(report.executive_summary)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ✅ Strengths")
                for s in report.strengths:
                    st.markdown(f"- {s}")
                st.markdown("#### 🔑 Key Findings")
                for k in report.key_findings:
                    st.markdown(f"- {k}")
            with col2:
                st.markdown("#### ⚠️ Weaknesses")
                for w in report.weaknesses:
                    st.markdown(f"- {w}")
                st.markdown("#### 🚀 Future Opportunities")
                for o in report.future_opportunities:
                    st.markdown(f"- {o}")

            st.markdown("#### Conclusion")
            st.write(report.conclusion)

            st.markdown("#### References")
            for r in report.references:
                st.markdown(f"- {r}")

            txt_content = export_report_txt(report)
            pdf_path = export_report_pdf(report)

            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                st.download_button(
                    "⬇️ Download TXT",
                    data=txt_content,
                    file_name=f"{topic.replace(' ', '_')}_report.txt",
                    mime="text/plain",
                    key=f"txt_{topic}",
                )
            with dl_col2:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        key=f"pdf_{topic}",
                    )
            with dl_col3:
                if st.button("📧 Send via Gmail", key=f"email_{topic}"):
                    from gmail_tools import send_email, GmailNotConfiguredError

                    try:
                        draft = context["email_draft"]
                        status = send_email(
                            to=recipient_email,
                            subject=draft["subject"],
                            body=draft["body"],
                            attachment_path=pdf_path,
                        )
                        st.success(status)
                    except GmailNotConfiguredError as exc:
                        st.warning(str(exc))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not send email: {exc}")

            with st.expander("✉️ Email Draft"):
                st.markdown(f"**Subject:** {context['email_draft']['subject']}")
                st.write(context["email_draft"]["body"])

            st.write("---")

# 🤖 Enterprise Research & Report Generation AI Assistant

An AI-powered research analyst for ABC Consulting Pvt. Ltd. It searches the
web and Wikipedia, answers questions from uploaded PDF/TXT documents (RAG),
researches multiple companies in parallel, and turns everything into a
structured, downloadable, emailable report — all from a single Streamlit
app, built entirely on **free/no-cost services** (Groq's free LLM tier,
DuckDuckGo search, Wikipedia, local HuggingFace embeddings, and a local
Chroma vector database).

Built for the Agentic AI Engineering Program capstone brief: **Project 1 —
Enterprise Research & Report Generation AI Assistant**.

---

## 1. Features

| Module | Capability |
|---|---|
| 1 | Conversational AI chat assistant (LangGraph agent) |
| 2 | Internet research via DuckDuckGo (no API key required) |
| 3 | Wikipedia research |
| 4 | PDF/TXT document RAG (Chroma + local HuggingFace embeddings) |
| 5 | Multi-source research — web + Wikipedia + documents merged into one answer |
| 6 | Structured report output (fixed Pydantic schema) |
| 7 | Parallel research across multiple topics/companies at once |
| 8 | Sequential pipeline: Research → Summarize → Report → Executive Summary → Email Draft |
| 9 | Short-term, persistent and long-term (client-profile) memory |
| 10 | Python/data-analysis tool + comparison-table tool (pandas) |
| 11 | Gmail integration to send reports |

Plus: TXT/PDF report export, expandable source panels, session history, and
graceful degradation when optional integrations aren't configured.

---

## 2. Architecture

```
User (Streamlit)
   │
   ├── Chat Assistant tab ──► agent.py ──► tools.py (search, wikipedia,
   │                                        document_search, python REPL,
   │                                        comparison_table)
   │                          + gmail_tools.py (optional)
   │                          + memory.py (LangGraph SqliteSaver)
   │
   └── Report Generator tab ─► parallel_research.py (Module 7)
                                     │
                                     ▼
                                multi_source.py (Module 5, RunnableParallel:
                                web + wikipedia + rag.py document retriever)
                                     │
                                     ▼
                          report_pipeline.py (Module 6 + 8, LCEL sequential
                          chain → structured Pydantic ResearchReport)
                                     │
                                     ▼
                          export.py (TXT/PDF)  +  gmail_tools.send_email
                                     │
                                     ▼
                          memory.py (persistent report_history + client
                          profile / frequently researched industries)
```

**LLM:** Groq (`langchain-groq`), free tier — no OpenAI key required.
**Embeddings:** local `sentence-transformers/all-MiniLM-L6-v2` — free, no key.
**Vector store:** Chroma, persisted to disk at `chroma_db/`.
**Web search:** DuckDuckGo — free, no key.

---

## 3. Folder Structure

```
enterprise-research-assistant/
├── app.py                  # Streamlit UI (entry point: streamlit run app.py)
├── main.py                 # Convenience/sanity-check entry stub
├── agent.py                 # Module 1 - conversational agent
├── tools.py                  # Modules 2, 3, 4, 10 - research & data tools
├── rag.py                     # Module 4 - PDF/TXT ingestion + Chroma retriever
├── multi_source.py             # Module 5 - parallel multi-source research
├── parallel_research.py         # Module 7 - parallel multi-topic research
├── report_pipeline.py            # Modules 6 & 8 - structured report + sequential chain
├── export.py                      # TXT/PDF export
├── memory.py                       # Module 9 - short-term/persistent/long-term memory
├── gmail_tools.py                   # Module 11 - Gmail integration
├── config.py                          # Environment & configuration management
├── .streamlit/
│   └── config.toml                     # Streamlit theme
├── sample_data/                         # Sample company documents for RAG demo
│   ├── nova_robotics_annual_report_2025.txt
│   └── nova_robotics_hr_policy.txt
├── uploads/                              # Runtime folder for user-uploaded documents
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

At runtime the app also creates (git-ignored): `chroma_db/`,
`agent_memory.db`, `client_profiles.db`, `generated_reports/`.

---

## 4. Installation

```bash
git clone <this-repo>
cd enterprise-research-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then edit .env and add your GROQ_API_KEY
```

## 5. Environment Variables

See `.env.example` for the full list. At minimum you need:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at **https://console.groq.com/keys**. Everything else
(search, Wikipedia, embeddings, vector store) works with no key at all.

## 6. Running the Application

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 7. Creating & Using the Knowledge Base

1. In the sidebar, upload one or more PDF/TXT files (try the files in
   `sample_data/` first).
2. Click **Create / Update Knowledge Base**. Documents are chunked,
   embedded locally and stored in Chroma at `chroma_db/`.
3. Ask the Chat Assistant a question referencing the document, e.g.
   *"What are Nova Robotics' key risk factors?"* — the `document_search`
   tool will retrieve the relevant chunks automatically.
4. The Report Generator's multi-source research also pulls from this same
   knowledge base.

## 8. How Each Major Module Works

- **agent.py** builds a LangGraph `create_agent` with the full tool set and
  a `SqliteSaver` checkpointer keyed by the "Client / Session ID" in the
  sidebar, giving each client their own conversation thread.
- **tools.py** exposes `search_tool` (DuckDuckGo), `wiki_tool` (Wikipedia),
  `document_search` (Chroma retriever), `python_tool` (Python REPL) and
  `comparison_table` (a typed pandas-based tool for "compare X vs Y").
- **multi_source.py** fetches web, Wikipedia and document results
  concurrently with `RunnableParallel`, then asks the LLM to merge them into
  one attributed answer.
- **parallel_research.py** builds a dynamic `RunnableParallel` — one branch
  per topic — so multiple companies are researched independently and
  simultaneously, then produces a short comparative overview.
- **report_pipeline.py** defines the `ResearchReport` Pydantic schema and
  chains `Research → Summarize → Generate Report → Executive Summary →
  Email Draft` with LCEL's `|` operator.
- **memory.py** provides short-term/persistent conversation memory
  (LangGraph `SqliteSaver`) and a separate long-term client-profile store
  (client name, preferred report style, frequently researched industries,
  and full report history).
- **export.py** renders a `ResearchReport` to plain text or PDF (`fpdf2`).

## 9. Gmail Setup (Module 11)

1. In [Google Cloud Console](https://console.cloud.google.com/), create a
   project and enable the **Gmail API**.
2. Create an **OAuth client ID** of type "Desktop app" and download the
   JSON file.
3. Save it in the project root as `credentials.json` (already in
   `.gitignore` — never commit it).
4. The first time you use "Send via Gmail" or a chat request that emails
   something, a browser window will open for you to authorize access; a
   `token_gmail.json` is then cached for future runs.
5. If `credentials.json` is missing, the app still runs — Gmail actions
   show a clear "not configured" message instead of crashing.

## 10. Example Prompts

- *"Research Tesla's AI strategy using the latest news and Wikipedia."*
- *"What are Nova Robotics' risk factors?"* (after uploading the sample doc)
- *"Compare revenue growth"* → use the Report Generator with
  `Google, Microsoft, Amazon, OpenAI` to see the parallel-research +
  comparison feature.
- *"Email the report to manager@company.com."*

## 11. Expected Workflow (Report Generator tab)

```
Enter topic(s) → Parallel/multi-source research → Merge information
→ Generate structured report → Save to memory → Display report
→ Download (TXT/PDF) → (optional) Send via Gmail
```

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Add it to `.env` and restart the app. |
| "No documents have been uploaded yet" | Upload + click "Create/Update Knowledge Base" first. |
| Gmail button shows "not configured" | Follow section 9; this is expected until `credentials.json` is added. |
| `to_markdown` / comparison table error | Ensure `tabulate` is installed (it's in `requirements.txt`). |
| Slow first response | The local embedding model downloads on first run; subsequent runs are fast. |

## 13. Project Deliverables

- ✅ Source code (this repository)
- ✅ Streamlit application (`app.py`)
- ✅ Requirements file (`requirements.txt`)
- ✅ Sample company documents (`sample_data/`)
- ✅ `.env.example` and configuration management (`config.py`)
- ✅ This README

## 14. Handbook Requirement Checklist

| Requirement | Implemented |
|---|---|
| AI Chat Assistant | ✅ |
| Internet Research | ✅ |
| Wikipedia Research | ✅ |
| Document RAG | ✅ |
| Multi-Source Research | ✅ |
| Structured Output | ✅ |
| Parallel Agent | ✅ |
| Sequential Chain | ✅ |
| Memory (short-term/persistent/long-term) | ✅ |
| Python / Data-Analysis Tool | ✅ |
| Gmail Integration | ✅ |
| Streamlit UI | ✅ |
| TXT/PDF Export | ✅ |
| Documentation | ✅ |
| Requirements File | ✅ |

"""
export.py

Report export/download support (TXT and PDF), used by the Streamlit UI's
download buttons and by the Gmail send action for attaching a PDF copy.
"""

import os
import re
from datetime import datetime

from fpdf import FPDF

import config
from report_pipeline import ResearchReport


def _safe_filename(topic: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", topic.strip()).strip("_") or "report"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{timestamp}"


def _report_sections(report: ResearchReport):
    yield "Executive Summary", [report.executive_summary]
    yield "Key Findings", report.key_findings
    yield "Strengths", report.strengths
    yield "Weaknesses", report.weaknesses
    yield "Future Opportunities", report.future_opportunities
    yield "Conclusion", [report.conclusion]
    yield "References", report.references


def report_to_text(report: ResearchReport) -> str:
    lines = [report.title, "=" * len(report.title), ""]
    for heading, items in _report_sections(report):
        lines.append(heading.upper())
        lines.append("-" * len(heading))
        for item in items:
            lines.append(f"- {item}" if heading != "Executive Summary" and heading != "Conclusion" else item)
        lines.append("")
    return "\n".join(lines)


def export_report_txt(report: ResearchReport) -> str:
    """Returns the report content as plain text (used directly by st.download_button)."""
    return report_to_text(report)


def export_report_pdf(report: ResearchReport, output_dir: str = None) -> str:
    """Writes the report to a PDF file on disk and returns the file path."""
    output_dir = output_dir or config.REPORTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, report.title)
    pdf.ln(2)

    for heading, items in _report_sections(report):
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading)
        pdf.set_font("Helvetica", "", 11)
        for item in items:
            bullet = f"- {item}" if heading not in ("Executive Summary", "Conclusion") else item
            pdf.multi_cell(0, 6, bullet)
        pdf.ln(3)

    filename = _safe_filename(report.title) + ".pdf"
    path = os.path.join(output_dir, filename)
    pdf.output(path)
    return path

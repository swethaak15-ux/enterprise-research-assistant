"""
gmail_tools.py

Module 11 - Gmail Integration

Provides:
  * get_gmail_tools()  - LangChain tools the conversational agent can call
                          itself when the user asks it to email something.
  * send_email()        - a direct helper used by the Report Generator tab
                          to send a finished report without going through
                          the agent's tool-selection step.

Gmail is optional. If ``credentials.json`` is not present, both functions
fail gracefully with a clear message instead of crashing the app.
"""

import base64
import mimetypes
import os
from email.message import EmailMessage
from typing import List, Optional

import config


class GmailNotConfiguredError(RuntimeError):
    pass


def _require_credentials() -> None:
    if not config.is_gmail_configured():
        raise GmailNotConfiguredError(
            f"Gmail is not configured. Add an OAuth client-secret file at "
            f"'{config.GMAIL_CREDENTIALS_FILE}' (see README 'Gmail Setup')."
        )


def get_gmail_tools() -> List:
    """
    Returns Gmail tools for the conversational agent's toolkit. Returns an
    empty list (instead of raising) when Gmail is not configured, so the
    agent still works for research even without email set up.
    """
    if not config.is_gmail_configured():
        return []

    from langchain_google_community import GmailToolkit
    from langchain_google_community.gmail.utils import build_resource_service, get_gmail_credentials

    credentials = get_gmail_credentials(
        token_file=config.GMAIL_TOKEN_FILE,
        scopes=["https://mail.google.com/"],
        client_secrets_file=config.GMAIL_CREDENTIALS_FILE,
    )

    api_resource = build_resource_service(credentials=credentials)
    toolkit = GmailToolkit(api_resource=api_resource)

    return toolkit.get_tools()


def send_email(to: str, subject: str, body: str, attachment_path: Optional[str] = None) -> str:
    """
    Send an email directly via the Gmail API. Used to deliver a generated
    report (Module 11) after the sequential report pipeline (Module 8)
    prepares the draft.
    """
    _require_credentials()

    from langchain_google_community.gmail.utils import build_resource_service, get_gmail_credentials

    credentials = get_gmail_credentials(
        token_file=config.GMAIL_TOKEN_FILE,
        scopes=["https://mail.google.com/"],
        client_secrets_file=config.GMAIL_CREDENTIALS_FILE,
    )
    service = build_resource_service(credentials=credentials)

    message = EmailMessage()
    message.set_content(body)
    message["To"] = to
    message["Subject"] = subject

    if attachment_path and os.path.exists(attachment_path):
        mime_type, _ = mimetypes.guess_type(attachment_path)
        maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
        with open(attachment_path, "rb") as f:
            message.add_attachment(
                f.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(attachment_path),
            )

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    create_message = {"raw": encoded_message}

    sent = service.users().messages().send(userId="me", body=create_message).execute()
    return f"Email sent to {to}. Message ID: {sent.get('id')}"

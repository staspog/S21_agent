"""Derive checkpoint keys scoped to a verified platform identity."""

import hashlib


def scoped_thread_id(subject: str, session_id: str) -> str:
    if not subject or not session_id:
        raise ValueError("subject and session ID are required")
    # Length delimiters make the encoding unambiguous even for arbitrary IDs.
    value = f"{len(subject)}:{subject}{len(session_id)}:{session_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

"""User-visible session references: "name (S3)", or "S3" where space is short.

S indices are 0-based positions in the session list (``session.id``).
"""


def session_tag(session_id):
    return f"S{int(session_id)}"


def session_label(sessions, session_id=None, *, short=False):
    """``sessions``: Data (its sessions) or a sequence of sessions, with
    ``session_id``; or a single session (``session_id`` None, its ``id``)."""
    if session_id is None:
        session = sessions
        session_id = getattr(session, "id", None)

        # Not registered yet (id -1): no position in the list.
        if session_id is None or session_id < 0:
            return str(getattr(session, "name", "") or "")
    else:
        sessions = getattr(sessions, "sessions", sessions)

        try:
            session = sessions[int(session_id)]
        except (IndexError, TypeError):
            session = None

    tag = session_tag(session_id)
    name = None if short else getattr(session, "name", None)

    return f"{name} ({tag})" if name else tag


def session_labels(sessions, session_ids, *, short=False):
    """Comma-separated labels of several sessions."""
    return ", ".join(session_label(sessions, sid, short=short) for sid in session_ids)

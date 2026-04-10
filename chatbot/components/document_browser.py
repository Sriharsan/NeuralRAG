from __future__ import annotations

from datetime import datetime

import streamlit as st

FORMAT_ICONS = {
    "PDF": "📄",
    "DOCX": "📝",
    "TXT": "📄",
    "CSV": "📊",
    "XLSX": "📊",
    "HTML": "🌐",
    "HTM": "🌐",
    "PNG": "🖼️",
    "JPG": "🖼️",
    "JPEG": "🖼️",
    "MD": "📘",
    "MARKDOWN": "📘",
}


def _format_indexed_at(indexed_at: str) -> str:
    """Format an ISO timestamp for display."""
    if not indexed_at:
        return "Unknown date"
    try:
        return datetime.fromisoformat(indexed_at).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return indexed_at


def get_format_icon(file_type: str) -> str:
    """Return the display icon for a file type."""
    return FORMAT_ICONS.get(file_type.upper(), "📁")


def render_document_browser(
    documents: list[dict[str, str]],
    active_source_id: str | None,
) -> tuple[str | None, str | None]:
    """
    Render the indexed document browser in the sidebar.

    Args:
        documents: Indexed document metadata.
        active_source_id: Currently active document scope.

    Returns:
        A tuple of ``(scope_source_id, remove_source_id)``.
    """
    scope_source_id: str | None = None
    remove_source_id: str | None = None

    st.sidebar.markdown(
        f"### Indexed Documents <span class='doc-badge'>{len(documents)}</span>",
        unsafe_allow_html=True,
    )

    if not documents:
        st.sidebar.info("No documents indexed yet.")
        return None, None

    for document in documents:
        icon = get_format_icon(document["file_type"])
        is_active = document["source_id"] == active_source_id
        st.sidebar.markdown(
            f"""
            <div class="doc-card">
                <div><strong>{icon} {document["file_name"]}</strong>
                <span class="doc-badge">{document["file_type"]}</span></div>
                <div class="doc-meta">Indexed: {_format_indexed_at(document.get("indexed_at", ""))}</div>
                <div class="doc-meta">{"Scoped to this doc" if is_active else "Available for global search"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left_column, right_column = st.sidebar.columns(2)
        with left_column:
            if st.button("🔍 Query this doc", key=f"scope_{document['source_id']}", use_container_width=True):
                scope_source_id = document["source_id"]
        with right_column:
            if st.button("🗑️ Remove", key=f"remove_{document['source_id']}", use_container_width=True):
                remove_source_id = document["source_id"]

    return scope_source_id, remove_source_id

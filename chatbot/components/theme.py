from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "background": "#1a1a2e",
        "sidebar": "#16213e",
        "accent": "#0f3460",
        "surface": "#222645",
        "surface_alt": "#28304d",
        "text": "#f7f7fb",
        "muted": "#c8d0eb",
        "border": "#30466a",
        "user_bubble": "#0f3460",
        "assistant_bubble": "#222645",
        "success": "#1e8e5a",
    },
    "light": {
        "background": "#ffffff",
        "sidebar": "#f0f2f6",
        "accent": "#4a90d9",
        "surface": "#f8faff",
        "surface_alt": "#edf4ff",
        "text": "#172033",
        "muted": "#54627a",
        "border": "#d2dced",
        "user_bubble": "#dcecff",
        "assistant_bubble": "#f3f6fb",
        "success": "#d9f5e6",
    },
}


def _normalize_theme(value: str | list[str] | None) -> str | None:
    """Normalize a theme value from query params."""
    if isinstance(value, list):
        value = value[0] if value else None
    if value in THEMES:
        return value
    return None


def initialize_theme() -> str:
    """Initialize the theme in session state and sync it with query params."""
    query_theme = _normalize_theme(st.query_params.get("theme"))
    if query_theme and st.session_state.get("theme") != query_theme:
        st.session_state.theme = query_theme

    if "theme" not in st.session_state:
        if query_theme:
            st.session_state.theme = query_theme
        else:
            st.session_state.theme = "light"
            components.html(
                """
                <script>
                const url = new URL(window.parent.location.href);
                if (!url.searchParams.get("theme")) {
                    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
                    url.searchParams.set("theme", prefersDark ? "dark" : "light");
                    window.parent.location.replace(url.toString());
                }
                </script>
                """,
                height=0,
            )

    return st.session_state.theme


def apply_theme(theme_name: str | None = None) -> dict[str, str]:
    """Apply the current theme CSS and return its token map."""
    theme_name = theme_name or initialize_theme()
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {theme["background"]};
            --sidebar-bg: {theme["sidebar"]};
            --accent: {theme["accent"]};
            --surface: {theme["surface"]};
            --surface-alt: {theme["surface_alt"]};
            --text-color: {theme["text"]};
            --muted-color: {theme["muted"]};
            --border-color: {theme["border"]};
            --user-bubble: {theme["user_bubble"]};
            --assistant-bubble: {theme["assistant_bubble"]};
            --success-bg: {theme["success"]};
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background: var(--app-bg);
            color: var(--text-color);
            transition: background 0.3s ease, color 0.3s ease;
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar-bg);
            transition: background 0.3s ease;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text-color);
        }}

        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: var(--text-color);
            transition: color 0.3s ease;
        }}

        .stTextInput input, .stChatInput input {{
            background: var(--surface);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            transition: all 0.3s ease;
        }}

        .stButton button {{
            background: var(--accent);
            color: #ffffff;
            border: 1px solid var(--accent);
            border-radius: 12px;
            transition: all 0.3s ease;
        }}

        [data-testid="stChatMessage"] {{
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 0.35rem 0.75rem;
            background: var(--assistant-bubble);
            transition: all 0.3s ease;
        }}

        .doc-card {{
            background: var(--surface);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            transition: all 0.3s ease;
        }}

        .doc-badge {{
            display: inline-block;
            margin-left: 0.4rem;
            padding: 0.15rem 0.45rem;
            border-radius: 999px;
            background: var(--surface-alt);
            color: var(--text-color);
            font-size: 0.78rem;
            font-weight: 600;
        }}

        .doc-meta, .theme-muted {{
            color: var(--muted-color);
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        .metrics-table th, .metrics-table td {{
            border: 1px solid var(--border-color);
            padding: 0.7rem;
            background: var(--surface);
            color: var(--text-color);
            text-align: left;
        }}

        .metrics-table .local-win {{
            background: var(--success-bg);
            font-weight: 700;
        }}

        .summary-box {{
            border: 1px solid #56b37d;
            background: rgba(86, 179, 125, 0.15);
            border-radius: 14px;
            padding: 1rem;
            margin-top: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return theme


def render_theme_toggle() -> str:
    """Render the top-right theme toggle and return the active theme name."""
    theme_name = initialize_theme()
    left_column, right_column = st.columns([9, 1])
    with left_column:
        st.write("")
    with right_column:
        toggle_label = "🌙" if theme_name == "light" else "☀️"
        if st.button(toggle_label, key="theme_toggle", help="Toggle theme", use_container_width=True):
            new_theme = "dark" if theme_name == "light" else "light"
            st.session_state.theme = new_theme
            st.query_params["theme"] = new_theme
            st.rerun()
    return st.session_state.theme

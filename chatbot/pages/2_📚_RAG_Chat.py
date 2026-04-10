# ruff: noqa: E402
import sys
from pathlib import Path

import streamlit as st

CHATBOT_DIR = Path(__file__).resolve().parents[1]
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))

from app_pages import get_default_rag_parameters, render_rag_page

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

render_rag_page(get_default_rag_parameters())

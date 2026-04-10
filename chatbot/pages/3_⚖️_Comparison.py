# ruff: noqa: E402
import sys
from pathlib import Path

import streamlit as st

CHATBOT_DIR = Path(__file__).resolve().parents[1]
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))

from app_pages import get_default_chat_parameters, render_comparison_page

st.set_page_config(page_title="LLM Comparison", page_icon="⚖️", layout="wide")

render_comparison_page(get_default_chat_parameters())

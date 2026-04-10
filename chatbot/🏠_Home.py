# ruff: noqa: I001
from app_pages import render_home_page
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="🏠", layout="wide")

render_home_page()

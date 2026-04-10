# ruff: noqa: I001
from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import (
    answer,
    answer_with_context,
    extract_content_after_reasoning,
    refine_question,
)
from bot.conversation.ctx_strategy import BaseSynthesisStrategy, get_ctx_synthesis_strategy
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import get_models, get_model_settings
from components.document_browser import render_document_browser
from components.theme import apply_theme, render_theme_toggle
from components.voice_input import render_voice_chat_input
from document_loader.format import Format
from document_loader.loader import load_uploaded_document
from document_loader.text_splitter import create_recursive_text_splitter
from entities.document import Document
from helpers.log import get_logger
from helpers.prettier import prettify_source

logger = get_logger(__name__)

SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "csv", "xlsx", "html", "png", "jpg", "jpeg", "md", "markdown"]
COMPARISON_METRICS = [
    ("Cost Per Query", "$0.00 FREE ✅", "~$0.015 💰", "~$0.010 💰", "~$0.007 💰"),
    ("Data Privacy", "100% Private ✅", "Sent to API ❌", "Sent to API ❌", "Sent to API ❌"),
    ("Works Offline", "Yes ✅", "No ❌", "No ❌", "No ❌"),
    ("Rate Limits", "None ✅", "Yes ❌", "Yes ❌", "Yes ❌"),
    ("No Subscription Needed", "Yes ✅", "Yes ❌", "Yes ❌", "Yes ❌"),
    ("Full Customization", "Yes ✅", "No ❌", "No ❌", "No ❌"),
    ("Runs on Your Hardware", "Yes ✅", "No ❌", "No ❌", "No ❌"),
    ("No Data Retention Risk", "Yes ✅", "No ❌", "No ❌", "No ❌"),
    ("Response Latency", "{local_latency}", "{claude_latency}", "{openai_latency}", "{gemini_latency}"),
    ("One-Time Setup Cost", "Free ✅", "Monthly Fee ❌", "Monthly Fee ❌", "Monthly Fee ❌"),
]


@dataclass(frozen=True)
class ChatPageParameters:
    """Runtime parameters for the normal chat page."""

    model: str
    max_new_tokens: int = 512


@dataclass(frozen=True)
class RagPageParameters:
    """Runtime parameters for the RAG chat page."""

    model: str
    synthesis_strategy: str
    k: int = 2
    max_new_tokens: int = 512
    chunk_size: int = 1000
    chunk_overlap: int = 50


@st.cache_resource(show_spinner=False)
def init_llm_client(model_folder: Path, model_name: str) -> LamaCppClient:
    """Create and cache the local llama.cpp client."""
    return LamaCppClient(model_folder=model_folder, model_settings=get_model_settings(model_name))


@st.cache_resource(show_spinner=False)
def init_index(vector_store_path: Path) -> Chroma:
    """Create and cache the persistent Chroma index."""
    return Chroma(is_persistent=True, persist_directory=str(vector_store_path), embedding=Embedder())


@st.cache_resource(show_spinner=False)
def init_ctx_synthesis_strategy(ctx_synthesis_strategy_name: str, _llm: LamaCppClient) -> BaseSynthesisStrategy:
    """Create and cache the context synthesis strategy."""
    return get_ctx_synthesis_strategy(ctx_synthesis_strategy_name, llm=_llm)


@st.cache_resource(show_spinner=False)
def init_chat_history(history_key: str, total_length: int = 2) -> ChatHistory:
    """Create a page-specific chat history."""
    _ = history_key
    return ChatHistory(total_length=total_length)


def get_root_folder() -> Path:
    """Return the project root folder."""
    return Path(__file__).resolve().parent.parent


def get_default_chat_parameters() -> ChatPageParameters:
    """Build default parameters for the normal chat page."""
    return ChatPageParameters(model=get_models()[0], max_new_tokens=512)


def get_default_rag_parameters() -> RagPageParameters:
    """Build default parameters for the RAG chat page."""
    return RagPageParameters(model=get_models()[0], synthesis_strategy="async-tree-summarization")


def render_page_chrome(title: str, icon_path: Path | None, sidebar_title: str) -> None:
    """Render the shared page header, sidebar title, and theme controls."""
    render_theme_toggle()
    apply_theme()
    left_column, center_column, right_column = st.columns([2, 1, 2])
    with left_column:
        st.write("")
    with center_column:
        if icon_path:
            st.image(str(icon_path), use_container_width=True)
        st.markdown(f"### {title}")
    with right_column:
        st.write("")
    st.sidebar.title(sidebar_title)


def get_message_store(session_key: str) -> list[dict[str, str]]:
    """Return the page-scoped message list."""
    return st.session_state.setdefault(session_key, [])


def append_message(session_key: str, role: str, content: str) -> None:
    """Append a message to the page-scoped history."""
    get_message_store(session_key).append({"role": role, "content": content})


def clear_message_store(session_key: str, chat_history: ChatHistory) -> None:
    """Clear page-scoped messages and the matching chat history."""
    st.session_state[session_key] = []
    chat_history.clear()


def maybe_reset_chat_history(button_key: str, session_key: str, chat_history: ChatHistory) -> None:
    """Render the clear-conversation button and reset state when clicked."""
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Conversation", key=button_key, use_container_width=True):
        clear_message_store(session_key, chat_history)
        st.rerun()
    get_message_store(session_key)


def display_messages(session_key: str) -> None:
    """Replay the stored chat messages to the screen."""
    for message in get_message_store(session_key):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def get_chunk_format(file_name: str) -> str:
    """Choose the text splitter format for a document."""
    if Path(file_name).suffix.lower() in {".md", ".markdown"}:
        return Format.MARKDOWN.value
    return Format.TEXT.value


def build_document_chunks(document: Document, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split a loaded document into vector-store chunks."""
    splitter = create_recursive_text_splitter(
        format=get_chunk_format(document.metadata.get("file_name", document.metadata.get("source", ""))),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents([document])


def index_uploaded_files(
    uploaded_files: list[Any],
    index: Chroma,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    """
    Load, chunk, and index uploaded files.

    Returns:
        A tuple of ``(indexed_file_count, indexed_chunk_count)``.
    """
    indexed_file_count = 0
    indexed_chunk_count = 0
    progress_bar = st.sidebar.progress(0.0, text="Preparing uploads...")

    for position, uploaded_file in enumerate(uploaded_files, start=1):
        progress_bar.progress((position - 1) / max(len(uploaded_files), 1), text=f"Reading {uploaded_file.name}...")
        document = load_uploaded_document(uploaded_file.name, uploaded_file.getvalue())
        document.metadata["indexed_at"] = datetime.now().isoformat(timespec="seconds")
        chunks = build_document_chunks(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        progress_bar.progress(
            min(position / max(len(uploaded_files), 1), 1.0),
            text=f"Indexing {uploaded_file.name} ({len(chunks)} chunks)...",
        )
        if chunks:
            index.from_chunks(chunks)
            indexed_file_count += 1
            indexed_chunk_count += len(chunks)

    progress_bar.empty()
    return indexed_file_count, indexed_chunk_count


def render_document_management(index: Chroma, parameters: RagPageParameters) -> None:
    """Render the RAG document upload and document browser controls."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Knowledge Base")
    active_source_id = st.session_state.get("rag_active_source_id")
    documents = index.get_indexed_documents()
    scoped_source_id, removed_source_id = render_document_browser(documents, active_source_id)

    if scoped_source_id:
        st.session_state.rag_active_source_id = scoped_source_id
        st.rerun()

    if removed_source_id:
        index.delete_document(removed_source_id)
        if st.session_state.get("rag_active_source_id") == removed_source_id:
            st.session_state.rag_active_source_id = None
        st.sidebar.success("Document removed from the index.")
        st.rerun()

    if active_source_id:
        active_document = next((item for item in documents if item["source_id"] == active_source_id), None)
        if active_document:
            st.sidebar.info(f"Scoped to: {active_document['file_name']}")
        if st.sidebar.button("Clear document scope", key="clear_doc_scope", use_container_width=True):
            st.session_state.rag_active_source_id = None
            st.rerun()

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Files are auto-indexed immediately after upload.",
    )
    if uploaded_files:
        try:
            with st.sidebar:
                with st.spinner("Auto-indexing uploaded files..."):
                    file_count, chunk_count = index_uploaded_files(
                        uploaded_files,
                        index=index,
                        chunk_size=parameters.chunk_size,
                        chunk_overlap=parameters.chunk_overlap,
                    )
            st.sidebar.success(f"Indexed {file_count} file(s) into {chunk_count} chunks.")
            st.rerun()
        except Exception as error:
            logger.error("Indexing failed: %s", error, exc_info=True)
            st.sidebar.error(f"Could not index the uploaded file(s): {error}")


def format_retrieval_preview(sources: list[dict[str, Any]]) -> str:
    """Format retrieved sources for display in the chat transcript."""
    if not sources:
        return "I did not detect any pertinent chunk of text from the documents."
    preview = "Here are the retrieved text chunks with a content preview:\n\n"
    for source in sources:
        preview += prettify_source(source)
        preview += "\n\n"
    return preview


def resolve_final_answer(llm: LamaCppClient, full_response: str, empty_message: str) -> str:
    """Resolve a visible answer for models that may emit hidden reasoning."""
    if llm.model_settings.reasoning:
        answer_text = extract_content_after_reasoning(full_response, llm.model_settings.reasoning_stop_tag)
        return answer_text or empty_message
    return full_response


def render_home_page() -> None:
    """Render the landing page for the multipage app."""
    root_folder = get_root_folder()
    render_page_chrome("RAG Chatbot Workspace", root_folder / "images" / "bot.png", "Navigate")
    st.write(
        "Explore the local chatbot, the RAG workflow with universal document indexing, "
        "and a side-by-side local vs cloud comparison page."
    )
    st.markdown(
        """
        <div class="doc-card">
            <h4>What’s Included</h4>
            <p class="theme-muted">
                Universal document indexing, scoped retrieval, voice input, and a persistent light/dark theme.
            </p>
        </div>
        <div class="doc-card">
            <h4>Pages</h4>
            <p class="theme-muted">Use the sidebar to open Normal Chat, RAG Chat, and the Comparison page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_page(parameters: ChatPageParameters) -> None:
    """Render the normal chatbot page."""
    root_folder = get_root_folder()
    render_page_chrome("Chatbot", root_folder / "images" / "bot-small.png", "Options")

    llm = init_llm_client(root_folder / "models", parameters.model)
    chat_history = init_chat_history("chat_history")
    maybe_reset_chat_history("clear_chat_history", "chat_messages", chat_history)

    if not get_message_store("chat_messages"):
        with st.chat_message("assistant"):
            st.write("How can I help you today?")
    display_messages("chat_messages")

    voice_result = render_voice_chat_input("chat_input", "Input your question!")
    user_input = voice_result.get("submitted_text")
    if not user_input:
        return

    st.session_state["chat_input_draft"] = ""
    append_message("chat_messages", "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    start_time = time.time()
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Generating the answer..."):
            for token in answer(
                llm=llm,
                question=user_input,
                chat_history=chat_history,
                max_new_tokens=parameters.max_new_tokens,
            ):
                full_response += llm.parse_token(token)
                message_placeholder.markdown(full_response + "▌")
        final_answer = resolve_final_answer(llm, full_response, "I didn't provide the answer; perhaps I can try again.")
        message_placeholder.markdown(final_answer)

    chat_history.append(f"question: {user_input}, answer: {final_answer}")
    append_message("chat_messages", "assistant", final_answer)
    logger.info("--- Took %.2f seconds ---", time.time() - start_time)


def render_rag_page(parameters: RagPageParameters) -> None:
    """Render the RAG chatbot page with document indexing and scoped retrieval."""
    root_folder = get_root_folder()
    render_page_chrome("RAG Chatbot", root_folder / "images" / "bot.png", "Tools & Settings")

    llm = init_llm_client(root_folder / "models", parameters.model)
    chat_history = init_chat_history("rag_history")
    ctx_synthesis_strategy = init_ctx_synthesis_strategy(parameters.synthesis_strategy, _llm=llm)
    index = init_index(root_folder / "vector_store" / "docs_index")

    render_document_management(index, parameters)
    maybe_reset_chat_history("clear_rag_history", "rag_messages", chat_history)

    if not get_message_store("rag_messages"):
        with st.chat_message("assistant"):
            st.write("How can I help you today?")
    display_messages("rag_messages")

    voice_result = render_voice_chat_input("rag_input", "Ask about your indexed documents...")
    user_input = voice_result.get("submitted_text")
    if not user_input:
        return

    st.session_state["rag_input_draft"] = ""
    append_message("rag_messages", "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    active_source_id = st.session_state.get("rag_active_source_id")
    retrieval_filter = {"source_id": active_source_id} if active_source_id else None

    with st.chat_message("assistant"):
        with st.spinner("Refining the question and retrieving relevant context..."):
            refined_user_input = refine_question(
                llm,
                user_input,
                chat_history=chat_history,
                max_new_tokens=parameters.max_new_tokens,
            )
            retrieved_contents, sources = index.similarity_search_with_threshold(
                query=refined_user_input,
                k=parameters.k,
                filter=retrieval_filter,
            )
        retrieval_preview = format_retrieval_preview(sources)
        st.markdown(retrieval_preview)
        append_message("rag_messages", "assistant", retrieval_preview)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Refining the context and generating the answer..."):
            streamer, _ = answer_with_context(
                llm,
                ctx_synthesis_strategy,
                user_input,
                chat_history,
                retrieved_contents,
                max_new_tokens=parameters.max_new_tokens,
            )
            for token in streamer:
                full_response += llm.parse_token(token)
                message_placeholder.markdown(full_response + "▌")
        answer_text = resolve_final_answer(
            llm,
            full_response,
            "I wasn't able to provide the answer; do you want me to try again?",
        )
        chat_history.append(f"question: {user_input}, answer: {answer_text}")
        message_placeholder.markdown(answer_text)
        append_message("rag_messages", "assistant", answer_text)


def load_api_environment(root_folder: Path) -> dict[str, str]:
    """Load API keys from the project .env file."""
    load_dotenv(root_folder / ".env", override=False)
    return {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
    }


def run_local_comparison_question(llm: LamaCppClient, question: str, max_new_tokens: int) -> dict[str, str]:
    """Run a comparison question against the local model."""
    started = time.perf_counter()
    content = llm.generate_answer(llm.generate_qa_prompt(question), max_new_tokens=max_new_tokens)
    return {
        "content": resolve_final_answer(llm, content, "The local model did not return a final answer."),
        "latency_ms": f"{(time.perf_counter() - started) * 1000:.0f} ms",
    }


def run_claude_question(api_key: str, question: str) -> dict[str, str]:
    """Run a comparison question against Claude."""
    if not api_key:
        return {"content": "API key not configured", "latency_ms": "N/A"}
    import anthropic

    started = time.perf_counter()
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    )
    content = "".join(block.text for block in message.content if getattr(block, "text", None))
    return {
        "content": content or "No response returned.",
        "latency_ms": f"{(time.perf_counter() - started) * 1000:.0f} ms",
    }


def run_openai_question(api_key: str, question: str) -> dict[str, str]:
    """Run a comparison question against OpenAI GPT-4o."""
    if not api_key:
        return {"content": "API key not configured", "latency_ms": "N/A"}
    from openai import OpenAI

    started = time.perf_counter()
    client = OpenAI(api_key=api_key)
    response = client.responses.create(model="gpt-4o", input=question)
    return {
        "content": response.output_text or "No response returned.",
        "latency_ms": f"{(time.perf_counter() - started) * 1000:.0f} ms",
    }


def run_gemini_question(api_key: str, question: str) -> dict[str, str]:
    """Run a comparison question against Gemini."""
    if not api_key:
        return {"content": "API key not configured", "latency_ms": "N/A"}
    import google.generativeai as genai

    started = time.perf_counter()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(question)
    return {
        "content": getattr(response, "text", "") or "No response returned.",
        "latency_ms": f"{(time.perf_counter() - started) * 1000:.0f} ms",
    }


def safe_model_call(label: str, func: Any, *args: Any) -> tuple[str, dict[str, str]]:
    """Wrap a model call and convert failures into user-friendly output."""
    try:
        return label, func(*args)
    except ModuleNotFoundError as error:
        return label, {"content": f"Dependency not installed: {error.name}", "latency_ms": "N/A"}
    except Exception as error:
        logger.error("%s comparison failed: %s", label, error, exc_info=True)
        return label, {"content": f"Request failed: {error}", "latency_ms": "N/A"}


def render_metrics_table(results: dict[str, dict[str, str]]) -> None:
    """Render the comparison metrics table."""
    rows: list[str] = []
    for metric, local_value, claude_value, openai_value, gemini_value in COMPARISON_METRICS:
        rows.append(
            f"""
            <tr>
                <td>{metric}</td>
                <td class="local-win">{local_value.format(local_latency=results["Local Model"]["latency_ms"])}</td>
                <td>{claude_value.format(claude_latency=results["Claude"]["latency_ms"])}</td>
                <td>{openai_value.format(openai_latency=results["OpenAI GPT-4o"]["latency_ms"])}</td>
                <td>{gemini_value.format(gemini_latency=results["Gemini"]["latency_ms"])}</td>
            </tr>
            """
        )
    st.markdown(
        """
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Local Model</th>
                    <th>Claude</th>
                    <th>OpenAI GPT-4o</th>
                    <th>Gemini</th>
                </tr>
            </thead>
            <tbody>
        """
        + "".join(rows)
        + """
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_local_advantage_summary() -> None:
    """Render the fixed local-model advantage summary box."""
    st.markdown(
        """
        <div class="summary-box">
            <strong>Why Local Wins</strong>
            <p>1. Your prompts and indexed documents stay on your machine for stronger privacy.</p>
            <p>2. The local model keeps working offline without API quotas, subscriptions, or retention risk.</p>
            <p>3. You control the model, hardware, and runtime end to end.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_page(parameters: ChatPageParameters | None = None) -> None:
    """Render the local-vs-cloud comparison page."""
    parameters = parameters or get_default_chat_parameters()
    root_folder = get_root_folder()
    render_page_chrome("Local vs Cloud LLM Comparison", root_folder / "images" / "bot-small.png", "Comparison")

    api_keys = load_api_environment(root_folder)
    llm = init_llm_client(root_folder / "models", parameters.model)
    question = st.text_area(
        "Ask one question to compare all models",
        placeholder="Explain retrieval-augmented generation in simple terms.",
    )

    if not st.button("Run comparison", key="run_comparison", use_container_width=True):
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return

    call_map = {
        "Local Model": (run_local_comparison_question, llm, question, parameters.max_new_tokens),
        "Claude": (run_claude_question, api_keys["ANTHROPIC_API_KEY"], question),
        "OpenAI GPT-4o": (run_openai_question, api_keys["OPENAI_API_KEY"], question),
        "Gemini": (run_gemini_question, api_keys["GOOGLE_API_KEY"], question),
    }

    results: dict[str, dict[str, str]] = {}
    with st.spinner("Running all model comparisons..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(safe_model_call, label, func_and_args[0], *func_and_args[1:])
                for label, func_and_args in call_map.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                label, result = future.result()
                results[label] = result

    ordered_labels = ["Local Model", "Claude", "OpenAI GPT-4o", "Gemini"]
    columns = st.columns(4)
    for column, label in zip(columns, ordered_labels):
        with column:
            st.markdown(f"#### {label}")
            st.write(results[label]["content"])
            st.caption(f"Latency: {results[label]['latency_ms']}")

    render_metrics_table(results)
    render_local_advantage_summary()

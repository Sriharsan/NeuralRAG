# ruff: noqa: C901, I001
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import streamlit as st

from helpers.log import get_logger

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def load_whisper_model() -> Any:
    """Load the Whisper model once for offline transcription."""
    import whisper

    return whisper.load_model("base")


def _transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe recorded audio bytes with Whisper."""
    model = load_whisper_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        result = model.transcribe(temp_audio_path, fp16=False)
        return result.get("text", "").strip()
    finally:
        Path(temp_audio_path).unlink(missing_ok=True)


def render_voice_chat_input(input_key: str, placeholder: str) -> dict[str, str | None]:
    """
    Render a text, send, and voice input row for chat pages.

    Args:
        input_key: Stable key prefix for widget state.
        placeholder: Placeholder text for the text input.

    Returns:
        A small dictionary describing any submitted text, transcription, or error.
    """
    result: dict[str, str | None] = {"submitted_text": None, "transcription": None, "error": None}
    draft_key = f"{input_key}_draft"
    recording_key = f"{input_key}_recording"
    auto_send_key = f"{input_key}_auto_send"
    caption_key = f"{input_key}_caption"
    error_key = f"{input_key}_voice_error"

    st.session_state.setdefault(draft_key, "")
    st.session_state.setdefault(recording_key, False)

    text_column, mic_column, send_column = st.columns([8, 1, 1])
    with text_column:
        st.text_input("Message", key=draft_key, placeholder=placeholder, label_visibility="collapsed")
    with mic_column:
        if st.button("🎤", key=f"{input_key}_mic", use_container_width=True):
            st.session_state[recording_key] = not st.session_state[recording_key]
            if not st.session_state[recording_key]:
                st.session_state.pop(error_key, None)
            st.rerun()
    with send_column:
        send_clicked = st.button("Send", key=f"{input_key}_send", use_container_width=True)

    if st.session_state.get(recording_key):
        st.caption("🔴 Recording... speak now")
        try:
            from audio_recorder_streamlit import audio_recorder

            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#4a90d9",
                icon_name="microphone",
                icon_size="2x",
                key=f"{input_key}_audio_recorder",
            )
            if audio_bytes:
                with st.spinner("⏳ Transcribing..."):
                    transcription = _transcribe_audio(audio_bytes)
                if transcription:
                    st.session_state[draft_key] = transcription
                    st.session_state[caption_key] = transcription
                    st.session_state[auto_send_key] = True
                else:
                    st.session_state[error_key] = "I couldn't detect speech in that recording."
                st.session_state[recording_key] = False
                st.rerun()
        except ModuleNotFoundError as error:
            st.session_state[recording_key] = False
            if error.name == "audio_recorder_streamlit":
                st.session_state[error_key] = "Install audio recorder support: pip install audio-recorder-streamlit"
            else:
                st.session_state[error_key] = "Install whisper: pip install openai-whisper"
            st.rerun()
        except Exception as error:
            logger.error("Voice input failed: %s", error, exc_info=True)
            st.session_state[recording_key] = False
            st.session_state[error_key] = f"Voice input failed: {error}"
            st.rerun()

    if st.session_state.get(error_key):
        st.info(st.session_state[error_key])

    if st.session_state.get(caption_key):
        result["transcription"] = st.session_state[caption_key]
        st.caption(st.session_state[caption_key])

    if st.session_state.get(auto_send_key):
        st.session_state[auto_send_key] = False
        submitted_text = st.session_state.get(draft_key, "").strip()
        if submitted_text:
            result["submitted_text"] = submitted_text
            return result

    if send_clicked:
        submitted_text = st.session_state.get(draft_key, "").strip()
        if submitted_text:
            result["submitted_text"] = submitted_text

    return result

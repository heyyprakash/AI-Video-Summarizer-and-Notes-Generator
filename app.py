#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import streamlit as st
import yt_dlp
import whisper
from groq import Groq

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="AI Video Summarizer with Chatbot", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "my_key")  # set GROQ_API_KEY env var or replace here

# -----------------------------
# Clients & models
# -----------------------------
try:
    client = Groq(api_key=GROQ_API_KEY)
    st.success("Groq API configured.")
except Exception as e:
    st.error(f"Error configuring Groq API: {e}")
    st.info("Get a key from https://console.groq.com/keys")
    st.stop()

@st.cache_resource
def load_whisper_model():
    """Load Whisper model used for transcription."""
    return whisper.load_model("base")

# -----------------------------
# Helpers
# -----------------------------
def _remove_old_audio(pattern="audio.*"):
    for p in glob.glob(pattern):
        try:
            os.remove(p)
        except Exception:
            pass

def download_audio_from_youtube(url: str) -> str | None:
    """
    Download best audio from YouTube and return path to MP3.
    Requires ffmpeg in PATH.
    """
    try:
        _remove_old_audio()
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ],
            "outtmpl": "audio.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "nocheckcertificate": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        mp3_path = "audio.mp3"
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            st.error("Audio file was not created or is empty.")
            return None

        st.info(f"Downloaded audio size: {os.path.getsize(mp3_path) / 1024:.2f} KB")
        return mp3_path
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def transcribe_audio(audio_path: str, model, language_option: str):
    """
    Transcribe audio to text. Returns (text, detected_language).
    language_option in {"Auto-detect", "Hindi", "English"}.
    """
    try:
        if not os.path.exists(audio_path):
            st.error(f"Audio file not found: {audio_path}")
            return None, None
        if os.path.getsize(audio_path) == 0:
            st.error("Audio file is empty.")
            return None, None

        if language_option == "Hindi":
            result = model.transcribe(audio_path, fp16=False, language="hi")
            detected = "hindi"
        elif language_option == "English":
            result = model.transcribe(audio_path, fp16=False, language="en")
            detected = "english"
        else:
            result = model.transcribe(audio_path, fp16=False)
            detected = result.get("language", "unknown")

        text = result.get("text", "").strip()
        if not text:
            st.warning("Transcription returned empty text.")
            return None, None

        st.info(f"Language: {detected.upper() if isinstance(detected, str) else detected}")
        return text, detected
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        st.info("Tips: ensure FFmpeg is installed, try a different video, check connectivity.")
        return None, None

def call_groq_ai(transcript: str, prompt_instruction: str) -> str | None:
    """Call Groq for text generation with the provided instruction and transcript."""
    if not transcript:
        return None
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt_instruction},
                {"role": "user", "content": transcript},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API error: {e}")
        return None

def chat_with_ai(user_question: str, transcript: str, chat_history: list[dict]) -> str:
    """Chat about the video content based on the transcript and prior turns."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions about a video "
                    "based on its transcript. If the answer is not in the transcript, say so. "
                    "You can respond in multiple languages including English and Hindi.\n\n"
                    f"Transcript:\n{transcript}"
                ),
            }
        ]
        for turn in chat_history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": user_question})

        resp = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# -----------------------------
# UI
# -----------------------------
st.title("AI Video Summarizer with Smart Chatbot")
st.write("Paste a YouTube link. The app will transcribe it, generate a summary and notes, and let you ask questions.")
st.info("Powered by Whisper (transcription) and Groq (summarization and chatbot). Supports English, Hindi, and more.")

# Session state
for key, default in [
    ("transcript", None),
    ("chat_history", []),
    ("summary", None),
    ("notes", None),
    ("detected_language", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.spinner("Loading speech-to-text model..."):
    whisper_model = load_whisper_model()
st.success("Speech-to-text model ready.")

st.write("---")

col1, col2 = st.columns([3, 1])
with col1:
    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
with col2:
    language_option = st.selectbox("Language", ["Auto-detect", "Hindi", "English"])

if st.button("Generate Summary & Notes", type="primary", use_container_width=True):
    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
    else:
        st.session_state.chat_history = []

        with st.spinner("Step 1/3: Downloading audio..."):
            audio_file = download_audio_from_youtube(youtube_url)

        if audio_file and os.path.exists(audio_file):
            st.success("Audio downloaded.")
            with st.spinner("Step 2/3: Transcribing audio..."):
                transcript, detected = transcribe_audio(audio_file, whisper_model, language_option)

            if transcript:
                st.success("Transcription complete.")
                st.session_state.transcript = transcript
                st.session_state.detected_language = detected

                with st.spinner("Step 3/3: Generating summary..."):
                    summary_prompt = (
                        "You are an expert video summarizer. Provide a clear, concise summary "
                        "of the following transcript. Focus on main points and key takeaways. "
                        "Respond in the same language as the transcript."
                    )
                    st.session_state.summary = call_groq_ai(transcript, summary_prompt)

                with st.spinner("Generating detailed notes..."):
                    notes_prompt = (
                        "You are an expert note-taker. Create comprehensive, well-organized "
                        "bullet-point notes from the transcript. Include important details, key "
                        "concepts, and actionable insights. Respond in the same language as the transcript."
                    )
                    st.session_state.notes = call_groq_ai(transcript, notes_prompt)

                if st.session_state.summary and st.session_state.notes:
                    st.success("Summary and notes ready.")
                else:
                    st.error("Failed to generate summary or notes.")
            else:
                st.error("Transcription failed. Please try another video.")
            try:
                os.remove(audio_file)
            except Exception:
                pass
        else:
            st.error("Failed to download audio. Check the URL and try again.")

# -----------------------------
# Results
# -----------------------------
if st.session_state.transcript:
    st.write("---")
    with st.expander("View Full Transcript"):
        st.text_area("Transcript", st.session_state.transcript, height=300, key="transcript_display")

    if st.session_state.summary:
        st.write("---")
        st.subheader("Summary")
        st.markdown(st.session_state.summary)

    if st.session_state.notes:
        st.write("---")
        st.subheader("Detailed Notes")
        st.markdown(st.session_state.notes)

# -----------------------------
# Chatbot
# -----------------------------
if st.session_state.transcript:
    st.write("---")
    st.subheader("Ask Questions About the Video")
    st.write("Ask in English or Hindi.")

    c1, c2 = st.columns([3, 1])
    with c1:
        user_question = st.text_input(
            "Your Question",
            placeholder="e.g., What are the main points?",
            key="user_question",
        )
    with c2:
        ask_button = st.button("Ask", use_container_width=True)

    if ask_button and user_question:
        with st.spinner("Thinking..."):
            answer = chat_with_ai(user_question, st.session_state.transcript, st.session_state.chat_history)
            st.session_state.chat_history.append({"question": user_question, "answer": answer})

    if st.session_state.chat_history:
        st.write("---")
        st.subheader("Chat History")
        for turn in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {turn['question']}")
            st.markdown(f"**AI:** {turn['answer']}")
            st.write("")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.markdown(
    """
<div style='text-align: center; color: gray; padding: 20px;'>
  <p>Built with Streamlit, Whisper, and Groq.</p>
  <p>Multilingual support: English, Hindi, and many more.</p>
  <p><small>Get a Groq API key at <a href='https://console.groq.com/keys' target='_blank'>console.groq.com</a></small></p>
</div>
""",
    unsafe_allow_html=True,
)

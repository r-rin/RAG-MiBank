"""Streamlit chat UI for the MiBank RAG demo.

Run:  streamlit run app.py
"""

from __future__ import annotations

import openai
import streamlit as st

from src import config
from src.generator import stream_answer
from src.ingest import ingest
from src.retriever import retrieve
from src.vector_store import count

EXAMPLE_QUESTIONS = [
    "Скільки коштує обслуговування картки Мі Gold?",
    "Який пільговий період на кредитній картці?",
    "Що буде, якщо достроково розірвати депозит?",
    "Яка комісія за SWIFT-переказ?",
    "Що робити, якщо вкрали картку?",
    "Чи можна відкрити рахунок у біткоїнах?",  # not in the documents -> should say "no info"
]

st.set_page_config(page_title="МіБанк Асистент", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"Джерела ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] `{src['source']}`** — {src['heading'] or '—'}  \n"
                f"схожість: `{src['score']:.3f}`"
            )
            st.progress(max(0.0, min(1.0, src["score"])))
            st.code(src["text"], language="markdown", wrap_lines=True)


def render_badge(rag: bool) -> None:
    st.caption("RAG увімкнено" if rag else "RAG вимкнено — відповідь лише з «пам'яті» моделі")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Налаштування")
    use_rag = st.toggle("Використовувати RAG", value=True)
    top_k = st.slider("Кількість фрагментів (top-k)", 1, 10, config.DEFAULT_TOP_K, disabled=not use_rag)

    st.divider()
    st.subheader("База знань")
    try:
        n_chunks = count()
    except Exception as exc:  # noqa: BLE001 - show any DB problem in the UI
        n_chunks = 0
        st.error(f"Помилка ChromaDB: {exc}")
    st.write(f"Проіндексовано фрагментів: **{n_chunks}**")
    if st.button("Переіндексувати документи", use_container_width=True):
        with st.spinner("Розбиваю документи та обчислюю ембедінги..."):
            try:
                st.success(f"Готово: {ingest()} фрагментів")
            except (openai.OpenAIError, RuntimeError) as exc:
                st.error(f"Помилка OpenRouter: {exc}")

    st.divider()
    st.subheader("Поширені питання")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("Очистити чат", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"LLM: `{config.LLM_MODEL}` · Ембедінги: `{config.EMBEDDING_MODEL}` · via OpenRouter")


# ---------- Main ----------
st.title("МіБанк Асистент")

if use_rag and n_chunks == 0:
    st.warning("База знань порожня. Натисніть «Переіндексувати документи» на бічній панелі.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_badge(msg["rag"])
            if msg.get("sources"):
                render_sources(msg["sources"])

question = st.chat_input("Поставте запитання про продукти МіБанку...")
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources: list[dict] = []
        try:
            chunks = None
            if use_rag:
                with st.spinner("Шукаю релевантні фрагменти..."):
                    chunks = retrieve(question, top_k=top_k)
                sources = [c.__dict__ for c in chunks]
            answer = st.write_stream(stream_answer(question, chunks))
        except (openai.OpenAIError, RuntimeError) as exc:
            answer = f"Помилка OpenRouter: {exc}"
            st.error(answer)

        render_badge(use_rag)
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "rag": use_rag, "sources": sources}
    )

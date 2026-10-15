import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "rag_index_curated"
MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="FinRAG Intelligence",
    page_icon=None,
    layout="wide"
)

st.title("FinRAG Intelligence")
st.caption("Retrieval-Augmented Generation for Financial Education")
st.info(
    "Ask general questions about financial concepts. "
    "This educational tool is not personalized financial advice."
)


@st.cache_resource
def load_retriever():
    docs_path = INDEX_DIR / "docs.jsonl"
    embeddings_path = INDEX_DIR / "embeddings.npy"

    if not docs_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Curated RAG index not found at {INDEX_DIR}"
        )

    with docs_path.open("r", encoding="utf-8") as file:
        docs = [json.loads(line) for line in file if line.strip()]

    embeddings = np.load(embeddings_path)

    if len(docs) != len(embeddings):
        raise ValueError("Document and embedding counts do not match.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    return docs, embeddings, model


def retrieve(question, docs, embeddings, model, top_k):
    query_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = embeddings @ query_vector
    indices = np.argsort(-scores)[:min(top_k, len(docs))]

    results = []
    for idx in indices:
        score = float(scores[idx])

        # Skip documents with weak semantic similarity.
        if score < 0.30:
            continue

        doc = docs[int(idx)]
        results.append({
            "score": score,
            "text": doc.get("text") or (
                f"Question: {doc.get('instruction', '')}\\n"
                f"Answer: {doc.get('output', '')}"
            )
        })

    return results

def generate_answer(question, results):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your local .env file."
        )

    context = "\n\n".join(
        f"[{i}] {item['text']}"
        for i, item in enumerate(results, start=1)
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        reasoning_effort="low",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are FinRAG Intelligence, a financial education assistant. "
                    "Answer using only the supplied retrieved context. Do not add "
                    "unsupported facts, examples, or formulas. If the context is "
                    "insufficient, say so. Cite supporting context using [1], [2], "
                    "etc. Explain clearly and do not provide personalized financial "
                    "advice. Write all formulas in plain text using readable "
                    "operators such as /, *, +, and =. Do not use LaTeX, math "
                    "blocks, or code fences for formulas. Keep formulas on a "
                    "single line when practical."
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved context:\n{context}"
            }
        ],
        max_tokens=700
    )

    return response.choices[0].message.content


with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Retrieved documents", min_value=1, max_value=5, value=5)
    st.caption("More documents may provide broader context.")

try:
    docs, embeddings, embedding_model = load_retriever()
    st.success(f"Knowledge base loaded: {len(docs)} curated documents")
except Exception as exc:
    st.error(f"Could not load the knowledge base: {exc}")
    st.stop()

question = st.text_area(
    "Ask a financial question",
    placeholder="Example: What is the formula for P/E ratio?",
    height=100
)

if st.button("Generate answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                results = retrieve(
                    question.strip(), docs, embeddings, embedding_model, top_k
                )
                if results:
                    answer = generate_answer(question.strip(), results)
                else:
                    answer = (
                        "I could not find sufficiently relevant information "
                        "in the knowledge base to answer this question. "
                        "Please try a question about a financial concept "
                        "covered by the available documents."
                    )

            st.subheader("Answer")
            if answer:
                st.markdown(answer.replace("\\[", "$$").replace("\\]", "$$"))
            else:
                st.warning("The model returned an empty answer.")

            with st.expander("View retrieved sources and similarity scores"):
                for i, item in enumerate(results, start=1):
                    st.markdown(
                        f"**[{i}] Similarity: {item['score']:.4f}**"
                    )
                    st.text(item["text"])
                    if i < len(results):
                        st.divider()

        except Exception as exc:
            st.error(f"Request failed: {exc}")

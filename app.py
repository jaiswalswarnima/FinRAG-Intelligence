import html
import json
import os
import re
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
MIN_SIMILARITY = 0.30
MAX_SCORE_GAP = 0.35

load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="FinRAG Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #10243A;
        --blue: #245B91;
        --teal: #11766F;
        --background: #E5EBF2;
        --text: #17283B;
        --muted: #526477;
        --border: #C5D2E0;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
    }

    .stApp,
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: #E5EBF2 !important;
        color: #17283B !important;
    }

    [data-testid="stHeader"] {
        background: #E5EBF2 !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: #D5E0EC !important;
        border-right: 1px solid #B8C8D9;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li {
        color: #24394F !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #10243A !important;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #17283B;
        line-height: 1.7;
    }

    h1, h2, h3, h4 {
        color: #10243A !important;
    }

    .hero {
        background: linear-gradient(120deg, #10243A 0%, #1B456D 65%, #176E73 100%);
        border: 1px solid #315B7A;
        border-radius: 18px;
        padding: 29px 32px;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(16, 36, 58, 0.14);
    }

    .hero-kicker {
        color: #B9E4E5 !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 36px;
        font-weight: 800;
        line-height: 1.25;
        margin: 0 0 9px 0;
    }

    .hero p {
        color: #EDF4FA !important;
        font-size: 15px;
        line-height: 1.7;
        max-width: 760px;
        margin: 0;
    }

    .section-heading {
        color: #10243A !important;
        font-size: 21px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #526477 !important;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .status-card {
        background: #FFFFFF;
        border: 1px solid #C5D2E0;
        border-radius: 12px;
        padding: 13px 17px;
        margin-bottom: 23px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 3px 12px rgba(16, 36, 58, 0.04);
    }

    .status-dot {
        min-width: 10px;
        height: 10px;
        background: #16866F;
        border-radius: 50%;
        box-shadow: 0 0 0 4px #DDF4EA;
    }

    .status-title {
        color: #183650 !important;
        font-size: 13px;
        font-weight: 800;
    }

    .status-detail {
        color: #526477 !important;
        font-size: 12px;
        margin-top: 3px;
    }

    [data-testid="stTextArea"] label {
        color: #17283B !important;
        font-weight: 700 !important;
    }

    [data-testid="stTextArea"] textarea {
        background: #FFFFFF !important;
        color: #17283B !important;
        border: 1px solid #A8B9CC !important;
        border-radius: 9px !important;
        caret-color: #17283B !important;
        -webkit-text-fill-color: #17283B !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #63758A !important;
        -webkit-text-fill-color: #63758A !important;
        opacity: 1 !important;
    }

    .stButton > button {
        background: #174D77 !important;
        color: #FFFFFF !important;
        border: 1px solid #174D77 !important;
        border-radius: 9px !important;
        min-height: 43px;
        font-weight: 700 !important;
    }

    .stButton > button p,
    .stButton > button span {
        color: #FFFFFF !important;
    }

    .stButton > button:hover {
        background: #103858 !important;
        border-color: #103858 !important;
    }

    .answer-label {
        color: #11766F !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #C5D2E0 !important;
        border-left: 4px solid #11766F !important;
        border-radius: 13px !important;
        box-shadow: 0 4px 16px rgba(16, 36, 58, 0.04);
    }

    .source-card {
        background: #E8EEF5 !important;
        color: #17283B !important;
        border: 1px solid #C5D2E0;
        border-left: 4px solid #11766F;
        border-radius: 10px;
        padding: 15px 17px;
        margin: 10px 0 14px 0;
        overflow-wrap: anywhere;
    }

    .source-heading {
        color: #11675F !important;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 9px;
    }

    .source-content {
        color: #17283B !important;
        font-size: 14px;
        line-height: 1.8;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #C5D2E0 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary span {
        color: #17283B !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #C5D2E0 !important;
    }

    .footer {
        border-top: 1px solid #C5D2E0;
        padding-top: 15px;
        margin-top: 30px;
        color: #526477 !important;
        font-size: 12px;
        line-height: 1.8;
    }

    .footer strong {
        color: #203A53 !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }

        .hero {
            padding: 22px 19px;
        }

        .hero h1 {
            font-size: 28px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI-powered financial knowledge</div>
        <h1>FinRAG Intelligence</h1>
        <p>
            Explore financial concepts using retrieval-augmented generation.
            Get contextual answers grounded in a curated financial knowledge base.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_retriever():
    docs_path = INDEX_DIR / "docs.jsonl"
    embeddings_path = INDEX_DIR / "embeddings.npy"

    if not docs_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Curated RAG index not found at {INDEX_DIR}"
        )

    docs = []

    with docs_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in docs.jsonl at line {line_number}."
                ) from exc

    embeddings = np.asarray(
        np.load(embeddings_path),
        dtype=np.float32,
    )

    if len(docs) != len(embeddings):
        raise ValueError(
            "Document and embedding counts do not match. Rebuild the index."
        )

    if not docs:
        raise ValueError("The curated knowledge base is empty.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    model = SentenceTransformer(EMBEDDING_MODEL)

    return docs, embeddings, model


def retrieve(question, docs, embeddings, model, top_k):
    query_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_vector
    indices = np.argsort(-scores)[:min(top_k, len(docs))]

    candidates = []

    for idx in indices:
        score = float(scores[idx])

        if score < MIN_SIMILARITY:
            continue

        doc = docs[int(idx)]
        text = doc.get("text")

        if not text:
            instruction = doc.get("instruction", "")
            output = doc.get("output", "")
            text = f"Question: {instruction}\nAnswer: {output}"

        if text and text.strip():
            candidates.append(
                {
                    "score": score,
                    "text": text.strip(),
                }
            )

    if not candidates:
        return []

    best_score = candidates[0]["score"]
    relevance_floor = max(
        MIN_SIMILARITY,
        best_score - MAX_SCORE_GAP,
    )

    results = [
        item
        for item in candidates
        if item["score"] >= relevance_floor
    ]

    return results


def clean_answer(answer):
    answer = (answer or "").strip()
    answer = answer.replace("\r\n", "\n").replace("\r", "\n")

    answer = re.sub(
        r"```(?:markdown|md|text)?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = answer.replace("```", "")

    answer = re.sub(
        r"(?im)^\s*(?:\*\*)?FINRAG RESPONSE(?:\*\*)?\s*$",
        "",
        answer,
    )

    answer = re.sub(
        r"(?<!\w)\[\d+\](?:\s*\[\d+\])*",
        "",
        answer,
    )

    answer = re.sub(
        r"(?m)^[ \t]*\[\d+\][ \t]*$",
        "",
        answer,
    )

    lines = answer.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            continue

        if re.fullmatch(
            r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?",
            stripped,
        ):
            continue

        cleaned_lines.append(line.rstrip().rstrip("\\"))

    answer = "\n".join(cleaned_lines).strip()
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer


def generate_answer(question, results):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your local .env file "
            "or Streamlit Cloud secrets."
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
                    "Answer using only the supplied retrieved context. "
                    "Do not invent unsupported facts, figures, examples, or formulas. "
                    "If the context is insufficient, state that clearly. "
                    "Use concise paragraphs and short bullet lists. "
                    "Do not use Markdown tables, HTML tables, LaTeX, math blocks, "
                    "or code fences. "
                    "For comparison questions, use bullet points. "
                    "Write currency amounts using INR followed by the amount, "
                    "such as INR 100,000. Never use currency symbols. "
                    "Do not include citation markers or numbered source references "
                    "in the final answer. "
                    "Do not output backslashes, broken table fragments, duplicate "
                    "answers, or unfinished text. "
                    "Write formulas in plain text using /, *, +, and =. "
                    "Do not provide personalized financial advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Return one complete, clean answer. "
                    "Use only supported information. "
                    "Do not repeat the question as a heading or append extra text."
                ),
            },
        ],
        max_tokens=700,
    )

    return clean_answer(response.choices[0].message.content)


with st.sidebar:
    st.markdown(
        """
        <div style="
            background: linear-gradient(145deg, #10243A, #245B79);
            padding: 19px 17px;
            border-radius: 12px;
            margin: 4px 0 20px 0;
        ">
            <div style="
                color: #C7E9E8;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1.5px;
                margin-bottom: 7px;
            ">FINRAG WORKSPACE</div>
            <div style="
                color: #FFFFFF;
                font-size: 19px;
                font-weight: 800;
            ">Search settings</div>
            <div style="
                color: #E1EAF2;
                font-size: 12px;
                margin-top: 5px;
            ">Configure knowledge retrieval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_k = st.slider(
        "Retrieved documents",
        min_value=1,
        max_value=5,
        value=5,
        help="Choose how many relevant documents to retrieve.",
    )

    st.caption("Documents are filtered using similarity scores.")

    st.divider()

    st.markdown("**How it works**")
    st.markdown(
        """
        1. Enter a financial question.
        2. Retrieve relevant documents.
        3. Generate a grounded answer.
        """
    )

    st.divider()

    st.caption("LLM · openai/gpt-oss-20b")
    st.caption("Embeddings · all-MiniLM-L6-v2")


try:
    docs, embeddings, embedding_model = load_retriever()

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-dot"></div>
            <div>
                <div class="status-title">Knowledge base ready</div>
                <div class="status-detail">
                    {len(docs)} curated documents available for retrieval
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

except Exception as exc:
    st.error(f"Could not load the knowledge base: {exc}")
    st.stop()


st.markdown(
    """
    <div class="section-heading">Ask your question</div>
    <div class="section-subtitle">
        Explore financial terms, ratios, profitability and interest concepts.
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.text_area(
    "Financial question",
    placeholder="Example: What is the difference between revenue and profit?",
    height=110,
    key="finrag_question_input",
)

generate_clicked = st.button(
    "Generate answer  →",
    type="primary",
    use_container_width=True,
)

if generate_clicked:
    if not question.strip():
        st.warning("Please enter a financial question.")
    else:
        st.session_state.finrag_answer = ""
        st.session_state.finrag_sources = []
        st.session_state.finrag_submitted_question = question.strip()

        try:
            with st.spinner(
                "Retrieving context and generating your answer..."
            ):
                results = retrieve(
                    question.strip(),
                    docs,
                    embeddings,
                    embedding_model,
                    top_k,
                )

                st.session_state.finrag_sources = results

                if results:
                    answer = generate_answer(
                        question.strip(),
                        results,
                    )
                else:
                    answer = (
                        "I could not find sufficiently relevant information "
                        "in the knowledge base to answer this question. "
                        "Please try a question about a financial concept "
                        "covered by the available documents."
                    )

                st.session_state.finrag_answer = answer

        except Exception as exc:
            st.error(f"Request failed: {exc}")


if st.session_state.get("finrag_answer"):
    st.markdown(
        """
        <div class="section-heading">Generated answer</div>
        <div class="section-subtitle">
            Context-grounded response from your financial knowledge base
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="answer-label">FINANCIAL EXPLANATION</div>',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.finrag_answer)

    with st.expander("View retrieved sources and similarity scores"):
        sources = st.session_state.get("finrag_sources", [])

        if not sources:
            st.caption("No relevant sources were retrieved.")
        else:
            for i, item in enumerate(sources, start=1):
                safe_text = html.escape(item["text"]).replace("\n", "<br>")

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-heading">
                            SOURCE {i} &nbsp;·&nbsp;
                            Similarity: {item["score"]:.4f}
                        </div>
                        <div class="source-content">{safe_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown(
    """
    <div class="footer">
        <strong>FinRAG Intelligence</strong> · Retrieval-Augmented Generation
        for Financial Education<br>
        Educational demonstration only. Answers depend on the available
        knowledge base and may be incomplete. Verify important financial
        information with reliable sources. This tool does not provide
        personalized financial advice.
    </div>
    """,
    unsafe_allow_html=True,
)
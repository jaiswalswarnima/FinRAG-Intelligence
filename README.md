# FinRAG Intelligence

**A Financial Question-Answering System Powered by Retrieval-Augmented Generation (RAG)**

FinRAG Intelligence is a finance-focused question-answering application that combines semantic search over a curated financial knowledge base with large language model (LLM) inference to generate context-grounded responses.

## Live Demo

**[Launch FinRAG Intelligence](https://finrag-intelligence-j7zntwukvbk9vjay7oqde9.streamlit.app/)**

**GitHub Repository:** [FinRAG-Intelligence](https://github.com/jaiswalswarnima/FinRAG-Intelligence)

## Overview

Financial question answering requires relevant evidence and careful handling of financial terminology. FinRAG Intelligence uses a retrieval-augmented generation pipeline to find relevant financial knowledge before generating an answer.

The application is intended for financial education and general financial concept questions. It does not provide personalized financial advice.

## Key Features

- Financial question answering using a curated Q&A knowledge base
- Dense semantic retrieval using Sentence Transformers
- Vector similarity scoring for retrieved documents
- Configurable top-k document retrieval
- Similarity-based filtering to reduce irrelevant context
- LLM inference through the Groq API
- Interactive web interface built with Streamlit
- Precomputed embeddings for the indexed knowledge base

## Architecture

1. **User Query:** The user submits a financial question through the Streamlit interface.
2. **Embedding:** The query is encoded using `sentence-transformers/all-MiniLM-L6-v2`.
3. **Retrieval:** The system compares the query embedding with indexed document embeddings and retrieves relevant context.
4. **Filtering:** Similarity-based thresholds help filter low-relevance results.
5. **Generation:** Retrieved context is passed to a Groq-hosted language model to generate a response.
6. **Response:** The application displays the generated answer and retrieved-source information.

## Technology Stack

- **Language:** Python
- **Application:** Streamlit
- **Retrieval:** Sentence Transformers, NumPy
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Inference:** Groq API
- **Generation Model:** `openai/gpt-oss-20b`
- **Environment Configuration:** python-dotenv
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## Knowledge Base

The current curated dataset contains 21 financial question-answer records. Topics include financial ratios, profitability concepts, and simple and compound interest.

The indexed knowledge base is stored under `data/rag_index_curated/`, with document records, vector embeddings, and index metadata.

## Project Structure

```text
FinRAG-Intelligence/
├── app.py
├── data/
│   ├── curated_finance_qa.jsonl
│   └── rag_index_curated/
│       ├── docs.jsonl
│       ├── embeddings.npy
│       └── meta.json
├── src/
│   └── build_rag_index.py
├── requirements.txt
└── README.md
```

## Setup and Local Execution

### 1. Clone the repository

```bash
git clone https://github.com/jaiswalswarnima/FinRAG-Intelligence.git
cd FinRAG-Intelligence
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Never commit `.env` or expose API credentials publicly.

### 5. Run the application

```bash
python -m streamlit run app.py
```

## Current Limitations

- Response quality depends on the relevance and coverage of the curated knowledge base.
- Similarity scores do not guarantee factual correctness.
- LLM-generated answers may contain errors and should be independently verified when accuracy is important.
- The application is designed for educational use, not personalized financial advice.
- Comprehensive, independently reproducible retrieval and answer-quality benchmarks remain future work.

## Future Improvements

- Expand and diversify the curated financial knowledge base.
- Evaluate retrieval using Precision@k and Recall@k.
- Evaluate answer relevance, factual grounding, and numerical correctness.
- Explore hybrid retrieval and reranking.
- Add stronger source traceability and deterministic financial calculations.
- Improve automated testing and reproducibility.

## Acknowledgements

This project draws inspiration from the open-source implementation by [Abhij248](https://github.com/Abhij248). Third-party code, datasets, and pretrained models remain subject to their respective licenses and attribution requirements.

**Disclaimer:** FinRAG Intelligence is an educational project and does not provide personalized financial, investment, or tax advice.
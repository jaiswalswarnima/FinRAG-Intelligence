# Architecture

## 1) High-Level System
Final deployed system is a **hybrid Finance QA pipeline: QLoRA fine-tuning + RAG-lite grounding + strict numeric prompt shaping**.

```text
User Question
   |
   v
Style Router (numeric_short / explanatory / mixed)
   |
   v
Dense Retriever (MiniLM embeddings over local finance index)
   |
   +--> top-k evidence chunks
   |
   v
Prompt Builder (style constraints + retrieved context + numeric contract)
   |
   v
Fine-tuned Generator (Qwen2.5-1.5B + QLoRA adapter)
   |
   v
Final Answer + Reason
```

---

## 2) Core Components

### A. Base and Fine-tuned Model
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning: QLoRA adapters
- Final adapter path: `checkpoints/best_model`

### B. Style Router
- Script: `src/style_routing.py`
- Styles:
  - `numeric_short`
  - `explanatory`
  - `mixed`

### C. Retrieval Layer (RAG-lite)
- Index builder: `src/build_rag_index.py`
- Inference: `src/rag_infer.py`
- Index path: `data/rag_index`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

### D. Prompt Builder
Prompt includes:
- style constraints
- retrieved context
- **strict numeric output contract (final official config)**

### E. Evaluation Layer
- Base vs Fine-tuned: `src/evaluate_models.py`
- Financial quality: `src/evaluate_financial_quality.py`
- RAG evaluation: `src/evaluate_rag.py`
- Human eval tooling:
  - `src/build_human_eval_sheet.py`
  - `src/score_human_eval.py`

---

## 3) Data Flow

### Training Flow
```text
Finance datasets -> formatting -> JSONL -> QLoRA fine-tuning -> adapter checkpoints
```

### RAG Index Flow
```text
Training corpus JSONL -> MiniLM embeddings -> vectors + metadata -> local retrieval index
```

### Inference Flow (Final Official)
```text
Question -> style detect -> retrieve top-k=2 -> build grounded + strict numeric prompt -> generate answer
```

---

## 4) Performance Snapshot (FinanceBench-150)

| System | ROUGE-L | BLEU | Semantic Similarity | Numeric Accuracy | Hallucination Proxy |
|---|---:|---:|---:|---:|---:|
| Base | 0.0857 | 0.0170 | 0.4206 | 0.1508 (19/126) | 0.4200 |
| Fine-tuned | 0.1102 | 0.0436 | 0.4509 | 0.0873 (11/126) | 0.2467 |
| RAG Baseline Recheck | 0.3844 | 0.1921 | 0.5594 | 0.5556 (70/126) | 0.3000 |
| **RAG Strict-Numeric-Prompt (Final)** | **0.4093** | **0.2113** | **0.5852** | **0.5794 (73/126)** | **0.2800** |

Human eval (fixed re-rubric, 150):
- Baseline RAG: `43.33%` correct
- **Strict-Numeric-Prompt RAG: `48.67%` correct**

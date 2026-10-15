# Ubitq Finance QA Project Report

Lightweight finance QA system using **QLoRA + RAG-lite**.
Fixed-rubric human evaluation on FinanceBench-150: **48.67%** correct (73/150).
Performance is comparable to published GPT-4 single-retrieval FinanceBench baselines (~50%).

## 1. Objective
Build a finance-domain QA assistant using a small open-source model and show measurable improvement across:
- Base model
- Fine-tuned model (QLoRA)
- Fine-tuned + RAG system

Primary goal: improve factual and numeric reliability for financial questions.

---

## 2. Model Selection
- Model: `Qwen/Qwen2.5-1.5B-Instruct`

Why this model:
- Fits low-memory hardware (RTX 3050, 4GB VRAM)
- Strong instruction-following baseline
- Good quality-to-compute trade-off

Trade-offs:
- Lower reasoning capacity than larger models
- Requires retrieval grounding for stronger factual precision

---

## 3. System Setup
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: QLoRA (4-bit)
- Retrieval: local dense index (`MiniLM` embeddings)
- Final official serving config:
  - `top_k=2`
  - `style_mode=auto`
  - `strict_numeric_prompt=true`
  - `numeric_guardrail=false`

Source of truth: `evaluation/submission_manifest_2026-04-09.json`

---

## 4. Dataset Design
Training data used across iterations:
- `data/train.jsonl` / `data/val.jsonl`
- `data/train_v3.jsonl` / `data/val_v3.jsonl`
- numeric-focused subsets in `data/train_v4_numeric*.jsonl`

Benchmark test set:
- `data/test.jsonl` (FinanceBench-150)

Processing highlights:
- Instruction format (`instruction`, `input`, `output`)
- Structured outputs in later runs (`Final Answer`, `Reason`)
- Iterative cleaning/filtering and subset construction

---

## 5. Training Pipeline
- Tokenization: HuggingFace tokenizer (Qwen)
- QLoRA with 4-bit quantization
- Typical run ranges:
  - LR: `5e-5` to `2e-4`
  - Steps: ~60 to ~220
  - Max sequence length: up to 448

---

## 6. Official Results (FinanceBench-150)

| System | ROUGE-L | BLEU | Semantic Similarity | Numeric Accuracy | Hallucination Proxy |
|---|---:|---:|---:|---:|---:|
| Base                                  | 0.0857 | 0.0170 | 0.4206 | 0.1508 (19/126) | 0.4200 |
| Fine-tuned                            | 0.1102 | 0.0436 | 0.4509 | 0.0873 (11/126) | 0.2467 |
| RAG Baseline Recheck                  | 0.3844 | 0.1921 | 0.5594 | 0.5556 (70/126) | 0.3000 |
| **RAG Strict-Numeric-Prompt (Final)** | 0.4093 | 0.2113 | 0.5852 | 0.5794 (73/126) | 0.2800 |

Human eval (fixed re-rubric, n=150):
- Baseline RAG: `65/150 (43.33%)` correct
- **Final strict-prompt RAG: `73/150 (48.67%)` correct**
- Improvement: `+5.34` points

### 6.1 Isolated Impact of Fine-Tuning (No RAG)

| System (No RAG) | ROUGE-L | BLEU | Semantic Similarity | Numeric Accuracy | Hallucination Proxy |
|---|---:|---:|---:|---:|---:|
| Base       | 0.0857 | 0.0170 | 0.4206 | 0.1508 (19/126) | 0.4200 |
| Fine-tuned | 0.1102 | 0.0436 | 0.4509 | 0.0873 (11/126) | 0.2467 |

Key interpretation:
- Fine-tuning improved semantic quality and output alignment.
- Numeric reliability remained a weakness without retrieval grounding.
- RAG was required to convert semantic gains into reliable factual/numeric performance.
- This behavior indicates that fine-tuning improved linguistic alignment but did not encode stable numeric grounding on its own.

---

## 7. Iteration Journey
- Base -> Fine-tuned: semantic gains, numeric weakness remained.
- Fine-tuned -> RAG: strong factual and numeric gains.
- RAG variants:
  - Strict numeric prompt: improved automated + human metrics.
  - Numeric guardrail retry variants: rejected due to regressions.

### 7.1 Why This Final System Works
- QLoRA fine-tuning improved response structure and finance-domain answer style.
- RAG provided evidence-grounded context at inference time.
- Strict numeric prompting reduced number-format drift for numeric queries.

System-level insight:
- For financial QA, factual reliability is primarily retrieval-driven; fine-tuning alone is not enough.

### 7.2 Decision Log (Important Engineering Trade-Off)
Tested and rejected:
- Numeric guardrail-only regeneration (`numeric_guardrail=true`) due to large semantic degradation and higher hallucination proxy.
- Strict+guardrail combined variant due to regression versus strict-only.

Selected:
- Strict numeric prompt without guardrail (`strict_numeric_prompt=true`, `numeric_guardrail=false`) as final deployment mode.

---

## 8. Failure Analysis
Main residual error taxonomy:

| Failure Type | Typical Symptom | Root Cause |
|---|---|---|
| Format error      | Missing parseable numeric value in final answer | Output contract drift |
| Wrong number/unit | Correct concept but wrong figure/unit scale     | Unit normalization/precision issue |
| Reasoning mismatch| Correct evidence available but wrong conclusion | Inference error |
| Retrieval miss    | Retrieved context lacks decisive evidence       | Retriever/chunk limitation |

This taxonomy motivated a prompt-level numeric tightening strategy and informed rejection of over-aggressive regeneration.

---

## 9. Final Recommendation
Deploy:
- model: `checkpoints/best_model`
- retrieval index: `data/rag_index`
- inference: `src/rag_infer.py` with `--strict-numeric-prompt`

This provides best observed balance of:
- semantic quality
- numeric correctness
- reduced hallucination proxy

---

## 10. Limitations & Future Work

Current limitations:
- The final system remains below ideal accuracy for production-grade financial decision support.
- Numeric and unit precision errors still appear on difficult ratio and multi-step computation questions.
- Response quality is sensitive to retrieval quality and embedding similarity.
- The compact 1.5B model has reasoning limits on longer multi-hop financial prompts.

Future work:
- Add DPO/RLHF-style preference alignment for better faithfulness and answer completeness.
- Improve retriever quality (hybrid dense+sparse retrieval and better chunking/reranking).
- Add calculator/tool-assisted reasoning for multi-step numeric questions.
- Optimize inference latency with faster serving backends and caching for repeated queries.

## 11. Reproducibility Artifacts
Primary:
- `evaluation/submission_manifest_2026-04-09.json`
- `evaluation/results_rag_full_baseline_recheck.json`
- `evaluation/results_rag_full_strict_only.json`
- `evaluation/human_eval_summary_rag_full_baseline_rerubric.json`
- `evaluation/human_eval_summary_rag_full_strict_only_rerubric.json`
- Hugging Face model artifacts (adapter): `https://huggingface.co/geek52/uptiq-finance-qa-qlora-rag`
- Hugging Face RAG index dataset: `https://huggingface.co/datasets/geek52/uptiq-finance-qa-rag-index`
- Base model reference (not re-uploaded): `Qwen/Qwen2.5-1.5B-Instruct`

Core scripts:
- `src/train_qlora.py`
- `src/evaluate_models.py`
- `src/evaluate_financial_quality.py`
- `src/evaluate_rag.py`
- `src/build_rag_index.py`
- `src/rag_infer.py`

---

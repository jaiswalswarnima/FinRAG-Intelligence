import argparse
import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def to_doc_text(row: dict) -> str:
    q = (row.get("instruction") or "").strip()
    a = (row.get("output") or "").strip()
    return f"Question: {q}\nAnswer: {a}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local dense retrieval index for RAG.")
    parser.add_argument("--corpus", default="data/train_v3.jsonl")
    parser.add_argument("--out-dir", default="data/rag_index")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    embedder = SentenceTransformer(args.embed_model, local_files_only=offline)

    rows = load_rows(args.corpus)
    docs = []
    seen = set()
    for i, row in enumerate(rows):
        key = (norm_text(row.get("instruction", "")), norm_text(row.get("output", "")))
        if key in seen:
            continue
        seen.add(key)
        docs.append(
            {
                "id": len(docs),
                "instruction": row.get("instruction", ""),
                "output": row.get("output", ""),
                "text": to_doc_text(row),
            }
        )

    texts = [d["text"] for d in docs]
    vectors = embedder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", vectors.astype(np.float32))
    with open(out_dir / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "corpus": args.corpus,
                "embed_model": args.embed_model,
                "num_docs": len(docs),
                "dim": int(vectors.shape[1]) if len(vectors) else 0,
            },
            f,
            indent=2,
        )

    print(f"Indexed {len(docs)} unique docs -> {out_dir}")


if __name__ == "__main__":
    main()

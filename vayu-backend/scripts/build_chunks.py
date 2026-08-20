"""Build dataset chunks — VĀYU "vast" chunking strategy.

Strategy (requirement #2: no naive fixed-size chunking):
1. PARENT-CHILD hierarchy   — full passages are parents; granular 1–2
   sentence groups are the indexed children (precision at retrieval,
   context continuity at answer time).
2. OVERLAP handling        — children slide with a 1-sentence overlap so
   sentence boundaries never orphan a concept mid-chunk.
3. METADATA-AWARE          — each child is enriched with Language, Category,
   Title, and Parent ID so retrieval can be filtered/faceted and the LLM
   receives provenance.
4. SEMANTIC boundaries     — split on sentence boundaries (regex), never
   mid-sentence, with fixed-size grouping of N sentences (hybrid of
   semantic + fixed-size).

Dataset support:
- PRIMARY: ai4bharat/MSMARCO-XI (the task's provided dataset) — schema has
  `query_id`, `query`, `passages` (list of {passage_text, pid, is_selected}),
  `answers`, `language`.
- FALLBACK: rajpurkar/squad (context/title) for quick offline testing.
"""
import os
import json
import re
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def chunk_text_vast(text, chunk_sentences=2, overlap_sentences=1):
    """Sentence-boundary splitting with sliding overlap."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return []
    step = max(1, chunk_sentences - overlap_sentences)
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + chunk_sentences])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def add_document(parents, child_chunks, doc_id, text, lang, category, title=""):
    """Index one parent passage + its overlapping, metadata-enriched children."""
    text = (text or "").strip()
    if not text:
        return
    parents[doc_id] = {
        "id": doc_id,
        "text": text,
        "metadata": {"lang": lang, "category": category, "title": title},
    }
    for i, sentence in enumerate(chunk_text_vast(text)):
        child_id = f"{doc_id}_{i}"
        enriched = (
            f"Language: {lang} | Category: {category} | ID: {doc_id}"
            f"{(' | Title: ' + title) if title else ''} | Content: {sentence}"
        )
        child_chunks.append({
            "child_id": child_id,
            "parent_id": doc_id,
            "text": enriched,
        })


def build_msmarco_xi(max_docs=3000):
    """Primary dataset: ai4bharat/MSMARCO-XI (multilingual, Hindi-first)."""
    print("Loading ai4bharat/MSMARCO-XI (streaming)...")
    dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train", streaming=True)
    parents, child_chunks = {}, []
    count = 0
    for row in dataset:
        if count >= max_docs:
            break
        lang = row.get("language", "en")
        passages = row.get("passages") or []
        for p in passages:
            if isinstance(p, dict):
                pid = str(p.get("pid", f"{count}_passage"))
                text = p.get("passage_text", "")
                selected = bool(p.get("is_selected", False))
                doc_id = f"{lang}_{pid}"
                # selected passages are the true answers -> boost category
                category = "msmarco-answer" if selected else "msmarco-context"
                add_document(
                    parents, child_chunks, doc_id, text, lang, category,
                    title=row.get("query", "")[:80],
                )
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} queries...")
    print(f"MSMARCO-XI: {len(parents)} passages, {len(child_chunks)} child chunks")
    return parents, child_chunks


def build_squad(max_docs=3000):
    """Offline fallback: SQuAD for quick local testing."""
    print("Loading rajpurkar/squad (streaming)...")
    dataset = load_dataset("rajpurkar/squad", split="train", streaming=True)
    parents, child_chunks = {}, []
    count = 0
    for row in dataset:
        if count >= max_docs:
            break
        add_document(
            parents, child_chunks,
            doc_id=str(row.get("id", count)),
            text=row.get("context", ""),
            lang="en",
            category="general",
            title=row.get("title", ""),
        )
        count += 1
        if count % 500 == 0:
            print(f"  Processed {count} documents...")
    print(f"SQuAD: {len(parents)} passages, {len(child_chunks)} child chunks")
    return parents, child_chunks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset_name = os.getenv("VAYU_DATASET", "msmarco-xi").lower()
    if dataset_name in ("squad", "fallback"):
        parents, child_chunks = build_squad()
    else:
        try:
            parents, child_chunks = build_msmarco_xi()
        except Exception as exc:
            print(f"MSMARCO-XI failed ({exc}); falling back to SQuAD...")
            parents, child_chunks = build_squad()

    with open(os.path.join(DATA_DIR, "parents.json"), "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False)
    with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(child_chunks, f, ensure_ascii=False)

    print(f"Exported {len(parents)} parents and {len(child_chunks)} children to {DATA_DIR}/")
    print("Next: python scripts/build_index.py")


if __name__ == "__main__":
    main()

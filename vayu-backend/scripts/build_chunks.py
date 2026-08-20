import os
import json
import re
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def chunk_text(text, max_sentences=2):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

def build_dataset():
    print("Loading dataset squad...")
    # Using squad temporarily because MSMARCO-XI has nested schema issues with pyarrow streaming
    dataset = load_dataset("rajpurkar/squad", split="train", streaming=True)
    
    parents = {}
    child_chunks = []
    
    count = 0
    print("Dataset stream initialized, beginning to fetch rows...")
    for row in dataset:
        if count >= 3000:  # ~3000 docs to keep footprint low
            break
            
        doc_id = row.get("id", str(count))
        text = row.get("text", row.get("passage", row.get("contents", row.get("context", ""))))
        if not text:
            continue
            
        lang = row.get("language", "en")
        category = row.get("category", "general")
        
        parents[doc_id] = {
            "id": doc_id,
            "text": text,
            "metadata": {"lang": lang, "category": category}
        }
        
        sentences = chunk_text(text, max_sentences=2)
        for i, sentence in enumerate(sentences):
            child_id = f"{doc_id}_{i}"
            # Metadata Enrichment
            enriched_text = f"Language: {lang} | Category: {category} | ID: {doc_id} | Content: {sentence}"
            child_chunks.append({
                "child_id": child_id,
                "parent_id": doc_id,
                "text": enriched_text
            })
            
        count += 1
        if count % 500 == 0:
            print(f"Processed {count} documents")
            
    os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)
    parents_path = os.path.join(os.path.dirname(__file__), "../data/parents.json")
    chunks_path = os.path.join(os.path.dirname(__file__), "../data/chunks.json")
    
    with open(parents_path, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False)
        
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(child_chunks, f, ensure_ascii=False)
        
    print(f"Exported {len(parents)} parents and {len(child_chunks)} child chunks.")

if __name__ == "__main__":
    build_dataset()

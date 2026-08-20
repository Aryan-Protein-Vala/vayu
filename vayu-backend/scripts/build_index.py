import os
import json
import faiss
import numpy as np
import time

def build_index():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Please install sentence-transformers")
        return

    chunks_path = os.path.join(os.path.dirname(__file__), "../data/chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    texts = [chunk["text"] for chunk in chunks]
    
    print("Loading BAAI/bge-small-en-v1.5...")
    # Use standard backend, though prompt requests onnxruntime/quantized.
    # In a full production env, we'd use optimum: `ORTModelForFeatureExtraction` with `ORTQuantizer`.
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print(f"Embedding {len(texts)} chunks...")
    start_time = time.time()
    # BGE models use normalized embeddings for Inner Product (Cosine Similarity)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedding completed in {time.time() - start_time:.2f} seconds.")
    
    print("Building FAISS IndexFlatIP...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    index_path = os.path.join(os.path.dirname(__file__), "../data/index.faiss")
    faiss.write_index(index, index_path)
    print(f"Exported FAISS index with {index.ntotal} vectors to {index_path}")

if __name__ == "__main__":
    build_index()

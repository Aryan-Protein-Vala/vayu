"""Build FAISS index using TF-IDF embeddings (no HF download needed).
Use sklearn TfidfVectorizer as a lightweight local embedding substitute.
When deployed where HF is reachable, swap to BAAI/bge-small-en-v1.5."""
import os, json, faiss, numpy as np, time, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def build_index():
    chunks_path = os.path.join(DATA_DIR, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Loaded {len(texts)} chunk texts")

    print("Computing TF-IDF embeddings...")
    start = time.time()
    vectorizer = TfidfVectorizer(
        max_features=512,
        stop_words='english',
        norm='l2',
        sublinear_tf=True
    )
    embeddings = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    print(f"Embedding done in {time.time() - start:.2f}s, shape={embeddings.shape}")

    print("Building FAISS IndexFlatIP...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(DATA_DIR, "index.faiss")
    vec_path = os.path.join(DATA_DIR, "vectorizer.pkl")

    faiss.write_index(index, index_path)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Exported FAISS index ({index.ntotal} vectors) to {index_path}")
    print(f"Exported vectorizer to {vec_path}")


if __name__ == "__main__":
    build_index()
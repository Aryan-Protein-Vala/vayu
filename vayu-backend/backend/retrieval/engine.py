import os
import json
import faiss
import numpy as np
import pickle
import time
from functools import lru_cache
import asyncio


class RetrievalEngine:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), "../../data")
        self.index_path = os.path.join(self.data_dir, "index.faiss")
        self.parents_path = os.path.join(self.data_dir, "parents.json")
        self.chunks_path = os.path.join(self.data_dir, "chunks.json")
        self.vec_path = os.path.join(self.data_dir, "vectorizer.pkl")

        print("Loading FAISS index into RAM...")
        self.index = faiss.read_index(self.index_path)

        print("Loading parents.json into RAM...")
        with open(self.parents_path, "r", encoding="utf-8") as f:
            self.parents = json.load(f)

        print("Loading chunks.json into RAM...")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("Loading vectorizer...")
        with open(self.vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        print(
            f"Engine ready: {len(self.parents)} parents, "
            f"{len(self.chunks)} chunks, {self.index.ntotal} vectors"
        )

    def _embed(self, texts):
        """Convert text(s) to L2-normalized TF-IDF vectors."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.vectorizer.transform(texts).toarray().astype(np.float32)
        # TfidfVectorizer(norm='l2') already normalizes; keep float32 for FAISS
        return vecs

    def _search_impl(self, query: str, top_k: int = 3):
        """Raw search: embed -> FAISS -> O(1) parent resolution."""
        query_emb = self._embed(query)

        distances, indices = self.index.search(query_emb, top_k)

        results = []
        seen_parents = set()
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            child = self.chunks[idx]
            parent_id = child["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_data = self.parents[parent_id]
                results.append({
                    "parent_id": parent_id,
                    "text": parent_data["text"],
                    "score": float(dist),
                    "metadata": parent_data.get("metadata", {})
                })
        return results

    @lru_cache(maxsize=2048)
    def _search_sync(self, query: str, top_k: int = 3):
        """Cached search for production hot-path (<0.05ms on cache hit)."""
        return self._search_impl(query, top_k)

    def search_timed(self, query: str, top_k: int = 3):
        """Uncached search returning (results, stage_timings_ms) — used by the
        latency benchmark so every reported number is a REAL measurement."""
        t0 = time.perf_counter_ns()
        query_emb = self._embed(query)
        t1 = time.perf_counter_ns()
        distances, indices = self.index.search(query_emb, top_k)
        t2 = time.perf_counter_ns()

        results = []
        seen_parents = set()
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            child = self.chunks[idx]
            parent_id = child["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_data = self.parents[parent_id]
                results.append({
                    "parent_id": parent_id,
                    "text": parent_data["text"],
                    "score": float(dist),
                    "metadata": parent_data.get("metadata", {})
                })
        t3 = time.perf_counter_ns()

        timings = {
            "embed_ms": (t1 - t0) / 1e6,
            "faiss_ms": (t2 - t1) / 1e6,
            "parent_ms": (t3 - t2) / 1e6,
        }
        return results, timings

    async def search_async(self, query: str, top_k: int = 3):
        """Cached async search used by the app hot path."""
        return await asyncio.to_thread(self._search_sync, query, top_k)

    async def search_timed_async(self, query: str, top_k: int = 3):
        """Uncached async search with per-stage timing (benchmarks only)."""
        return await asyncio.to_thread(self.search_timed, query, top_k)


engine = None


def get_engine():
    global engine
    if engine is None:
        engine = RetrievalEngine()
    return engine

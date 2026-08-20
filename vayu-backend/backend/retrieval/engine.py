import os
import json
import faiss
import numpy as np
from functools import lru_cache
import asyncio
import time

class RetrievalEngine:
    def __init__(self):
        self.index_path = os.path.join(os.path.dirname(__file__), "../../data/index.faiss")
        self.parents_path = os.path.join(os.path.dirname(__file__), "../../data/parents.json")
        self.chunks_path = os.path.join(os.path.dirname(__file__), "../../data/chunks.json")
        
        print("Loading FAISS index into RAM...")
        self.index = faiss.read_index(self.index_path)
        
        print("Loading parents.json into RAM...")
        with open(self.parents_path, "r", encoding="utf-8") as f:
            self.parents = json.load(f)
            
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
            
        print("Loading Embedding Model...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except ImportError:
            self.model = None
            print("Warning: sentence_transformers not installed")

    # In-memory LRU Cache: < 0.2ms resolution for repeated queries
    @lru_cache(maxsize=2048)
    def _search_sync(self, query: str, top_k: int = 3):
        if not self.model:
            return []
            
        # 1. Embed Query
        query_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        
        # 2. FAISS RAM Search
        distances, indices = self.index.search(query_emb, top_k)
        
        # 3. Resolve Parent Texts O(1)
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
                    "metadata": parent_data["metadata"]
                })
                
        return results

    async def search_async(self, query: str, top_k: int = 3):
        # Run CPU-bound LRU cache / FAISS search in threadpool
        return await asyncio.to_thread(self._search_sync, query, top_k)

engine = None
def get_engine():
    global engine
    if engine is None:
        engine = RetrievalEngine()
    return engine

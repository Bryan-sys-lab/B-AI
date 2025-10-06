import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, index_file='vector_store/faiss_index.idx', metadata_file='vector_store/metadata.json', model_name='all-MiniLM-L6-v2'):
        self.index_file = index_file
        self.metadata_file = metadata_file
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # Inner product for cosine similarity
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> int:
        id = len(self.metadata)
        self.index.add(np.array([vector], dtype=np.float32))
        self.metadata[str(id)] = metadata
        self.save()
        return id

    def add_text(self, text: str, metadata: Dict[str, Any]) -> int:
        # Store the text in metadata for retrieval
        metadata = metadata.copy()
        metadata["text"] = text
        vector = self.embed_text(text)
        return self.add_vector(vector, metadata)

    def add_document(self, doc: Dict[str, Any]) -> int:
        text = doc.get("content", "")
        metadata = doc.get("metadata", {})
        metadata["timestamp"] = doc.get("timestamp")
        return self.add_text(text, metadata)

    def search_vectors(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        query = np.array([query_vector], dtype=np.float32)
        distances, indices = self.index.search(query, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    'id': int(idx),
                    'score': float(dist),
                    'metadata': self.metadata.get(str(idx), {})
                })
        return results

    def search_text(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embed_text(query_text)
        return self.search_vectors(query_vector, k)

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        results = self.search_text(query, limit)
        docs = []
        for r in results:
            meta = r["metadata"]
            docs.append({
                "content": meta.get("text", ""),
                "metadata": meta,
                "score": r["score"]
            })
        return docs

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        return self.metadata

    def delete_vector(self, id: int):
        # FAISS doesn't support deletion easily, so we can mark as deleted in metadata
        if str(id) in self.metadata:
            self.metadata[str(id)]['deleted'] = True
            self.save()

    def rebuild_index(self):
        # To handle deletions, rebuild index without deleted vectors
        new_index = faiss.IndexFlatIP(self.dim)
        new_metadata = {}
        new_id = 0
        for old_id, meta in self.metadata.items():
            if not meta.get('deleted', False):
                # Need to re-embed if we don't store vectors, but since we don't, assume metadata has text
                if 'text' in meta:
                    vector = self.embed_text(meta['text'])
                    new_index.add(np.array([vector], dtype=np.float32))
                    new_metadata[str(new_id)] = meta
                    new_id += 1
        self.index = new_index
        self.metadata = new_metadata
        self.save()
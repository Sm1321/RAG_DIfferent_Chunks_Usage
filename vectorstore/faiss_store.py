import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])

        return results

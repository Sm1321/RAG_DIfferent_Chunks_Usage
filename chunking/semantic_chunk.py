import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

from embedding.openai_embed import get_embedding


def split_into_sentences(text):
    # Basic sentence split (you can improve later)
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunk(text, threshold=0.80):
    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return sentences

    # Get embeddings for all sentences
    embeddings = [get_embedding(sentence) for sentence in sentences]

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        prev_embedding = np.array(embeddings[i - 1]).reshape(1, -1)
        curr_embedding = np.array(embeddings[i]).reshape(1, -1)

        similarity = cosine_similarity(prev_embedding, curr_embedding)[0][0]

        if similarity < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

        current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

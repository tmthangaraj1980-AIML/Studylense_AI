
import numpy as np
import faiss

from functools import lru_cache
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def chunk_document(pages, chunk_size=800, overlap=150):

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = chunk_size - overlap

    for page in pages:

        text = " ".join(page["text"].split())
        page_number = page["page"]

        for start in range(0, len(text), step):

            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if start + chunk_size >= len(text):
                break

    return chunks


def create_vector_index(chunks):

    if not chunks:
        raise ValueError("No document chunks available")

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_index(index, chunks, question, top_k=3):

    if not question.strip():
        raise ValueError("Question cannot be empty")

    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, chunk_id in zip(scores[0], indices[0]):

        if chunk_id == -1:
            continue

        chunk = chunks[int(chunk_id)]

        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results

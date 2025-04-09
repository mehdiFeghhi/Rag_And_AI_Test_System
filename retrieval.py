import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from app_config import CACHE_FILE, NAME_MODEL_EMBEDDING


def load_chunks_and_embeddings(filename: str = CACHE_FILE) -> Tuple[Optional[List[dict]], Optional[dict]]:
    """
    Load text chunks and their embeddings from cache file.

    Args:
        filename (str): Path to the cache file.

    Returns:
        Tuple: A tuple containing a list of chunks and a dictionary of embeddings,
               or (None, None) if the file does not exist.
    """
    if not os.path.exists(filename):
        return None, None

    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["embeddings"]


def split_into_subchunks(text: str, chunk_size: int = 100, overlap: int = 30) -> List[str]:
    """
    Split a text into overlapping subchunks.

    Args:
        text (str): Input text to be split.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words.

    Returns:
        List[str]: A list of subchunk strings.
    """
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]


def find_top_k_chunks(query: str, top_k: int = 3) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        query (str): Input query string.
        top_k (int): Number of top relevant chunks to return.

    Returns:
        List[dict]: List of top relevant chunk dictionaries.

    Raises:
        ValueError: If no chunks or embeddings are loaded.
    """
    if main_chunks is None or embedding_cache is None:
        raise ValueError("No chunks or embeddings found. Please load them first.")

    query_embedding = model.encode(query)
    query_embedding_np = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_embedding_np)

    scored_chunks = []

    for chunk in main_chunks:
        content = chunk["content"]

        if content in embedding_cache:
            subchunks = embedding_cache[content]["subchunks"]
            subchunk_embeddings = embedding_cache[content]["embeddings"]
        else:
            subchunks = split_into_subchunks(content)
            subchunk_embeddings = model.encode(subchunks)
            embedding_cache[content] = {
                "subchunks": subchunks,
                "embeddings": subchunk_embeddings
            }

        subchunk_embeddings_np = np.array(subchunk_embeddings, dtype=np.float32)
        faiss.normalize_L2(subchunk_embeddings_np)

        index = faiss.IndexFlatIP(subchunk_embeddings_np.shape[1])
        index.add(subchunk_embeddings_np)

        scores, _ = index.search(query_embedding_np, 1)
        max_score = scores[0][0]

        scored_chunks.append((max_score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]


# Load model and cached data
model = SentenceTransformer(NAME_MODEL_EMBEDDING)
main_chunks, embedding_cache = load_chunks_and_embeddings()

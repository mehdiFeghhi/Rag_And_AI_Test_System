import os
import json
import pickle
from typing import List, Optional
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer


from app_config import (
    NAME_MODEL_EMBEDDING,
    CACHE_JSON_LOCAL_MODEL,
    CACHE_JSON_GPT_LARGE,
    CACHE_JSON_GPT_SMALL,
    EMBEDER_KEY,

)

# -------------------------------
# Initialization
# -------------------------------

# Initialize OpenAI API client
client = openai.OpenAI(
    api_key= EMBEDER_KEY,
    base_url="https://api.avalai.ir/v1"
)

# Load local embedding model
local_model = SentenceTransformer(NAME_MODEL_EMBEDDING)


# -------------------------------
# Embedding Loading
# -------------------------------

def load_chunks_and_embeddings(model_name: str = "Local") -> dict:
    """
    Load cached chunks and embeddings from a JSON file based on the selected model.
    """
    if model_name == "Local":
        filename = CACHE_JSON_LOCAL_MODEL
    elif model_name == "GPT_large":
        filename = CACHE_JSON_GPT_LARGE
    elif model_name == "GPT_small":
        filename = CACHE_JSON_GPT_SMALL
    else:
        raise ValueError("Invalid model_name. Choose from 'Local', 'GPT_small', or 'GPT_large'.")

    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# Embedding Computation
# -------------------------------

def get_gpt_embedding(texts, name: str):
    """
    Compute embeddings for the given texts using the specified engine.
    """
    if name == "Local":
        embeddings = local_model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()  # ✅ Move to CPU and convert to numpy

    model_map = {
        "GPT_small": "text-embedding-3-small",
        "GPT_large": "text-embedding-3-large"
    }

    if name not in model_map:
        raise ValueError("Invalid name. Choose from 'Local', 'GPT_small', or 'GPT_large'.")

    response = client.embeddings.create(
        input=texts,
        model=model_map[name]
    )
    return [item.embedding for item in response.data]


# -------------------------------
# Semantic Search
# -------------------------------

def find_relevant_subchunks(
    query_embedding: list,
    data: dict,
    top_k: int = 3,
    context_size: int = 2
) -> List[dict]:
    """
    Perform a FAISS-based semantic search over precomputed subchunks.
    """

    # Flatten all subchunks with metadata
    all_subchunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "subchunk_id": sub["subchunk_id"],
            "text": sub["text"],
            "embedding": sub["embedding"]
        }
        for chunk in data["chunks"]
        for sub in chunk["subchunks"]
    ]

    # Build FAISS index with normalized embeddings
    embeddings = np.array([np.array(s["embedding"]).flatten() for s in all_subchunks]).astype('float32')
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Normalize query embedding
    query_embedding = np.array(query_embedding).flatten().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    # Search for top matches
    _, indices = index.search(query_embedding, top_k)
    top_indices = indices[0]

    # Collect relevant text with context
    results = []
    for idx in top_indices:
        start = max(0, idx - context_size)
        end = min(len(all_subchunks), idx + context_size)

        combined_text = ""
        for i in range(start, end + 1):
            sub_text = all_subchunks[i]["text"]
            if i == start:
                combined_text += sub_text
            else:
                words = sub_text.split()
                combined_text += " " + " ".join(words[30:]) if len(words) >= 30 else " " + sub_text

        # Extract a cleaner section from the combined text
        first_period = combined_text.find('.')
        last_period = combined_text.rfind('.')
        if first_period != -1 and last_period != -1 and first_period < last_period:
            combined_text = combined_text[first_period + 1: last_period + 1].strip()

        results.append({"content": combined_text})

    return results


# -------------------------------
# Main Interface
# -------------------------------

def find_top_k_chunks(query: str, top_k: int = 3, embeder_name: str = "GPT_large") -> List[dict]:
    """
    Main entry to search for relevant chunks given a query.
    """
    data = load_chunks_and_embeddings(embeder_name)
    query_embedding = get_gpt_embedding(query, embeder_name)
    return find_relevant_subchunks(query_embedding, data, top_k=top_k)


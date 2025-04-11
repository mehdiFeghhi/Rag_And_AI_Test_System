import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from app_config import  NAME_MODEL_EMBEDDING, CACHE_JSON_LOCAL_MODEL, CACHE_JSON_GPT_LARGE, CACHE_JSON_GPT_SMALL
import openai



def load_chunks_and_embeddings(model_name: str = "Local") -> dict:

    if model_name == "Local":
        filename = CACHE_JSON_LOCAL_MODEL
    elif model_name == "GPT_large":
        filename = CACHE_JSON_GPT_LARGE
    elif model_name == "GPT_small":
        filename = CACHE_JSON_GPT_SMALL
    else:
        raise ValueError("Invalid name parameter. Must be 'Local', 'GPT_small', or 'GPT_large'.")
    
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)



def get_gpt_embedding(texts, name):
    """
    Compute embeddings for the provided texts based on the specified engine 'name'.

    Parameters:
      texts (list or str): Input text or list of texts for embedding.
      name (str): Determines the embedding method:
         - "Local" to use a local model (expects 'local_model' to be defined in the environment).
         - "GPT_small" to use the OpenAI API with the "text-embedding-3-small" model.
         - "GPT_large" to use the OpenAI API with the "text-embedding-3-large" model.

    Returns:
      Embeddings computed either by a local model or from the API response.
      
    Raises:
      ValueError: If 'name' is not one of "Local", "GPT_small", or "GPT_large".
    """
    # Use the local model if the name is "Local"
    if name == "Local":
        # Ensure that 'local_model' is defined in your environment
        return local_model.encode(texts, convert_to_tensor=True)
    
    # Use the OpenAI API for GPT_small or GPT_large
    elif name == "GPT_small":
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    
    elif name == "GPT_large":
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-large"
        )
        return [item.embedding for item in response.data]
    
    # If name is none of the expected values, raise an error
    else:
        raise ValueError("Invalid name parameter. Must be 'Local', 'GPT_small', or 'GPT_large'.")



def find_relevant_subchunks(query_embedding: list, data: dict, top_k: int = 3, context_size: int = 2) -> str:
    # Extract all subchunks
    all_subchunks = []
    for chunk in data["chunks"]:
        for subchunk in chunk["subchunks"]:
            all_subchunks.append({
                "chunk_id": chunk["chunk_id"],
                "subchunk_id": subchunk["subchunk_id"],
                "text": subchunk["text"],
                "embedding": subchunk["embedding"]
            })

    # Prepare embeddings array
    embeddings = np.array([np.array(s["embedding"]).flatten() for s in all_subchunks]).astype('float32')

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
    index.add(embeddings)

    # Prepare query embedding
    query_embedding = np.array(query_embedding).flatten().astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    # Search with FAISS
    distances, indices = index.search(query_embedding, top_k)
    top_indices = indices[0]  # FAISS returns results in descending order of similarity
    print(top_indices)

    # Collect context around top results
    relevant_subchunks = []
    for idx in top_indices:
        start = max(0, idx - context_size)
        end = min(len(all_subchunks), idx + context_size + 1)
        relevant_subchunks.extend(all_subchunks[start:end])

    # Remove duplicates while preserving order
    seen = set()
    unique_subchunks = []
    for s in relevant_subchunks:
        key = (s["chunk_id"], s["subchunk_id"])
        if key not in seen:
            seen.add(key)
            unique_subchunks.append(s)

    # Sort by chunk and subchunk IDs
    unique_subchunks.sort(key=lambda x: (x["chunk_id"], x["subchunk_id"]))
    relevant_text = ""
    for res in unique_subchunks:
        relevant_text += res['text']

    first_period_idx = relevant_text.find('.')
    last_period_idx = relevant_text.rfind('.')
    if first_period_idx != -1 and last_period_idx != -1 and first_period_idx <= last_period_idx:
        relevant_text = relevant_text[first_period_idx: last_period_idx + 1].strip()
    else:
        relevant_text = relevant_text

    return relevant_text


def find_top_k_chunks(query: str, top_k: int = 3,embeder_name:str = "GPT_large") -> List[dict]:

    if embeder_name == "Local":
        data = load_chunks_and_embeddings(CACHE_JSON_LOCAL_MODEL)

    query_embedding = get_gpt_embedding(query)

    relevant_text = find_relevant_subchunks(query_embedding, data,top_k)




# Initialize the OpenAI client once for API-based embeddings.
client = openai.OpenAI(
    api_key="aa-N7QvBs4uwSsEMRWzb1UUCr0jqlSYhknZaKtsNciBRcSGeOgh",
    base_url="https://api.avalai.ir/v1"
)
# Load local_model and cached openaidata
local_model = SentenceTransformer(NAME_MODEL_EMBEDDING)
# main_chunks, embedding_cache = load_chunks_and_embeddings()

from langchain_openai import ChatOpenAI  # pip install -U langchain_openai

from app_config import API_KEY

# Initialize OpenAI with your API key

# Choose your model here: "gpt-4" or "gpt-4-turbo"
DEFAULT_MODEL = "gpt-4o-mini"


def generate_chat_response(user_query: str, retrieved_chunks: list[str], model_name: str = DEFAULT_MODEL) -> str:
    """
    Generates a main response using OpenAI's ChatGPT based on the user query and retrieval context.

    Args:
        user_query (str): The user's query.
        retrieved_chunks (list[str]): Contextual chunks to provide to the model.
        model_name (str): Which model to use (e.g., "gpt-4", "gpt-4-turbo").

    Returns:
        str: The response from ChatGPT.
    """
    system_prompt = (
        "You are an expert AI assistant. Use the provided context to answer the user's query as helpfully and accurately as possible."
    )

    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {user_query}"}
    ]



    try:
        
        llm = ChatOpenAI(
             model=model_name, base_url="https://api.avalai.ir/v1", api_key=API_KEY)

        response = llm.invoke(messages)
            
        return response.dict()["content"]

    except Exception as e:
        return f"[Error generating AI response: {e}]"

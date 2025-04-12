from typing import List
from app_config import API_KEY
from utils import preprocess_md_to_html,check_persian
import json
import re
# Initialize OpenAI with your API key
from langchain_openai import ChatOpenAI 

# Choose your model here: "gpt-4" or "gpt-4-turbo"
# DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MODEL = "gpt-4o"

def generate_translate_model_response(text: str, destination: str) -> str:
    """
    Translates the given text into the specified target language using ChatOpenAI.

    Args:
        text (str): The text to translate.
        destination (str): The target language code (e.g., "fa" for Persian, "eng" for English).

    Returns:
        str: The translated text.
    """
    system_prompt = (
        "Translate the following text into the target language specified. "
        "Provide only the translated text in JSON format with the structure: "
        "```json\n{\"translation\": \"<translated_text>\"}\n```"
    )
    user_content = f"Text: {text}\n\nTarget Language: {destination}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    llm = ChatOpenAI(
        model=DEFAULT_MODEL, base_url="https://api.avalai.ir/v1", api_key=API_KEY
    )

    response = llm.invoke(messages)
    content = response.dict().get("content", "").strip()

    # Use regex to extract JSON content from triple backticks if present.
    json_match = re.search(r"```json(.*?)```", content, re.DOTALL)
    json_content = json_match.group(1).strip() if json_match else content

    try:
        parsed_response = json.loads(json_content)
        return parsed_response.get("translation", json_content)
    except json.JSONDecodeError:
        return json_content


def generate_medical_model_response(user_query: str, retrieved_chunks: List[str], model_name: str = DEFAULT_MODEL) -> str:

    """
    Generates a main response using OpenAI's ChatGPT based on the user query and retrieval context.

    Args:
        user_query (str): The user's query.
        retrieved_chunks (list[str]): Contextual chunks to provide to the model.
        model_name (str): Which model to use (e.g., "gpt-4", "gpt-4-turbo").

    Returns:
        str: The response from ChatGPT.
    """
    

    
    if len(retrieved_chunks) != 0:
        system_prompt = (
            "You are an expert medical assistant, specifically a specialist in dentistry. Use the provided context to answer the user's query as helpfully and accurately as possible. "
            "Only answer if the query is directly related to medical topics. If the query is not medical, politely inform the user that you only handle medical-related queries."
        )
        
        context = "\n\n".join(retrieved_chunks)

        user_content = f"Context:\n{context}\n\nQuery: {user_query}"
    else:
        system_prompt = (
            "You are an expert medical assistant, specifically a specialist in dentistry. Answer the user's query as helpfully and accurately as possible, but only if it is a medical query. "
            "If the query is not medical, respond by saying: 'I'm sorry, but I can only provide guidance for medical-related questions.'"
        )
        user_content = f"Query: {user_query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

        
        
    llm = ChatOpenAI(
             model=model_name, base_url="https://api.avalai.ir/v1", api_key=API_KEY)

    response = llm.invoke(messages)
        
    print(response.dict()["content"])
    return response.dict()["content"]





def generate_chat_response(user_query: str, retrieved_chunks: List[str], model_name: str = DEFAULT_MODEL) -> str:

        if check_persian(user_query):
            user_query = generate_translate_model_response(text=user_query,destination="eng")             
        
        try:

            res_chat_gpt = generate_medical_model_response(user_query,retrieved_chunks,model_name)
            
            persian_chat_gpt_res = generate_translate_model_response(text=res_chat_gpt,destination="fa")
            
            return preprocess_md_to_html(persian_chat_gpt_res,model_name=model_name)

        
        except Exception as e:
            return f"[Error generating AI response: {e}]"

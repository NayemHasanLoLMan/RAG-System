import os
import openai
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL
from typing import List, Optional, Dict

# Validate API key on import
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file. Please create a .env file and add it.")

# Initialize the client once and reuse it
client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Generate embeddings for a batch of texts using OpenAI.
    Filters out empty strings before sending to API.
    """
    # Filter out empty or whitespace-only strings, as they cause API errors
    valid_texts = [t.strip() for t in texts if t and t.strip()]
    
    if not valid_texts:
        return []

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=valid_texts
        )
        return [data.embedding for data in response.data]
    except openai.AuthenticationError:
        print("Authentication Error: Invalid OpenAI API key.")
        raise
    except openai.RateLimitError:
        print("Rate Limit Exceeded: Please wait and try again.")
        return None
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None
    
def get_chat_response(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Generate a chat response using OpenAI's chat completion.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        print("Authentication Error: Invalid OpenAI API key.")
        raise
    except openai.RateLimitError:
        print("Rate Limit Exceeded: Please wait and try again.")
        return None
    except Exception as e:
        print(f"Error in chat completion: {e}")
        return None
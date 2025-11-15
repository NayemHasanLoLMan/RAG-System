import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
CHAT_MODEL = "gpt-4-turbo"

# FAISS Configuration
FAISS_INDEX_PATH = "harry_potter_faiss.index"
METADATA_PATH = "harry_potter_metadata.json" # Using JSON instead of pickle
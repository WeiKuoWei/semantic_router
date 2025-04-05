import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"

# Database path
DB_PATH = os.getenv("DB_PATH", str("db"))

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default values
DEFAULT_TOP_K = 3
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
import os
from dotenv import load_dotenv
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MODEL_PATH = os.getenv("MODEL_PATH", "models/intervention_model.joblib")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "models/model_metadata.json")

"""
Shared configuration for the ingestion pipeline.
All secrets come from environment variables (set in GitHub Actions secrets).
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PIPELINE_DIR / "opportunities_markdown"
CSV_OUTPUT = PIPELINE_DIR / "latest_opportunities.csv"
SOURCE_META_PATH = PIPELINE_DIR / "source_metadata.json"
OPPORTUNITIES_JSON = PIPELINE_DIR / "opportunities_en.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Scraping ───────────────────────────────────────────────────────────
BASE_URL = "https://opportunitiescorners.com/"
EXCLUDE_DOMAINS = ["https://opportunitiescorners"]

# ── Database ───────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", os.getenv("db_host")),
    "port": os.getenv("DB_PORT", os.getenv("db_port")),
    "dbname": os.getenv("DB_NAME", os.getenv("db_name")),
    "user": os.getenv("DB_USER", os.getenv("db_user")),
    "password": os.getenv("DB_PASSWORD", os.getenv("db_password")),
}

# ── LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("grok_api"))
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", os.getenv("cerebras_api"))
LLM_MODEL_GROQ = "openai/gpt-oss-120b"
LLM_MODEL_CEREBRAS = "gpt-oss-120b"
SOURCE_LANGUAGE = "en"

# ── Embeddings (Jina) ─────────────────────────────────────────────────
JINA_API_KEY = os.getenv("JINA_API_KEY", os.getenv("jina_api_key"))
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
EMBED_BATCH_SIZE = 100

# ── Qdrant ─────────────────────────────────────────────────────────────
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT", os.getenv("qdrant_endpoint"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", os.getenv("qdrant_api_key"))
QDRANT_COLLECTION = "opportunities_v1"
UPSERT_BATCH_SIZE = 10

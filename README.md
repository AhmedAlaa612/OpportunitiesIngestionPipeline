# Ingestion Pipeline

Automated pipeline that scrapes, extracts, and embeds scholarship/opportunity data.

## Pipeline Steps

| Step | Script       | Description                                                                       |
| ---- | ------------ | --------------------------------------------------------------------------------- |
| 1    | `scrape.py`  | Scrapes new opportunities from opportunitiescorners.com → Markdown files          |
| 2    | `extract.py` | Extracts structured info via LLM → PostgreSQL (EN + AR) + `opportunities_en.json` |
| 3    | `embed.py`   | Generates Jina embeddings → Qdrant vector store                                   |

## Usage

### Run locally

```bash
cd search/pipeline

# Full pipeline
python run_pipeline.py

# Individual steps
python run_pipeline.py scrape
python run_pipeline.py extract
python run_pipeline.py embed
```

Requires a `.env` file (or exported env vars) with:

```
DB_HOST=...
DB_PORT=...
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
GROQ_API_KEY=...
CEREBRAS_API_KEY=...
JINA_API_KEY=...
QDRANT_ENDPOINT=...
QDRANT_API_KEY=...
```

### GitHub Actions

The workflow at `.github/workflows/ingestion.yml` runs daily at 06:00 UTC.  
could also be triggered manually from the **Actions** tab.

#### Required repository secrets

| Secret             | Description             |
| ------------------ | ----------------------- |
| `DB_HOST`          | PostgreSQL host         |
| `DB_PORT`          | PostgreSQL port         |
| `DB_NAME`          | Database name           |
| `DB_USER`          | Database user           |
| `DB_PASSWORD`      | Database password       |
| `GROQ_API_KEY`     | Groq API key            |
| `CEREBRAS_API_KEY` | Cerebras API key        |
| `JINA_API_KEY`     | Jina embeddings API key |
| `QDRANT_ENDPOINT`  | Qdrant cluster URL      |
| `QDRANT_API_KEY`   | Qdrant API key          |

## Architecture

```
scrape.py ──► opportunities_markdown/*.md + source_metadata.json
                    │
extract.py ◄────────┘
    │
    ├──► PostgreSQL (opportunities table, EN + AR)
    └──► opportunities_en.json
                    │
embed.py ◄──────────┘
    │
    └──► Qdrant (opportunities_v1 collection)
```

The pipeline is **incremental**: it checks the last `created_at` date in PostgreSQL and only processes newer opportunities. If no new data is found, downstream steps are skipped automatically.

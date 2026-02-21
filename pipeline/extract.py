"""
Step 2: Extract structured information from scraped Markdown files using LLMs.

Reads each Markdown file, sends it to an LLM for information extraction,
translates to both English and Arabic, and saves to PostgreSQL + JSON.
"""

import json
import re
import random
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json
from openai import OpenAI

from countries import normalize_country, normalize_countries

from config import (
    DB_CONFIG,
    GROQ_API_KEY,
    CEREBRAS_API_KEY,
    LLM_MODEL_GROQ,
    LLM_MODEL_CEREBRAS,
    SOURCE_LANGUAGE,
    OUTPUT_DIR,
    SOURCE_META_PATH,
    OPPORTUNITIES_JSON,
)

logger = logging.getLogger(__name__)


# ── LLM round-robin ───────────────────────────────────────────────────

_CLIENTS: List[Dict] = []
_client_index = 0


def _init_clients():
    global _CLIENTS
    if _CLIENTS:
        return
    if GROQ_API_KEY:
        _CLIENTS.append({
            "client": OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
            "model": LLM_MODEL_GROQ,
            "name": "groq",
        })
        logger.info("Groq client initialized")
    else:
        logger.warning("GROQ_API_KEY not set — skipping Groq client")
    if CEREBRAS_API_KEY:
        _CLIENTS.append({
            "client": OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1/"),
            "model": LLM_MODEL_CEREBRAS,
            "name": "cerebras",
        })
        logger.info("Cerebras client initialized")
    else:
        logger.warning("CEREBRAS_API_KEY not set — skipping Cerebras client")
    if not _CLIENTS:
        raise RuntimeError("No LLM API keys configured (GROQ_API_KEY / CEREBRAS_API_KEY)")
    logger.info("LLM clients ready: %s", ", ".join(c["name"] for c in _CLIENTS))


def _get_next_client() -> Dict:
    global _client_index
    entry = _CLIENTS[_client_index % len(_CLIENTS)]
    _client_index += 1
    return entry


def llm_call(messages, temperature=0.3, max_tokens=5000) -> str:
    _init_clients()
    primary = _get_next_client()
    try:
        logger.info("LLM call using %s (%s)", primary["name"], primary["model"])
        resp = primary["client"].chat.completions.create(
            model=primary["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("%s succeeded", primary["name"])
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error("%s FAILED: %s", primary["name"], str(e)[:120])
        fallback = _get_next_client()
        logger.info("Retrying with %s (%s)...", fallback["name"], fallback["model"])
        try:
            resp = fallback["client"].chat.completions.create(
                model=fallback["model"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info("%s succeeded (fallback)", fallback["name"])
            return resp.choices[0].message.content.strip()
        except Exception as e2:
            logger.error("%s FAILED (fallback): %s", fallback["name"], str(e2)[:120])
            raise


# ── Extraction prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an information extraction system for scholarships and opportunities.

You will receive a markdown document describing one or more opportunities.

Your task is to extract structured information for EACH opportunity in valid JSON format.

IMPORTANT: Extract the data in the language it appears. We will handle translation separately.

RULES:

* If a field is not mentioned in the text, DO NOT include it in the output JSON.
* Do NOT hallucinate or infer missing information.
* If a date is unclear or missing, omit that field entirely.
* For benefits, extract as a list of strings.

* For country AND eligible_nationalities — COUNTRY NAME NORMALIZATION IS CRITICAL:
  - ALWAYS use the standard short English name for every country. Examples:
    ✓ "USA"  (NOT "United States", "United States of America", "U.S.", "America")
    ✓ "UK"   (NOT "United Kingdom", "Great Britain", "England")
    ✓ "UAE"  (NOT "United Arab Emirates", "Emirates")
    ✓ "South Korea"  (NOT "Korea", "Republic of Korea")
    ✓ "Saudi Arabia"  (NOT "Kingdom of Saudi Arabia", "KSA")
    ✓ "Czech Republic"  (NOT "Czechia")
    ✓ "Netherlands"  (NOT "Holland", "The Netherlands")
    ✓ "Turkey"  (NOT "Türkiye")
  - For all other countries use the common short name: "Germany", "France", "Japan", "Egypt", "Canada", etc.
  - NEVER use formal/official names like "Federal Republic of Germany" or "Arab Republic of Egypt"
  - NEVER prefix with "The" (e.g., use "Philippines" not "The Philippines")
  - This rule applies to BOTH the "country" field AND every entry in "eligible_nationalities"

* For eligible_nationalities:
  - If explicitly stated as unrestricted, return "all".
  - Otherwise, return a list of countries using the normalized short names above.
  - If not mentioned, omit the field.

* For type.subtype, return an ARRAY of applicable subtypes
  (e.g., ["masters", "bachelor", "phd"]). An opportunity can be for multiple degree types.
  - program is considered academic only if it gives a degree (bachelor, master, phd). If it doesn't explicitly give a degree then it's non-academic even if it's educational in nature (e.g., a conference or exchange program).

* For target_segment:
  - Extract eligibility levels: "high school", "undergraduate", "graduate"
  - target_segment is for those who can apply, for example a bachelor's scholarship is open to high school and undergraduate students
  - Return as an array if multiple segments are eligible
  - target_segment can't be null if not clear then add the three segments as eligible

* For documents_required:
  - Extract any mentioned required documents: "cv", "transcript", "motivation letter", "cover letter", "portfolio", etc.
  - mention only major documents and Only include if documents are actually required by the opportunity
  - Return as an array of document types
  - resume and any form of cv should be written as "cv"

* For language_requirements:
  - If no language requirement is mentioned, DO NOT include this field at all.
  - If language tests are required, return an object where each key is the exam name and the value is the score.
    Example: { "IELTS": "6.5", "TOEFL": "90" }
  - If an exam is mentioned but NO score is specified, return an empty string as its value.

* For application_fee:
  - If the text doesn't clearly state a fee, OMIT the field entirely.
  - Only include application_fee if a specific non-zero amount is stated.

* type must not be null, if it is not a degree then it's non-academic
* if program is not funded or doesn't state fund type then omit it
* if program has several fund types add them to the array

Return one JSON object per opportunity.
If there are multiple opportunities in one document, return a JSON array with multiple objects.

Allowed top-level fields (only include those that actually appear in the text):

{
  "title": "opportunity title in original language",
  "description": "main description in original language",
  "eligibility": "eligibility criteria in original language",
  "country": "[normalized short country names: USA, UK, Germany, etc.]",
  "location": "",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "duration": "",
  "fund_type": "[\\"fully_funded\\", \\"partially_funded\\"]",
  "benefits": ["list of bullet points in original language"],
  "deadline": "YYYY-MM-DD",
  "gpa": "float",
  "min_age": "",
  "max_age": "",
  "type": {
    "category": "academic | non_academic",
    "subtype": ["masters", "bachelor", "phd", "conference", "exchange", "prize", "internship", "camp", "volunteering", "workshop"]
  },
  "application_fee": "only include if a specific non-zero amount is stated",
  "application_link": "",
  "official_website": "",
  "target_segment": ["high school", "undergraduate", "graduate"],
  "language_requirements": {
    "exam_name": "score or empty string"
  },
  "eligible_nationalities": "all | [list of normalized short country names]",
  "documents_required": [],
  "is_remote": false
}
"""


# ── Extraction & translation ──────────────────────────────────────────


def extract_opportunity_info(markdown_content: str, filename: str) -> Optional[Dict[str, Any]]:
    user_prompt = f"""Extract structured information from the following markdown document about an opportunity:

    ---MARKDOWN START---
    {markdown_content}
    ---MARKDOWN END---

    Output rules:
    - Return ONLY valid JSON (no explanations, no comments, no markdown)
    - Return a single JSON object
    - If a field is not explicitly mentioned in the text, OMIT IT ENTIRELY
    - Do NOT guess, infer, or add missing information
    - Preserve the original language of the document exactly
    - Follow the SYSTEM_PROMPT schema strictly
    """
    try:
        response_text = llm_call(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=5000,
        )

        if "```" in response_text:
            matches = re.findall(r"```json(.*?)```|```(.*?)```", response_text, re.S)
            if matches:
                response_text = next(x for pair in matches for x in pair if x).strip()

        if not response_text.startswith("{") and not response_text.startswith("["):
            json_start = response_text.find("{")
            if json_start != -1:
                response_text = response_text[json_start:]

        parsed_data = json.loads(response_text)

        if isinstance(parsed_data, list):
            for item in parsed_data:
                item["id"] = str(uuid.uuid4())
                item["_source_file"] = filename
        else:
            parsed_data["id"] = str(uuid.uuid4())
            parsed_data["_source_file"] = filename

        return parsed_data

    except Exception as e:
        logger.error("Extraction failed for %s: %s", filename, str(e)[:80])
        return None


def translate_to_language(data: Dict[str, Any], target_language: str) -> Dict[str, Any]:
    preserved_fields = {}
    for field in ("id", "_source_file"):
        if field in data:
            preserved_fields[field] = data[field]

    data_to_translate = {k: v for k, v in data.items() if k not in preserved_fields}
    data_json_str = json.dumps(data_to_translate, ensure_ascii=False, indent=2)

    if target_language == "en":
        instruction = (
            "Translate the values in this JSON to English.\n"
            "STRICT RULES:\n"
            "1. Translate ALL string values.\n"
            "2. If a value is snake_case, translate to normal text.\n"
            "3. DO NOT translate JSON keys.\n"
            "4. DO NOT translate: numeric strings, URLs, Emails, ISO dates.\n"
            "5. Return ONLY valid JSON."
        )
    else:
        instruction = (
            "Translate the values in this JSON to Egyptian Arabic with a friendly tone.\n"
            "STRICT RULES:\n"
            "1. Translate ALL string values to Arabic.\n"
            "2. If a value is snake_case, translate to its Arabic equivalent.\n"
            "3. DO NOT translate JSON keys.\n"
            "4. DO NOT translate: numeric strings, URLs, Emails, ISO dates.\n"
            "5. Return ONLY valid JSON."
        )

    try:
        response_text = llm_call(
            messages=[
                {"role": "system", "content": "You are a professional translator. Respond with valid JSON only."},
                {"role": "user", "content": f"{instruction}\n\n{data_json_str}"},
            ],
            temperature=0.3,
            max_tokens=5000,
        )

        if "```" in response_text:
            matches = re.findall(r"```json(.*?)```|```(.*?)```", response_text, re.S)
            if matches:
                response_text = next(x for pair in matches for x in pair if x).strip()

        if not response_text.startswith("{"):
            json_start = response_text.find("{")
            if json_start != -1:
                response_text = response_text[json_start:]

        translated = json.loads(response_text)
        for field, value in preserved_fields.items():
            translated[field] = value
        return translated

    except Exception as e:
        logger.warning("Translation failed: %s", str(e)[:80])
        return data


# ── DB helpers ─────────────────────────────────────────────────────────


def ensure_list(val):
    if val is None:
        return None
    return [val] if isinstance(val, str) else list(val)


def parse_date(val):
    if not val or not isinstance(val, str):
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        return val
    return None


def normalize_opp_countries(data: dict):
    if "country" in data:
        raw = data["country"]
        data["country"] = normalize_countries(raw if isinstance(raw, list) else [raw])
    if "eligible_nationalities" in data:
        raw = data["eligible_nationalities"]
        if isinstance(raw, list):
            data["eligible_nationalities"] = normalize_countries(raw)
        elif isinstance(raw, str) and raw.lower() != "all":
            data["eligible_nationalities"] = [normalize_country(raw)]


def save_to_db(
    opportunity_id: str,
    data_en: dict,
    data_ar: dict,
    source: str = "opportunitiescorners",
    source_url: str = None,
    source_md: str = None,
):
    normalize_opp_countries(data_en)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    now = datetime.now(timezone.utc)

    type_info = data_en.get("type", {}) or {}
    category = type_info.get("category")
    subtype = ensure_list(type_info.get("subtype"))
    country = ensure_list(data_en.get("country"))
    fund_type = ensure_list(data_en.get("fund_type"))
    target_segment = ensure_list(data_en.get("target_segment"))
    deadline = parse_date(data_en.get("deadline"))
    is_remote = bool(data_en.get("is_remote", False))

    cur.execute(
        """
        INSERT INTO opportunities (
            id, source, source_url, source_md,
            data_en, data_ar,
            category, subtype, country, fund_type, target_segment,
            deadline, is_remote,
            created_at, updated_at
        )
        VALUES (
            %s::uuid, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            source = EXCLUDED.source,
            source_url = EXCLUDED.source_url,
            source_md = EXCLUDED.source_md,
            data_en = EXCLUDED.data_en,
            data_ar = EXCLUDED.data_ar,
            category = EXCLUDED.category,
            subtype = EXCLUDED.subtype,
            country = EXCLUDED.country,
            fund_type = EXCLUDED.fund_type,
            target_segment = EXCLUDED.target_segment,
            deadline = EXCLUDED.deadline,
            is_remote = EXCLUDED.is_remote,
            updated_at = EXCLUDED.updated_at;
        """,
        (
            opportunity_id, source, source_url, source_md,
            Json(data_en), Json(data_ar),
            category, subtype, country, fund_type, target_segment,
            deadline, is_remote,
            now, now,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()


# ── Main ───────────────────────────────────────────────────────────────


def run() -> bool:
    """
    Execute the extraction step.
    Returns True if new opportunities were extracted and saved.
    """
    # Load source metadata
    source_metadata = {}
    if SOURCE_META_PATH.exists():
        with open(SOURCE_META_PATH, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)
        logger.info("Loaded source metadata for %d files", len(source_metadata))

    markdown_files = sorted(OUTPUT_DIR.glob("*.md"))
    logger.info("Found %d markdown files to process", len(markdown_files))

    if not markdown_files:
        logger.warning("No markdown files found in %s — run scrape step first", OUTPUT_DIR)
        return False

    all_extracted_data = []
    successful = 0
    failed = 0
    skipped_no_link = 0

    for idx, file_path in enumerate(markdown_files, 1):
        logger.info("[%d/%d] Processing: %s", idx, len(markdown_files), file_path.name[:50])

        try:
            markdown_content = file_path.read_text(encoding="utf-8")
            extracted_data = extract_opportunity_info(markdown_content, file_path.name)

            if extracted_data:
                meta = source_metadata.get(file_path.name, {})
                items = extracted_data if isinstance(extracted_data, list) else [extracted_data]

                valid_items = []
                for item in items:
                    if not item.get("application_link"):
                        skipped_no_link += 1
                        continue
                    item["_source"] = meta.get("source", "opportunitiescorners")
                    item["_source_url"] = meta.get("source_url", "")
                    item["_source_md"] = markdown_content
                    valid_items.append(item)

                if valid_items:
                    all_extracted_data.extend(valid_items)
                    logger.info("  Extracted %d with application link", len(valid_items))
                else:
                    logger.info("  Skipped (no application_link)")
                successful += 1
            else:
                logger.warning("  No data extracted")
                failed += 1

            time.sleep(random.uniform(10, 20))

        except Exception as e:
            logger.error("  Error: %s", str(e)[:80])
            failed += 1

    logger.info("Extraction done — %d ok, %d failed, %d skipped (no link)", successful, failed, skipped_no_link)

    if not all_extracted_data:
        logger.warning("No valid opportunities extracted")
        return False

    # ── Translate + save to DB ─────────────────────────────────────────
    all_en_data = []
    saved = 0

    for idx, item in enumerate(all_extracted_data, 1):
        opp_id = item["id"]
        title_short = item.get("title", "unknown")[:50]
        logger.info("[%d/%d] %s", idx, len(all_extracted_data), title_short)

        source = item.pop("_source", "opportunitiescorners")
        source_url = item.pop("_source_url", None)
        source_md = item.pop("_source_md", None)

        try:
            if SOURCE_LANGUAGE == "en":
                data_en = item
                logger.info("  Translating to Arabic...")
                data_ar = translate_to_language(item, "ar")
            else:
                data_ar = item
                logger.info("  Translating to English...")
                data_en = translate_to_language(item, "en")

            for d in (data_en, data_ar):
                d.pop("_source_file", None)

            save_to_db(opp_id, data_en, data_ar, source=source, source_url=source_url, source_md=source_md)
            logger.info("  Saved to DB")
            saved += 1
            all_en_data.append(data_en)

            time.sleep(random.uniform(10, 20))

        except Exception as e:
            logger.error("  Failed: %s", str(e)[:80])

    # Save English JSON for embed step
    with open(OPPORTUNITIES_JSON, "w", encoding="utf-8") as f:
        json.dump(all_en_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d opportunities to %s", len(all_en_data), OPPORTUNITIES_JSON)
    logger.info("Total saved to DB: %d", saved)
    return saved > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()

"""
Step 1: Scrape latest opportunities from opportunitiescorners.com.

Fetches the homepage, finds new opportunities published after the last
scraped date in PostgreSQL, downloads each page, and converts to Markdown.
"""

import json
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import psycopg2
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from config import (
    BASE_URL,
    DB_CONFIG,
    EXCLUDE_DOMAINS,
    OUTPUT_DIR,
    CSV_OUTPUT,
    SOURCE_META_PATH,
)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def get_last_scraped_date() -> Optional[datetime]:
    """Query PostgreSQL for the max created_at date from the opportunities table."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT MAX(created_at) FROM opportunities;")
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        if result:
            if result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)
            logger.info("Last scraped date from DB: %s", result.isoformat())
            return result
        logger.info("No existing opportunities in DB — will scrape all")
        return None
    except Exception as e:
        logger.warning("Could not query DB for last date: %s — will scrape all", e)
        return None


def html_to_clean_md(html: str, exclude_domains: Optional[List[str]] = None) -> str:
    """Parse HTML, remove unwanted elements, convert to clean Markdown."""
    try:
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            try:
                tag.decompose()
            except Exception:
                pass

        for button in soup.find_all("button"):
            try:
                a = button.find("a", href=True)
                if a:
                    button.replace_with(a)
            except Exception:
                pass

        if exclude_domains:
            a_tags = list(soup.find_all("a", href=True))
            for a_tag in a_tags:
                try:
                    href = a_tag.get("href", "") if a_tag else None
                    if href and any(domain in href for domain in exclude_domains):
                        p_tag = a_tag.find_parent("p")
                        if p_tag and "Also Check" in p_tag.get_text():
                            p_tag.decompose()
                        else:
                            parent_tag = a_tag.find_parent()
                            if parent_tag and parent_tag.name != "body":
                                try:
                                    parent_tag.decompose()
                                except Exception:
                                    pass
                except Exception:
                    pass

        clean_html = soup.decode_contents()
        return md(clean_html, heading_style="ATX", strip=["img"])
    except Exception as e:
        logger.error("html_to_clean_md error: %s", e)
        return ""


def sanitize_filename(filename: str) -> str:
    """Convert a title to a safe filename."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename).strip()[:100]
    return sanitized if sanitized else "opportunity"


# ── Main ───────────────────────────────────────────────────────────────


def run() -> bool:
    """
    Execute the scraping step.
    Returns True if there are new opportunities to process, False otherwise.
    """
    import pandas as pd

    last_scraped_date = get_last_scraped_date()

    logger.info("Fetching homepage from %s ...", BASE_URL)
    response = requests.get(BASE_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    latest_section = soup.find("div", {"id": "tdi_13"})
    if not latest_section:
        logger.error("Could not find Latest Opportunities section on homepage")
        return False

    opportunity_items = latest_section.find_all("div", class_="td_module_6")
    all_opportunities = []

    for item in opportunity_items:
        title_elem = item.find("h3", class_="entry-title")
        if title_elem:
            a_tag = title_elem.find("a")
            title = a_tag.text.strip() if a_tag else None
            link = a_tag["href"] if a_tag else None
        else:
            title, link = None, None

        date_elem = item.find("time", class_="td-module-date")
        date_text = date_elem.text.strip() if date_elem else None
        datetime_attr = date_elem.get("datetime") if date_elem else None

        all_opportunities.append(
            {"title": title, "link": link, "date_text": date_text, "datetime": datetime_attr}
        )

    logger.info("Found %d total opportunities on homepage", len(all_opportunities))

    # Filter by last scraped date
    if last_scraped_date:
        opportunities_data = []
        skipped = 0
        for opp in all_opportunities:
            if opp["datetime"]:
                opp_date = datetime.fromisoformat(opp["datetime"])
                if opp_date > last_scraped_date:
                    opportunities_data.append(opp)
                else:
                    skipped += 1
            else:
                opportunities_data.append(opp)
        logger.info("%d NEW opportunities (skipped %d already scraped)", len(opportunities_data), skipped)
    else:
        opportunities_data = all_opportunities
        logger.info("Processing all %d opportunities (first run)", len(opportunities_data))

    if not opportunities_data:
        logger.info("No new opportunities to process. Pipeline done.")
        return False

    # ── Save CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(opportunities_data)
    df.to_csv(CSV_OUTPUT, index=False)
    logger.info("Saved %d opportunities metadata to %s", len(opportunities_data), CSV_OUTPUT)

    # ── Save source_metadata.json ──────────────────────────────────────
    source_meta = {}
    for opp in opportunities_data:
        if opp.get("title"):
            fname = sanitize_filename(opp["title"]) + ".md"
            source_meta[fname] = {
                "source": opp.get("source", "opportunitiescorners"),
                "source_url": opp.get("source_url") or opp.get("link"),
            }
    with open(SOURCE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(source_meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved source metadata for %d opportunities", len(source_meta))

    # ── Scrape each page → Markdown ────────────────────────────────────
    for old_file in OUTPUT_DIR.glob("*.md"):
        old_file.unlink()
    logger.info("Cleared %s/ for fresh batch", OUTPUT_DIR)

    successful = 0
    failed = 0

    for idx, opp in enumerate(opportunities_data, 1):
        if not opp["link"]:
            logger.warning("[%d] %s — no link found", idx, opp["title"])
            failed += 1
            continue

        try:
            logger.info("[%d/%d] Fetching: %s", idx, len(opportunities_data), (opp["title"] or "")[:60])
            opp_response = requests.get(opp["link"], timeout=15)
            opp_response.raise_for_status()

            opp_soup = BeautifulSoup(opp_response.content, "html.parser")
            main_div = opp_soup.find("div", class_="td-main-content")
            article = main_div.find("article") if main_div else None

            if article:
                content_html = article.decode_contents()
            else:
                content_div = opp_soup.find("div", class_="td-post-content")
                if content_div:
                    content_html = content_div.decode_contents()
                else:
                    logger.warning("[%d] No content found", idx)
                    failed += 1
                    continue

            markdown_content = html_to_clean_md(content_html, exclude_domains=EXCLUDE_DOMAINS)

            opp["source_url"] = opp["link"]
            opp["source"] = "opportunitiescorners"

            filename = sanitize_filename(opp["title"]) + ".md"
            filepath = OUTPUT_DIR / filename

            full_md = (
                f"# {opp['title']}\n\n"
                f"**Date:** {opp['date_text']}\n\n"
                f"**Source:** [{opp['link']}]({opp['link']})\n\n---\n\n"
                f"{markdown_content}"
            )
            opp["source_md"] = full_md

            filepath.write_text(full_md, encoding="utf-8")
            logger.info("[%d] Saved %s", idx, filename)
            successful += 1
            time.sleep(1)

        except Exception as e:
            logger.error("[%d] Error: %s", idx, str(e)[:80])
            failed += 1

    logger.info("Scrape complete — %d succeeded, %d failed", successful, failed)
    return successful > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()

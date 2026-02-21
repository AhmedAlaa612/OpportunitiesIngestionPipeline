"""
Pipeline orchestrator — runs scrape → extract → embed in sequence.

Usage:
    python run_pipeline.py          # run all 3 steps
    python run_pipeline.py scrape   # run only scrape
    python run_pipeline.py extract  # run only extract
    python run_pipeline.py embed    # run only embed
"""

import logging
import sys
import time

import scrape
import extract
import embed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")

STEPS = {
    "scrape": scrape.run,
    "extract": extract.run,
    "embed": embed.run,
}


def main():
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(STEPS.keys())

    for step_name in requested:
        if step_name not in STEPS:
            logger.error("Unknown step: %s (choose from %s)", step_name, ", ".join(STEPS))
            sys.exit(1)

    logger.info("=== Starting pipeline: %s ===", " → ".join(requested))
    start = time.time()

    for step_name in requested:
        logger.info("── Step: %s ──", step_name)
        step_start = time.time()

        try:
            has_work = STEPS[step_name]()
        except Exception:
            logger.exception("Step '%s' failed with an unhandled exception", step_name)
            sys.exit(1)

        elapsed = time.time() - step_start
        logger.info("── %s finished in %.1fs (has_work=%s) ──", step_name, elapsed, has_work)

        # If scrape/extract returned False (nothing new), skip downstream steps
        if not has_work and step_name in ("scrape", "extract") and len(requested) > 1:
            logger.info("No new data from '%s' — skipping remaining steps", step_name)
            break

    total = time.time() - start
    logger.info("=== Pipeline completed in %.1fs ===", total)


if __name__ == "__main__":
    main()

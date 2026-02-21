"""
Country name normalizer using pycountry + a small custom alias table.

Usage:
    normalize_country("Kingdom of Saudi Arabia")  # -> "Saudi Arabia"
    normalize_country("U.S.A")                    # -> "USA"
    normalize_countries(["United States", "UK"])   # -> ["USA", "UK"]
"""

from __future__ import annotations

import re
from typing import List, Optional

import pycountry

_CUSTOM_ALIASES: dict[str, list[str]] = {
    "USA": [
        "usa", "u.s.", "u.s", "u.s.a", "u.s.a.", "us", "america",
        "united states", "united states of america",
        "the united states", "the united states of america",
        "united states of amereca",  # common typo
    ],
    "UK": [
        "uk", "u.k.", "u.k", "united kingdom",
        "great britain", "britain", "england",
        "the united kingdom",
    ],
    "UAE": [
        "uae", "u.a.e.", "u.a.e", "united arab emirates",
        "the uae", "the united arab emirates", "emirates",
    ],
    "South Korea": [
        "south korea", "korea", "republic of korea",
        "rok", "s. korea", "korea, republic of",
    ],
    "North Korea": [
        "north korea", "dprk",
        "democratic people's republic of korea",
        "korea, democratic people's republic of",
    ],
    "Russia": [
        "russia", "russian federation", "the russian federation",
    ],
    "Czech Republic": [
        "czech republic", "czechia", "the czech republic",
    ],
    "Taiwan": [
        "taiwan", "chinese taipei", "republic of china",
        "taiwan, province of china",
    ],
    "Saudi Arabia": [
        "saudi arabia", "ksa", "k.s.a.", "k.s.a",
        "kingdom of saudi arabia", "the kingdom of saudi arabia",
        "saudi",
    ],
    "Turkey": [
        "turkey", "türkiye", "turkiye", "turkia",
    ],
    "Netherlands": [
        "netherlands", "the netherlands", "holland",
    ],
    "Palestine": [
        "palestine", "palestinian territories",
        "state of palestine", "palestine, state of",
    ],
    "New Zealand": [
        "new zealand", "nz",
    ],
    "Bosnia": [
        "bosnia", "bosnia and herzegovina",
    ],
    "North Macedonia": [
        "north macedonia", "macedonia",
    ],
    "Myanmar": [
        "myanmar", "burma",
    ],
    "Hong Kong": [
        "hong kong",
    ],
    "Macau": [
        "macau", "macao",
    ],
}

_LOOKUP: dict[str, str] = {}

# 1. Custom aliases first (highest priority)
for _canonical, _aliases in _CUSTOM_ALIASES.items():
    _LOOKUP[_canonical.lower()] = _canonical
    for _alias in _aliases:
        _LOOKUP[_alias.lower()] = _canonical

# 2. pycountry entries (name, official_name, common_name, alpha_2, alpha_3)
for _c in pycountry.countries:
    _name = _c.name
    # Skip if already handled by custom aliases
    if _name.lower() not in _LOOKUP:
        _LOOKUP[_name.lower()] = _name
    for _attr in ("official_name", "common_name"):
        _alt = getattr(_c, _attr, None)
        if _alt and _alt.lower() not in _LOOKUP:
            _LOOKUP[_alt.lower()] = _name
    # alpha codes
    if _c.alpha_2.lower() not in _LOOKUP:
        _LOOKUP[_c.alpha_2.lower()] = _name
    if _c.alpha_3.lower() not in _LOOKUP:
        _LOOKUP[_c.alpha_3.lower()] = _name


_STRIP_ARTICLE = re.compile(r"^the\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_country(name: str) -> str:
    """
    Map any country name variant to its short form.
    Falls back to pycountry fuzzy search, then returns title-cased input.
    """
    if not name or not isinstance(name, str):
        return name

    cleaned = _WHITESPACE.sub(" ", name.strip())
    key = cleaned.lower()

    # 1. Direct lookup
    if key in _LOOKUP:
        return _LOOKUP[key]

    # 2. Strip leading "the" and retry
    stripped = _STRIP_ARTICLE.sub("", key).strip()
    if stripped in _LOOKUP:
        return _LOOKUP[stripped]

    # 3. pycountry fuzzy search
    try:
        results = pycountry.countries.search_fuzzy(cleaned)
        if results:
            found = results[0].name
            # Check if custom alias overrides pycountry name
            canonical = _LOOKUP.get(found.lower(), found)
            # Cache for next time
            _LOOKUP[key] = canonical
            return canonical
    except LookupError:
        pass

    # 4. Give up
    return cleaned.title()


def normalize_countries(countries: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalize a list of country names. Deduplicates after normalization.
    Returns None if input is None, empty list if input is empty.
    """
    if countries is None:
        return None
    seen: set[str] = set()
    result: list[str] = []
    for c in countries:
        n = normalize_country(c)
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result

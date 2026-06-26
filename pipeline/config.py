"""Load and lightly validate the YAML configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml

from .util import ROOT

CONFIG_DIR = os.path.join(ROOT, "config")


def _load(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{name}: expected a mapping at the top level")
    return data


@dataclass
class Category:
    key: str
    label: str
    weight: float
    keywords: str


@dataclass
class Country:
    name: str
    map_name: str
    iso2: str
    lang: str
    weight: float
    feeds: list = field(default_factory=list)        # list[{url, source, lang}]
    gnews: list = field(default_factory=list)         # list[{q, hl, gl, ceid}]
    match: list = field(default_factory=list)         # attribution terms (en + native)
    exclude: list = field(default_factory=list)       # ambiguous terms to reject


@dataclass
class Instrument:
    id: str
    name: str
    symbol: str
    source: str
    asset_class: str
    currency: str = "USD"
    fallback_source: str | None = None
    fallback_symbol: str | None = None
    pegged: bool = False


@dataclass
class Config:
    settings: dict
    categories: list            # list[Category]
    cat_weight: dict            # key -> weight
    cat_label: dict             # key -> label
    source_credibility: dict
    countries: list             # list[Country]
    shared_feeds: list          # list[{url, source, lang}]
    instruments: list           # list[Instrument]

    def category_keys(self) -> list:
        return [c.key for c in self.categories]


def _build_categories(cats_raw, weight_set="default"):
    """Build Category objects, selecting the active weight column. 'goldstein'
    uses the literature-anchored alternative weights defined in categories.yml;
    'default' uses the standard weights (METHODOLOGY_REVIEW F5 / PATCH_PLAN P11)."""
    wkey = "weight_goldstein" if weight_set == "goldstein" else "weight"
    return [
        Category(key=c["key"], label=c["label"],
                 weight=float(c.get(wkey, c["weight"])),
                 keywords=c.get("keywords", "") or "")
        for c in cats_raw["categories"]
    ]


def load() -> Config:
    settings = _load("settings.yml")
    cats_raw = _load("categories.yml")
    countries_raw = _load("countries.yml")
    markets_raw = _load("markets.yml")

    weight_set = str(settings.get("score", {}).get("weight_set", "default")).lower()
    categories = _build_categories(cats_raw, weight_set)
    cat_weight = {c.key: c.weight for c in categories}
    cat_label = {c.key: c.label for c in categories}
    source_credibility = {str(k).lower(): float(v)
                          for k, v in (cats_raw.get("source_credibility") or {}).items()}

    attribution = countries_raw.get("attribution") or {}
    countries = []
    for c in countries_raw["countries"]:
        attr = attribution.get(c["name"], {})
        countries.append(Country(
            name=c["name"], map_name=c["map_name"], iso2=c["iso2"],
            lang=c.get("lang", "en"), weight=float(c.get("weight", 1.0)),
            feeds=c.get("feeds") or [], gnews=c.get("gnews") or [],
            match=attr.get("match") or [], exclude=attr.get("exclude") or [],
        ))
    shared_feeds = countries_raw.get("shared_feeds") or []

    instruments = []
    for m in markets_raw["instruments"]:
        instruments.append(Instrument(
            id=m["id"], name=m["name"], symbol=m["symbol"], source=m["source"],
            asset_class=m["asset_class"], currency=m.get("currency", "USD"),
            fallback_source=m.get("fallback_source"),
            fallback_symbol=m.get("fallback_symbol"),
            pegged=bool(m.get("pegged", False)),
        ))

    if not countries:
        raise ValueError("countries.yml: no countries configured")
    if not categories:
        raise ValueError("categories.yml: no categories configured")

    return Config(
        settings=settings, categories=categories, cat_weight=cat_weight,
        cat_label=cat_label, source_credibility=source_credibility,
        countries=countries, shared_feeds=shared_feeds, instruments=instruments,
    )

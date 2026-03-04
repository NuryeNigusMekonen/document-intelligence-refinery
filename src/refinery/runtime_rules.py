from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import Settings


DEFAULT_RUNTIME_RULES: dict[str, Any] = {
    "extraction": {
        "fast_min_chars_per_page": 100.0,
        "fast_max_image_area_ratio": 0.50,
        "fast_confidence_floor": 0.60,
        "layout_confidence_floor": 0.72,
        "escalate_to_vision_floor": 0.50,
        "handwriting_whitespace_threshold": 0.92,
        "handwriting_char_density_threshold": 0.00005,
        "handwriting_image_ratio_threshold": 0.55,
    },
    "chunking": {
        "max_tokens": 350,
        "list_split_item_boundary_only": True,
        "heading_regex": r"^(\d+(\.\d+)*)\s+.+|^[A-Z][A-Za-z\s]{3,60}$",
        "list_item_regex": r"^\s*(\d+\.|[-*])\s+",
        "cross_ref_regex": r"see\s+(table|figure|section)\s+\d+",
        "resolvable_ref_regex": r"see\s+(table|figure|section)\s+(\d+)",
        "table_block_overlap_suppress_ratio": 0.60,
        "table_block_suppress_ethiopic_min_chars": 3,
    },
    "triage": {
        "domain_keywords": {
            "financial": ["balance", "income", "financial", "earnings", "assets", "revenue", "expense"],
            "legal": ["contract", "agreement", "terms", "legal", "whereas", "party", "liability"],
            "technical": ["manual", "spec", "api", "technical", "architecture", "endpoint", "algorithm"],
            "medical": ["clinical", "patient", "medical", "diagnosis", "treatment", "prescription"],
        }
    },
    "file_type_profiles": {
        "image": {
            "origin_type": "scanned_image",
            "layout_complexity": "figure_heavy",
            "estimated_extraction_cost": "needs_vision_model",
            "origin_confidence": 0.88,
            "layout_confidence": 0.75,
            "extraction_cost_confidence": 0.90,
            "domain_hint": "general",
            "domain_confidence": 0.55,
        },
        "excel": {
            "origin_type": "native_digital",
            "layout_complexity": "table_heavy",
            "estimated_extraction_cost": "needs_layout_model",
            "origin_confidence": 0.85,
            "layout_confidence": 0.85,
            "extraction_cost_confidence": 0.80,
            "domain_hint": "general",
            "domain_confidence": 0.55,
        },
        "docx": {
            "origin_type": "native_digital",
            "layout_complexity": "mixed",
            "estimated_extraction_cost": "fast_text_sufficient",
            "origin_confidence": 0.80,
            "layout_confidence": 0.72,
            "extraction_cost_confidence": 0.75,
            "domain_hint": "general",
            "domain_confidence": 0.55,
        },
        "markdown": {
            "origin_type": "native_digital",
            "layout_complexity": "mixed",
            "estimated_extraction_cost": "fast_text_sufficient",
            "origin_confidence": 0.82,
            "layout_confidence": 0.72,
            "extraction_cost_confidence": 0.75,
            "domain_hint": "general",
            "domain_confidence": 0.55,
        },
        "default": {
            "origin_type": "mixed",
            "layout_complexity": "mixed",
            "estimated_extraction_cost": "needs_layout_model",
            "origin_confidence": 0.72,
            "layout_confidence": 0.70,
            "extraction_cost_confidence": 0.70,
            "domain_hint": "general",
            "domain_confidence": 0.55,
        },
    },
    "facts": {
        "financial_keys": ["Revenue", "Net profit", "Total assets", "Operating income", "EPS"],
        "context_window_chars": 60,
        "include_year_pattern": True,
        "year_regex": r"\\b(FY\\s?\\d{2,4}|\\d{4})\\b",
        "amharic_keywords_file": "src/refinery/resources/amharic_keywords.yaml",
    },
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _rules_file(settings: Settings) -> Path:
    path = settings.runtime_rules_file
    if path.is_absolute():
        return path
    return (settings.workspace_root / path).resolve()


def load_runtime_rules(settings: Settings) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_RUNTIME_RULES)
    path = _rules_file(settings)
    if not path.exists():
        return merged
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return merged
    if not isinstance(loaded, dict):
        return merged
    return _deep_merge(merged, loaded)

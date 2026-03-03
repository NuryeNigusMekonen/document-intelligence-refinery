from __future__ import annotations

import importlib
import importlib.util
import re
from typing import Literal


GE_EZ_RE = re.compile(r"[\u1200-\u137F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str, mode: Literal["script", "langdetect"] = "script") -> dict:
    content = (text or "").strip()
    if not content:
        return {"language": "unknown", "confidence": 0.0, "signals": {"empty": True}}

    if mode == "langdetect":
        try:
            if importlib.util.find_spec("langdetect") is None:
                raise ModuleNotFoundError("langdetect is not installed")
            module = importlib.import_module("langdetect")
            detect_langs = getattr(module, "detect_langs")
            langs = detect_langs(content)
            if not langs:
                return {"language": "unknown", "confidence": 0.0, "signals": {"mode": "langdetect", "candidates": []}}
            best = langs[0]
            code = str(best.lang)
            conf = float(best.prob)
            if code.startswith("am"):
                return {"language": "am", "confidence": conf, "signals": {"mode": "langdetect", "code": code}}
            if code.startswith("en"):
                return {"language": "en", "confidence": conf, "signals": {"mode": "langdetect", "code": code}}
            return {"language": "unknown", "confidence": conf * 0.5, "signals": {"mode": "langdetect", "code": code}}
        except Exception:
            mode = "script"

    geez_count = len(GE_EZ_RE.findall(content))
    latin_count = len(LATIN_RE.findall(content))
    alpha_count = sum(1 for ch in content if ch.isalpha())
    alpha_count = max(alpha_count, 1)

    geez_ratio = geez_count / alpha_count
    latin_ratio = latin_count / alpha_count

    if geez_count > 0 and geez_ratio >= 0.15:
        return {
            "language": "am",
            "confidence": min(0.99, 0.75 + geez_ratio * 0.2),
            "signals": {"mode": "script", "geez_count": geez_count, "latin_count": latin_count, "geez_ratio": geez_ratio},
        }
    if latin_count > 0 and latin_ratio >= 0.4:
        return {
            "language": "en",
            "confidence": min(0.9, 0.55 + latin_ratio * 0.25),
            "signals": {"mode": "script", "geez_count": geez_count, "latin_count": latin_count, "latin_ratio": latin_ratio},
        }
    return {
        "language": "unknown",
        "confidence": 0.3,
        "signals": {"mode": "script", "geez_count": geez_count, "latin_count": latin_count},
    }


def select_ocr_lang(
    profile_language: str,
    profile_conf: float,
    *,
    ocr_amharic_enabled: bool = True,
    ocr_lang_default: str = "eng",
    ocr_lang_fallback: str = "eng+amh",
) -> str:
    lang = (profile_language or "unknown").lower()
    if lang == "am" and ocr_amharic_enabled:
        return "amh+eng"
    if lang == "unknown" or profile_conf < 0.45:
        return ocr_lang_fallback
    return ocr_lang_default

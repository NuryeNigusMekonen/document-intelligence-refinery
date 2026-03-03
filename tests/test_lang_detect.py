import importlib.util

from refinery.lang_detect import detect_language
from tests.fixtures.amharic_samples import AMHARIC_PARAGRAPH, MIXED_EN_AM


def test_detect_language_script_amharic():
    result = detect_language(AMHARIC_PARAGRAPH, mode="script")
    assert result["language"] == "am"
    assert result["confidence"] >= 0.7


def test_detect_language_script_english_or_amharic_on_mixed():
    result = detect_language(MIXED_EN_AM, mode="script")
    assert result["language"] in {"am", "en"}


def test_detect_language_langdetect_mode_safe_without_dependency():
    result = detect_language("Revenue increased this year", mode="langdetect")
    assert result["language"] in {"en", "unknown", "am"}


def test_detect_language_langdetect_if_installed():
    if importlib.util.find_spec("langdetect") is None:
        return
    result = detect_language("This is an English sentence.", mode="langdetect")
    assert result["language"] in {"en", "unknown"}

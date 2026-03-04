from pathlib import Path

from refinery.config import Settings
from refinery.extraction import VisionExtractor


def _vision(tmp_path: Path) -> VisionExtractor:
    settings = Settings(workspace_root=tmp_path)
    return VisionExtractor(settings)


def test_scanned_quality_score_is_high_for_clean_ocr(tmp_path: Path):
    vision = _vision(tmp_path)
    metrics = {
        "ocr_word_conf_mean": 82.0,
        "ocr_word_conf_p10": 45.0,
        "garbage_ratio": 0.08,
        "avg_word_len": 4.6,
        "token_diversity": 0.62,
    }

    score = vision._quality_score_from_metrics(metrics, "scanned_image")
    assert score > 0.70


def test_scanned_quality_score_is_low_for_noisy_ocr(tmp_path: Path):
    vision = _vision(tmp_path)
    metrics = {
        "ocr_word_conf_mean": 47.0,
        "ocr_word_conf_p10": 18.0,
        "garbage_ratio": 0.40,
        "avg_word_len": 2.1,
        "token_diversity": 0.14,
    }

    score = vision._quality_score_from_metrics(metrics, "scanned_image")
    assert score < 0.60

from refinery.config import Settings
from refinery.triage import DomainClassification, DomainClassifier, TriageMetrics, classify_profile


class _AlwaysMedicalClassifier(DomainClassifier):
    def classify(self, doc_name: str, text_sample: str) -> DomainClassification:
        return DomainClassification(domain_hint="medical", confidence=0.91)


def test_triage_scanned_profile():
    metrics = TriageMetrics(
        page_count=4,
        avg_char_density=0.00003,
        image_area_ratio=0.85,
        whitespace_ratio=0.9,
        low_char_pages_fraction=0.8,
        estimated_columns=1,
        table_signal_ratio=0.0,
        text_sample="",
    )
    profile = classify_profile("scan.pdf", "abc", metrics)
    assert profile.origin_type == "scanned_image"
    assert profile.estimated_extraction_cost == "needs_vision_model"
    assert 0.0 <= profile.origin_confidence <= 1.0
    assert 0.0 <= profile.layout_confidence <= 1.0
    assert 0.0 <= profile.domain_confidence <= 1.0
    assert 0.0 <= profile.extraction_cost_confidence <= 1.0


def test_triage_table_heavy_profile():
    metrics = TriageMetrics(
        page_count=8,
        avg_char_density=0.0004,
        image_area_ratio=0.05,
        whitespace_ratio=0.3,
        low_char_pages_fraction=0.0,
        estimated_columns=1,
        table_signal_ratio=0.4,
        text_sample="Revenue report",
    )
    profile = classify_profile("financial_report.pdf", "abc", metrics)
    assert profile.layout_complexity == "table_heavy"
    assert profile.domain_hint == "financial"
    assert profile.domain_confidence > 0.5


def test_triage_multi_column_profile():
    metrics = TriageMetrics(
        page_count=3,
        avg_char_density=0.00055,
        image_area_ratio=0.02,
        whitespace_ratio=0.22,
        low_char_pages_fraction=0.0,
        estimated_columns=2,
        table_signal_ratio=0.0,
        text_sample="Contract terms",
    )
    profile = classify_profile("contract_terms.pdf", "abc", metrics)
    assert profile.layout_complexity == "multi_column"
    assert profile.estimated_extraction_cost in {"needs_layout_model", "fast_text_sufficient"}


def test_triage_zero_text_document_explicit_handling():
    metrics = TriageMetrics(
        page_count=5,
        avg_char_density=0.0,
        image_area_ratio=0.9,
        whitespace_ratio=1.0,
        low_char_pages_fraction=1.0,
        estimated_columns=1,
        table_signal_ratio=0.0,
        text_sample="",
        zero_text_document=True,
    )
    profile = classify_profile("scan_zero_text.pdf", "abc", metrics)
    assert profile.origin_type == "scanned_image"
    assert profile.domain_hint == "general"
    assert profile.language_hint.language == "unknown"
    assert profile.estimated_extraction_cost == "needs_vision_model"
    assert profile.domain_confidence < 0.4


def test_triage_mixed_mode_pages_can_drive_mixed_origin():
    metrics = TriageMetrics(
        page_count=6,
        avg_char_density=0.00035,
        image_area_ratio=0.15,
        whitespace_ratio=0.4,
        low_char_pages_fraction=0.05,
        estimated_columns=1,
        table_signal_ratio=0.1,
        text_sample="operations report",
        mixed_mode_pages_fraction=0.6,
    )
    profile = classify_profile("ops_report.pdf", "abc", metrics)
    assert profile.origin_type == "mixed"


def test_triage_thresholds_are_configuration_driven():
    metrics = TriageMetrics(
        page_count=4,
        avg_char_density=0.00019,
        image_area_ratio=0.22,
        whitespace_ratio=0.5,
        low_char_pages_fraction=0.1,
        estimated_columns=1,
        table_signal_ratio=0.1,
        text_sample="neutral text",
    )
    strict = Settings(
        triage_mixed_min_image_ratio=0.20,
        triage_mixed_max_char_density=0.0002,
    )
    relaxed = Settings(
        triage_mixed_min_image_ratio=0.30,
        triage_mixed_max_char_density=0.00015,
    )
    strict_profile = classify_profile("threshold.pdf", "abc", metrics, settings=strict)
    relaxed_profile = classify_profile("threshold.pdf", "abc", metrics, settings=relaxed)
    assert strict_profile.origin_type == "mixed"
    assert relaxed_profile.origin_type == "native_digital"


def test_triage_uses_pluggable_domain_classifier():
    metrics = TriageMetrics(
        page_count=2,
        avg_char_density=0.0005,
        image_area_ratio=0.01,
        whitespace_ratio=0.2,
        low_char_pages_fraction=0.0,
        estimated_columns=1,
        table_signal_ratio=0.0,
        text_sample="generic text without obvious domain words",
    )
    profile = classify_profile(
        "arbitrary.pdf",
        "abc",
        metrics,
        domain_classifier=_AlwaysMedicalClassifier(),
    )
    assert profile.domain_hint == "medical"
    assert profile.domain_confidence == 0.91

from refinery.triage import TriageMetrics, classify_profile


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

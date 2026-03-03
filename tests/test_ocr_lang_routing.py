from refinery.lang_detect import select_ocr_lang


def test_ocr_lang_routing_amharic_enabled():
    lang = select_ocr_lang("am", 0.9, ocr_amharic_enabled=True, ocr_lang_default="eng", ocr_lang_fallback="eng+amh")
    assert lang == "amh+eng"


def test_ocr_lang_routing_english():
    lang = select_ocr_lang("en", 0.8, ocr_amharic_enabled=True, ocr_lang_default="eng", ocr_lang_fallback="eng+amh")
    assert lang == "eng"


def test_ocr_lang_routing_unknown_fallback():
    lang = select_ocr_lang("unknown", 0.2, ocr_amharic_enabled=True, ocr_lang_default="eng", ocr_lang_fallback="eng+amh")
    assert lang == "eng+amh"

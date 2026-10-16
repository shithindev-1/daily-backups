"""The document-type registry and the focused (per-type) extraction it drives."""

from typing import Any

import pytest

from app.schemas.ocr import COMPANY_EXTRACTION_FIELDS, INDIVIDUAL_EXTRACTION_FIELDS, VERIFIABLE_TYPES
from app.services import document_registry as registry
from app.services.gemini import prompts


def test_every_supported_type_has_an_entry() -> None:
    assert set(registry.REGISTRY) == set(VERIFIABLE_TYPES)


@pytest.mark.parametrize("document_type", sorted(VERIFIABLE_TYPES))
@pytest.mark.parametrize("company", [False, True])
def test_every_field_is_known_and_defined(document_type: str, company: bool) -> None:
    fields = registry.extraction_fields(document_type, company)
    known = COMPANY_EXTRACTION_FIELDS if company else INDIVIDUAL_EXTRACTION_FIELDS
    assert fields and set(fields) <= set(known)
    rules = registry.field_rules(document_type, company)
    definitions = registry.COMPANY_FIELD_RULES if company else registry.INDIVIDUAL_FIELD_RULES
    for field in fields:
        assert field in definitions or field in registry._ADDRESS_FIELDS, field
        assert field in rules or field in registry._ADDRESS_FIELDS


def test_a_focused_instruction_asks_only_for_its_own_fields() -> None:
    instruction, fields = prompts.focused_system_instruction("passport", company=False)
    assert "Passport_Number: <value or null>" in instruction
    assert "Visa_Number: <value or null>" not in instruction and "National_Id: <value or null>" not in instruction
    assert "Document_Types_Present" in instruction and "may be wrong" in instruction
    assert prompts.focused_system_instruction("other", company=False) is None


def test_licence_entries_read_members_for_an_individual_and_company_details_for_a_company() -> None:
    assert registry.extraction_fields("trade_license", company=False) == ["License_Members"]
    company = registry.extraction_fields("initial_approval", company=True)
    assert "Initial_Approval_Number" in company and "Trade_License_Number" not in company


def test_required_details_and_link_numbers_come_from_the_registry() -> None:
    assert "two lines of letters and numbers" in registry.required_details("passport")
    assert "light-grey" in prompts.covered_parts_rules("passport") and "passport number" in prompts.covered_parts_rules("passport")
    assert registry.link_numbers("visa", company=False) == ("Visa_Number", "Passport_Number", "National_Id")
    assert registry.link_numbers("trade_license", company=True) == ("Trade_License_Number",)


def test_detect_endpoints_read_focused_and_old_endpoints_do_not(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from app.core.config import settings
    from app.main import app
    from app.schemas.ocr import DATA_FIELDS
    from app.services import blob_storage
    from app.services.gemini import get_gemini_service

    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: None)
    calls: list[dict[str, Any]] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"data": dict.fromkeys(DATA_FIELDS), "documentComplete": True, "documentTypesPresent": ["national_id"],
                "frontVisible": True, "backVisible": True}

    async def fake_segment(pages: list[bytes]) -> list[dict[str, Any]]:
        return [{"regions": [{"document_type": "national_id", "side": "single", "document_number": 1,
                              "box_2d": [0, 0, 1000, 1000], "covered_by": []}]}]

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "segment_documents", fake_segment)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), "white").save(buffer, format="PNG")
    png = buffer.getvalue()
    client = TestClient(app)

    client.post("/api/v1/ocr/leasing/detect/upload", files=[("file", ("eid.png", png, "image/png"))],
                data={"tenant_type": "individual", "document_refnumber": "r1"})
    client.post("/api/v1/ocr/leasing/upload", files=[("file", ("eid.png", png, "image/png"))],
                data={"tenant_type": "individual", "document_refnumber": "r2", "document_id": "1",
                      "document_type": "national_id", "document_name": "nid"})

    assert [call.get("focused", False) for call in calls] == [True, False]


def test_a_place_field_with_no_letters_is_dropped_from_a_focused_reading() -> None:
    from app.services.ocr_pipeline import drop_numeric_places

    data = {"Passport_Issuing_Location": "3", "Nationality": "EGYPTIAN", "Home_City": "12-34",
            "Passport_Number": "A28334482", "Resident_City": "Dubai"}
    assert drop_numeric_places(data) == {"Passport_Issuing_Location": None, "Nationality": "EGYPTIAN",
                                         "Home_City": None, "Passport_Number": "A28334482", "Resident_City": "Dubai"}


def test_issuing_location_is_defined_as_a_place_never_an_office_code() -> None:
    rules = registry.field_rules("passport", company=False)
    assert "never a place" in rules and "Issuing Office" in rules
    assert "never the value of a name, place, nationality or date field" in rules


def test_decoding_settings_are_sent_on_every_call() -> None:
    from app.services.gemini import get_gemini_service

    config = get_gemini_service()._generation_config()
    assert (config.temperature, config.seed, config.top_p, config.top_k) == (0.0, 0, 1.0, 1)

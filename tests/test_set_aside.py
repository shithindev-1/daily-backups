"""Copies a tenant does not need are set aside (type-detection): `ignored` with a reason and a note,
never an error - and only for a copy PROVEN to be the same holder's. Plus the document-results log."""

import json
import logging
from typing import Any

import pytest

from app.core.config import settings
from app.services.leasing.outcomes import mask_numbers, without_private_keys
from app.services.leasing.tenant_results import _set_aside_copies
from app.services.type_detection.tenants import Tenant, Unit, copy_proof

ANUP = {"Tenant_Name_En": "Anupkumar Krishnan", "Date_Of_Birth": "22/12/1978"}
EXPIRED = "Your National ID expired on 11-06-2025. Please upload a valid National ID."


def _units(*specs: tuple[str, dict[str, Any]]) -> list[Unit]:
    return [Unit(key=(index,), document_type=document_type, data=data) for index, (document_type, data) in enumerate(specs)]


def _run(units: list[Unit], tenants: list[Tenant], errors: list[str | None], *, shas: list[str] | None = None,
         company: bool = False) -> tuple[list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    reports = [{"error": error} for error in errors]
    items = [{"document_name": f"Document {index + 1}"} for index in range(len(units))]
    sha_by_key = {unit.key: ((shas or [])[unit.key[0]] if shas else f"sha{unit.key[0]}",) for unit in units}
    aside = _set_aside_copies(tenants, units, items, reports, sha_by_key, company)
    return reports, aside


def test_an_expired_copy_of_the_same_card_is_set_aside_without_an_error() -> None:
    units = _units(
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "03/07/2027", **ANUP}),
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "11/06/2025", **ANUP}),
    )
    tenant = Tenant(keys=[(0,), (1,)])
    reports, _ = _run(units, [tenant], [None, EXPIRED])

    old = reports[1]
    assert (old["error"], old["ignored"], old["ignored_reason"]) == (None, True, "expired_copy")
    assert old["note"] == "An older National ID that has expired. The valid one ('Document 1') is used."
    assert old["_detail"]["proof"] == "same_document_number"
    assert old["_detail"]["replaced_error"]
    assert tenant.keys == [(0,)]


def test_a_failed_copy_that_is_not_expired_is_replaced_by_the_valid_one() -> None:
    # The old card's back is missing (not an expiry problem): the valid new card is used.
    units = _units(
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "03/07/2027", **ANUP}),
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "03/07/2027", **ANUP}),
    )
    reports, _ = _run(units, [Tenant(keys=[(0,), (1,)])], [None, "Please upload the back of your National ID."])
    assert (reports[1]["error"], reports[1]["ignored_reason"]) == (None, "replaced_copy")


@pytest.mark.parametrize("new_expiry", ["01/01/2035", None])
def test_a_failed_copy_that_may_be_the_newer_one_keeps_its_error(new_expiry: str | None) -> None:
    # An old valid passport and a blurry new one (expiry later, or unreadable): the new one's problem is shown.
    units = _units(
        ("passport", {"Passport_Number": "Z1111111", "Passport_Expiry": "01/01/2027", **ANUP}),
        ("passport", {"Passport_Number": "Z2222222", "Passport_Expiry": new_expiry, **ANUP}),
    )
    blurry = "Your passport is too blurry to read. Please upload a clearer copy."
    reports, aside = _run(units, [Tenant(keys=[(0,), (1,)])], [None, blurry])
    assert not aside
    assert reports[1]["error"] == blurry


def test_the_same_card_uploaded_twice_is_a_duplicate() -> None:
    data = {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "03/07/2027", **ANUP}
    units = _units(("national_id", data), ("national_id", dict(data)))
    tenant = Tenant(keys=[(0,), (1,)])
    reports, _ = _run(units, [tenant], [None, None])
    assert reports[0].get("ignored") is None
    assert (reports[1]["ignored_reason"], reports[1]["error"]) == ("duplicate", None)
    assert tenant.keys == [(0,)]


def test_a_renewed_passport_replaces_the_old_one_even_when_both_are_valid() -> None:
    units = _units(
        ("passport", {"Passport_Number": "Z1111111", "Passport_Expiry": "01/01/2027", **ANUP}),
        ("passport", {"Passport_Number": "Z2222222", "Passport_Expiry": "01/01/2035", **ANUP}),
    )
    reports, _ = _run(units, [Tenant(keys=[(0,), (1,)])], [None, None])
    # The one that expires last is used.
    assert reports[0]["ignored_reason"] == "replaced_copy"
    assert reports[0]["_detail"]["proof"] == "same_name_and_birth"
    assert reports[1].get("ignored") is None


def test_twins_are_never_taken_for_the_same_person() -> None:
    # Grouped together (same birth date, a shared name word) - but different ID numbers and names.
    units = _units(
        ("national_id", {"National_Id": "784-1990-1111111-1", "National_Id_Expiry": "01/01/2030",
                         "Tenant_Name_En": "Ahmed Ali Hassan", "Date_Of_Birth": "01/03/1990"}),
        ("national_id", {"National_Id": "784-1990-2222222-2", "National_Id_Expiry": "01/01/2024",
                         "Tenant_Name_En": "Mohammed Ali Hassan", "Date_Of_Birth": "01/03/1990"}),
    )
    reports, aside = _run(units, [Tenant(keys=[(0,), (1,)])], [None, EXPIRED])
    assert not aside
    assert reports[1]["error"] == EXPIRED


def test_another_tenants_valid_card_never_covers_for_an_expired_one() -> None:
    units = _units(
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "11/06/2025", **ANUP}),
        ("national_id", {"National_Id": "784-1985-1234567-1", "National_Id_Expiry": "01/01/2030",
                         "Tenant_Name_En": "Sara Ali", "Date_Of_Birth": "15/07/1985"}),
    )
    reports, aside = _run(units, [Tenant(keys=[(0,)]), Tenant(keys=[(1,)])], [EXPIRED, None])
    assert not aside
    assert reports[0]["error"] == EXPIRED


def test_when_every_copy_failed_only_the_newest_keeps_its_error() -> None:
    units = _units(
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "11/06/2023", **ANUP}),
        ("national_id", {"National_Id": "784-1978-2940540-2", "National_Id_Expiry": "11/06/2025", **ANUP}),
    )
    reports, _ = _run(units, [Tenant(keys=[(0,), (1,)])], [EXPIRED, EXPIRED])
    assert (reports[0]["ignored_reason"], reports[0]["error"]) == ("older_copy", None)
    assert reports[0]["note"] == "An older National ID. Please fix the newer one ('Document 2')."
    assert reports[1]["error"] == EXPIRED


def test_copies_with_unreadable_expiry_all_keep_their_errors_when_none_is_valid() -> None:
    units = _units(("national_id", {"National_Id": "784-1978-2940540-2", **ANUP}),
                   ("national_id", {"National_Id": "784-1978-2940540-2", **ANUP}))
    _, aside = _run(units, [Tenant(keys=[(0,), (1,)])], ["Unreadable.", "Unreadable."])
    assert not aside


def test_bank_statements_are_only_set_aside_when_the_same_file_comes_twice() -> None:
    units = _units(("bank_statement", {"Tenant_Name_En": "Anupkumar Krishnan"}),
                   ("bank_statement", {"Tenant_Name_En": "Anupkumar Krishnan"}),
                   ("bank_statement", {"Tenant_Name_En": "Anupkumar Krishnan"}))
    reports, _ = _run(units, [Tenant(keys=[(0,), (1,), (2,)])], [None, None, None], shas=["a", "b", "a"])
    assert [report.get("ignored_reason") for report in reports] == [None, None, "duplicate"]


def test_a_signatory_shared_by_two_company_tenants_is_judged_once() -> None:
    signatory = {"Signatory_Name_En": "Bhargava Subhash Kalluri", "Signatory_National_Id": "784-1980-5555555-5"}
    units = _units(
        ("trade_license", {"Trade_License_Number": "111"}),
        ("trade_license", {"Trade_License_Number": "222"}),
        ("national_id", {**signatory, "Signatory_National_Id_Expiry": "01/01/2030"}),
        ("national_id", {**signatory, "Signatory_National_Id_Expiry": "01/01/2024"}),
    )
    first = Tenant(keys=[(0,), (2,), (3,)], company_keys=[(0,)], signatory_keys=[(2,), (3,)])
    second = Tenant(keys=[(1,), (2,), (3,)], company_keys=[(1,)], signatory_keys=[(2,), (3,)])
    reports, aside = _run(units, [first, second], [None, None, None, EXPIRED], company=True)
    assert list(aside) == [(3,)]
    assert reports[3]["ignored_reason"] == "expired_copy"
    assert first.signatory_keys == second.signatory_keys == [(2,)]


def test_licences_of_one_company_are_copies_two_companies_are_not() -> None:
    old, new, other = _units(
        ("trade_license", {"Trade_License_Number": "2527004.01", "Company_Name_En": "Al Frejun LLC"}),
        ("commercial_license", {"Trade_License_Number": "2527004.01", "Company_Name_En": "Al Frejun LLC"}),
        ("trade_license", {"Trade_License_Number": "999", "Company_Name_En": "Beta Foods LLC"}),
    )
    assert copy_proof(old, new, company=True) == "same_document_number"
    assert copy_proof(old, other, company=True) is None


def test_long_numbers_are_masked_but_dates_are_kept() -> None:
    text = "The ID number 784-1978-2940540-2 on 'Visa' differs; it expired on 28-04-2024."
    assert mask_numbers(text) == f"The ID number {'*' * 14}40-2 on 'Visa' differs; it expired on 28-04-2024."


def test_private_keys_never_reach_a_caller() -> None:
    assert without_private_keys([{"error": None, "_stage": "x", "_detail": {}}]) == [{"error": None}]


def test_the_old_endpoint_response_is_unchanged_and_its_outcome_is_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services import blob_storage
    from tests.test_leasing_endpoint import _answer_by_filename, _raw, _upload

    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: None)
    _answer_by_filename(monkeypatch, {"eid_front.png": _raw(["national_id"], complete=False, front=True, back=False)})

    with caplog.at_level(logging.INFO, logger="uae_ocr.document_results"):
        body = _upload(TestClient(app), [("eid_front.png", "1", "National Id", "nid front")]).json()

    [document] = body["documents"]
    assert set(document) == {"document_type", "document_type_is_correct", "merged", "sources", "error"}
    [line] = [json.loads(record.getMessage()) for record in caplog.records if record.name == "uae_ocr.document_results"]
    assert (line["endpoint"], line["status"]) == ("leasing", "failed")
    assert line["documents"][0]["status"] == "invalid"
    assert line["documents"][0]["step"]
    assert line["documents"][0]["files"] == [{"file": "eid_front.png", "pages": [1]}]

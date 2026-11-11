"""Why each document ended as it did: the step that failed it, and logs/document-results.log.

Every report entry can carry two private keys while a request runs - never returned to a caller:
  _stage   the step whose check failed the document (see STEPS)
  _detail  extra facts about that outcome (for a copy set aside: which copy was kept and why)

At the end of a request `log_document_results` writes ONE JSON object: the request's status, each
tenant's status, and per document its type, files, status, failing step, the message the caller got
(long numbers masked; off with LOG_ERROR_MESSAGES=false) and the details - so a failed request can be
explained from the log alone.
"""

import json
import logging
import re
from typing import Any

from app.core.config import settings
from app.core.logging import log_timestamps

results_logger = logging.getLogger("uae_ocr.document_results")
logger = logging.getLogger("uae_ocr.leasing")

# The steps a document can fail at, in pipeline order.
STEPS = {
    "download": "the file could not be downloaded",
    "file_check": "the file type, size or content was not accepted",
    "segmentation": "the file could not be split into its documents",
    "unsupported_document": "the document is not a supported type",
    "extraction": "the document could not be read",
    "type_check": "the second reading did not confirm the document type",
    "pairing": "its front/back or other pages could not be matched",
    "covered_by_other_document": "most of it is hidden under another document",
    "self_verification": "its own check failed (expired, unreadable, incomplete, covered ...)",
    "tenant_match": "it could not be tied to any tenant",
    "not_listed_on_licence": "the person is not listed on any uploaded licence or approval",
    "no_signatory_uploaded": "nobody the licence lists was uploaded",
    "no_national_id": "the person has no National ID in the request",
    "cross_verification": "it disagrees with another document of the same tenant",
    "crop_rotate_upload": "its processed copy could not be saved",
    "set_aside": "another copy of the same document is used instead (ignored, not an error)",
}

# A date (kept - it is what most messages are about) or a long number (masked).
_DATE_OR_NUMBER = re.compile(r"(?P<date>\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b)|(?P<number>\d[\d -]{4,}\d)")


def tag_errors(documents_report: list[dict[str, Any]], stage: str) -> None:
    """Marks every document that has an error but no failing step yet as failed at `stage`."""
    for report in documents_report:
        if report.get("error") and not report.get("_stage"):
            report["_stage"] = stage


def mask_numbers(text: str | None) -> str | None:
    """"784-1978-2940540-2" -> "**************40-2": long numbers keep their last 4 characters only;
    dates ("28-04-2024") are kept."""
    if not text:
        return text

    def masked(match: re.Match[str]) -> str:
        if match.group("date"):
            return match.group()
        return "*" * (len(match.group()) - 4) + match.group()[-4:]

    return _DATE_OR_NUMBER.sub(masked, text)


def message_for_log(text: str | None) -> str | None:
    return mask_numbers(text) if settings.log_error_messages else None


def without_private_keys(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report entries as a caller may see them: the private _stage/_detail keys removed."""
    return [{key: value for key, value in document.items() if not key.startswith("_")} for document in documents]


def outcome(
    *, number: int, names: list[str], document_type: str, status: str, files: list[dict[str, Any]],
    report: dict[str, Any], error: str | None, tenant_ids: list[str] | None = None,
) -> dict[str, Any]:
    """One document's line in document-results.log."""
    entry: dict[str, Any] = {
        "document": number,
        "label": " + ".join(names),
        "document_type": document_type,
        "status": status,
        "files": files,
    }
    if tenant_ids is not None:
        entry["tenant_ids"] = tenant_ids
    stage = report.get("_stage") or ("set_aside" if report.get("ignored") else None)
    if status != "valid" and stage:
        entry["step"] = stage
        entry["why"] = STEPS.get(stage, stage)
    if report.get("ignored_reason"):
        entry["ignored_reason"] = report["ignored_reason"]
    if error:
        entry["message"] = message_for_log(error)
    if report.get("_detail"):
        entry["detail"] = report["_detail"]
    return entry


def log_document_results(
    *, request_id: str, endpoint: str, tenant_type: str, status: str, documents: list[dict[str, Any]],
    tenants: list[dict[str, Any]] | None = None, reason: str | None = None,
) -> None:
    """Writes the request's line to document-results.log, and a short summary to ocr-api.log."""
    counts: dict[str, int] = {}
    for document in documents:
        counts[document["status"]] = counts.get(document["status"], 0) + 1
    record: dict[str, Any] = {
        **log_timestamps(),
        "request_id": request_id,
        "endpoint": endpoint,
        "tenant_type": tenant_type,
        "status": status,
    }
    if reason:
        record["reason"] = message_for_log(reason)
    if tenants is not None:
        record["tenants"] = {tenant["tenant_id"]: tenant["status"] for tenant in tenants}
    record["counts"] = counts
    record["documents"] = documents
    try:
        results_logger.info(json.dumps(record, ensure_ascii=False, default=str))
    except Exception:  # a log line must never fail a request
        logger.exception("Could not write document-results.log request_id=%s", request_id)
    failed = [
        f"{document['document']}:{document['document_type']}:{document.get('step', '?')}"
        for document in documents if document["status"] == "invalid"
    ]
    ignored = [
        f"{document['document']}:{document['document_type']}:{document.get('ignored_reason', '?')}"
        for document in documents if document["status"] == "ignored"
    ]
    logger.info(
        "Document results request_id=%s endpoint=%s status=%s counts=%s failed=%s ignored=%s",
        request_id, endpoint, status, counts, failed, ignored,
    )

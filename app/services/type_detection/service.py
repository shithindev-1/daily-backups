"""Running a type-detection request: split each file into its documents, run them through the
leasing batch with the types found, and build the response."""

import posixpath
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.schemas.leasing import SourceDocument
from app.services import blob_storage
from app.services.concurrency import run_concurrently
from app.services.gemini import (
    get_gemini_service,
    record_stage,
    share_usage,
    start_usage_tracking,
    timed_step,
    usage_for,
)
from app.services.leasing import sources as source_fetch
from app.services.leasing.batch import process_batch
from app.services.leasing.outcomes import log_document_results, outcome
from app.services.leasing.rules import require_mandatory_document
from app.services.leasing.sources import safe_name
from app.services.ocr_pipeline import VerificationUnavailable, call_with_one_retry, log_request_usage
from app.services.post_processing import document_label
from app.services.type_detection.responses import build_detect_response, document_entry
from app.services.type_detection.segmentation import OTHER_TYPE, Segment, parse_side, segment_file
from app.services.validation import ALLOWED_FILES, validate_document_request


def _file_name(item: dict[str, Any]) -> str:
    return item.get("filename") or posixpath.basename(urlparse(item["source"]).path) or item["source"]


def _file_usage_name(item: dict[str, Any]) -> str:
    """The name a file's split call is charged under until its documents are known."""
    return f"file {_file_name(item)} #{id(item)}"


async def _segment_one(item: dict[str, Any], request_id: str) -> None:
    """Validates one file and splits it into its documents, in `item["segments"]` - or sets
    `item["error"]`."""
    if item.get("error"):
        return
    upload = SimpleNamespace(filename=item["filename"], content_type=item["content_type"])
    try:
        validated = validate_document_request(
            file=upload, content=item["content"], document_name=safe_name(item["filename"]),
            document_type=OTHER_TYPE, settings=settings,
        )
    except HTTPException as exc:
        item["error"] = f"'{_file_name(item)}': {exc.detail}"
        item["_stage"] = "file_check"
        return
    item["mime_type"] = validated.mime_type
    try:
        # Charged to the file until its documents are known - then shared among them (process_detected).
        with usage_for((_file_usage_name(item), "")), timed_step("segmentation"):
            item["segments"] = await call_with_one_retry(
                lambda: segment_file(
                    content=item["content"], mime_type=validated.mime_type, sha256=validated.sha256,
                    settings=settings, gemini_service=get_gemini_service(),
                ),
                what="Segmentation", request_id=request_id,
            )
    except VerificationUnavailable:
        item["error"] = (
            f"We could not read '{_file_name(item)}' right now. Please try again in a few minutes."
        )
        item["_stage"] = "segmentation"


def _labels(found: list[tuple[dict[str, Any], Segment]]) -> list[str]:
    """"Passport"; numbered ("National ID 1", "National ID 2") when a type occurs more than once.
    Only the found type - never the file's own name, which may say anything."""
    totals: dict[str, int] = {}
    for _, segment in found:
        totals[segment.document_type] = totals.get(segment.document_type, 0) + 1
    seen: dict[str, int] = {}
    labels = []
    for item, segment in found:
        label = document_label(segment.document_type)
        label = label[:1].upper() + label[1:]
        if totals[segment.document_type] > 1:
            seen[segment.document_type] = seen.get(segment.document_type, 0) + 1
            label = f"{label} {seen[segment.document_type]}"
        labels.append(label)
    return labels


def _unsupported_error(segment: Segment, file_name: str, shared: bool) -> str:
    """The error for a document outside the supported types: what it is, where, and that it is not
    listed - "The DEWA utility bill found on page 2 of 'docs.pdf' is not listed."."""
    name = segment.detected_type
    # "Driving licence" -> "driving licence"; an acronym ("UAE driving licence", "DEWA bill") is kept.
    if name and not (len(name) > 1 and name[1].isupper()):
        name = name[:1].lower() + name[1:]
    where = f"found on {_pages_text(segment.pages)} of '{file_name}'" if shared else f"in '{file_name}'"
    return f"The {name or 'document'} {where} is not listed."


# A side of a document (or a one-sided document) more than this share covered by another document is
# rejected outright - measured in code from the split boxes, so the same photo always gets the same answer.
_MAX_COVERED_SHARE = 0.5


def _covered_error(segment: Segment) -> str | None:
    """The message for a document with a side mostly hidden under another document, or None."""
    for region, share in zip(segment.regions, segment.covered):
        if share > _MAX_COVERED_SHARE:
            kind = document_label(segment.document_type)
            side = parse_side(region.label)
            whose = f"The {side} of your {kind}" if side in ("front", "back") else f"Your {kind}"
            return f"{whose} is mostly covered by another document. Please upload a photo with nothing on top of it."
    return None


def _batch_item(item: dict[str, Any], segment: Segment, label: str, number: int) -> dict[str, Any]:
    """What `process_batch` takes for one found document: the original file when it holds only this
    document, otherwise the document's own cut-out file."""
    document_id = f"{number}_{segment.document_type}"
    batch_item = {
        "source": item["source"], "document_id": document_id,
        "document_type": segment.document_type, "document_name": label,
    }
    if segment.content is None:
        batch_item.update(content=item["content"], filename=item["filename"], content_type=item["content_type"])
        if item.get("blob_path"):
            batch_item["blob_path"] = item["blob_path"]
        return batch_item

    # Already cut to its own edges, with other documents blanked out: the final crop takes it whole.
    batch_item["precut"] = True
    if error := _covered_error(segment):
        batch_item["covered_error"] = error
    stem = Path(item["filename"]).stem or "document"
    filename = f"{safe_name(stem)}_{document_id}{segment.extension}"
    batch_item.update(content=segment.content, filename=filename, content_type=ALLOWED_FILES[segment.extension])
    if item.get("blob_path"):
        # Where its processed copy goes: beside the original, named after its type at upload.
        batch_item["source_blob_path"] = item["blob_path"]
        folder = posixpath.dirname(item["blob_path"])
        batch_item["blob_path"] = posixpath.join(folder, filename) if folder else filename
    return batch_item


def _pages_text(pages: list[int]) -> str:
    if len(pages) == 1:
        return f"page {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pages {pages[0]}-{pages[-1]}"
    return "pages " + ", ".join(str(page) for page in pages)


def locate_error(error: str | None, document_type: str, places: list[tuple[str, list[int]]]) -> str | None:
    """Says where a document was found when its file holds several documents, so a message about a
    visa inside "Trade License 11.pdf" reads as being about that visa, not about the licence.

    `places` are (file name, pages) of the sources that sit in a file with other documents; empty
    means every source is a file of its own and the message is returned unchanged. A message starting
    "Your <document>" is reworded to "The <document> found on page N of '<file>'"; any other message
    gets that location in front.
    """
    if not error or not places:
        return error
    plain = document_label(document_type)  # "visa", "National ID", "trade licence"
    label = plain[:1].upper() + plain[1:]
    where = " and ".join(f"{_pages_text(pages)} of '{file_name}'" for file_name, pages in places if pages) or         " and ".join(f"'{file_name}'" for file_name, _ in places)
    lead = f"your {plain}".lower()
    if error.lower().startswith(lead):
        return f"The {plain} found on {where}{error[len(lead):]}"
    return f"{label} found on {where}: {error}"


def _where(names: list[str], origin: dict[str, tuple[dict[str, Any], Segment]], shared_files: set[int]) -> str:
    """Where the user uploaded a document, in their own file names: "in 'EID-Front.jpeg' and
    'EID-back.jpeg'", or "on page 2 of 'Documents.pdf'" when the file holds other documents too."""
    by_file: dict[str, tuple[bool, list[int]]] = {}
    for name in names:
        item, segment = origin[name]
        shared, pages = by_file.get(_file_name(item), (False, []))
        by_file[_file_name(item)] = (shared or id(item) in shared_files, pages + list(segment.pages))
    if not any(shared for shared, _ in by_file.values()):
        return "in " + " and ".join(f"'{file_name}'" for file_name in by_file)
    return " and ".join(
        f"on {_pages_text(sorted(set(pages)))} of '{file_name}'" if shared and pages else f"in '{file_name}'"
        for file_name, (shared, pages) in by_file.items()
    )


def _ignored_note(reason: str, document_type: str, where: str, *, original_failed: bool = False) -> str:
    """The plain-English note on a copy set aside, naming the file of the copy that is used."""
    label = document_label(document_type)
    if reason == "duplicate" and original_failed:
        return f"This {label} was uploaded more than once. Please fix the {label} {where}."
    return {
        "expired_copy": f"An older {label} that has expired. The valid {label} {where} is used.",
        "replaced_copy": f"Another copy of this {label} was uploaded. The {label} {where} is used.",
        "duplicate": f"This {label} was uploaded more than once. The one {where} is used.",
        "older_copy": f"An older {label}. Please fix the {label} {where}.",
    }[reason]


def _log_files(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """A document's files for document-results.log: file names and pages only, never a link."""
    return [{"file": source["file_name"], "pages": source["pages"]} for source in entry["sources"]]


def _source_row(item: dict[str, Any], pages: list[int], ocr_file: str | None = None) -> dict[str, Any]:
    return {"source": item["source"], "file_name": _file_name(item), "pages": pages, "ocr_file": ocr_file}


async def process_detected(
    *, tenant_type: str, document_refnumber: str, items: list[dict[str, Any]], started_at: float | None = None,
) -> dict[str, Any]:
    """Each item is source, and either content/filename/content_type (plus blob_path when it came
    from blob storage) or the `error` its download failed with."""
    started_at = time.perf_counter() if started_at is None else started_at
    if not items:
        raise HTTPException(status_code=400, detail="Please add at least one document.")

    # One token tally for the whole request, started before the split calls so they are counted
    # too; the batch below adds to it instead of starting a new one.
    start_usage_tracking(started_at)
    # Until now: receiving the upload, or downloading the linked files (timing.log).
    record_stage("receive", time.perf_counter() - started_at)
    split_started = time.perf_counter()
    await run_concurrently([partial(_segment_one, item, document_refnumber) for item in items])
    record_stage("segmentation", time.perf_counter() - split_started)

    def finish(status: str, documents: list[dict[str, Any]], tenants: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return build_detect_response(
            status=status, tenant_type=tenant_type, document_refnumber=document_refnumber,
            started_at=started_at, documents=documents, tenants=tenants,
        )

    # A file that could not be read at all leaves nothing to judge the request by.
    if any(item.get("error") for item in items):
        log_request_usage(document_refnumber, "failed", settings.detection_enabled)
        entries = [
            document_entry(
                document_type="unknown" if item.get("error") else "not_checked",
                status="invalid" if item.get("error") else "not_checked",
                sources=[_source_row(item, [])],
                error=item.get("error"),
            )
            for item in items
        ]
        log_document_results(
            request_id=document_refnumber, endpoint="detect", tenant_type=tenant_type, status="failed",
            documents=[
                outcome(number=number, names=[_file_name(item)], document_type=entry["document_type"],
                        status=entry["status"], files=_log_files(entry), error=entry["error"],
                        report={"_stage": item.get("_stage") or ("download" if item.get("error") else None)})
                for number, (item, entry) in enumerate(zip(items, entries), start=1)
            ],
        )
        return finish("failed", entries)

    found = [(item, segment) for item in items for segment in item["segments"]]
    supported = [(item, segment) for item, segment in found if segment.document_type != OTHER_TYPE]
    # Files that hold more than one document: their documents' messages say where each was found.
    shared_files = {id(item) for item in items if len(item["segments"]) > 1}
    # A document outside the supported types is never read: it is rejected, by its name, on its own -
    # it fails the request, not the tenant whose documents are still checked and extracted.
    unsupported = [
        document_entry(
            document_type=OTHER_TYPE, status="invalid", sources=[_source_row(item, segment.pages)],
            error=_unsupported_error(segment, _file_name(item), id(item) in shared_files),
        )
        for item, segment in found if segment.document_type == OTHER_TYPE
    ]

    # A file's split call is shared among the documents found in it (document-usage.log).
    found_labels = dict(zip(map(id, (segment for _, segment in supported)), _labels(supported)))
    for item in items:
        share_usage(_file_usage_name(item), [
            (found_labels[id(segment)], segment.document_type) if id(segment) in found_labels
            else (f"{segment.detected_type or 'Other document'} (not listed)", segment.document_type)
            for segment in item.get("segments") or []
        ])

    # Same rule and message as the normal endpoints, judged on the types that were found. Each
    # tenant's own mandatory document is checked once the documents are grouped into tenants.
    try:
        require_mandatory_document(
            tenant_type, [segment.document_type for _, segment in supported], accept_initial_approval=True,
        )
    except HTTPException as exc:
        # Rejected after the split calls were already made - their tokens are still logged.
        log_request_usage(document_refnumber, "rejected", settings.detection_enabled)
        log_document_results(
            request_id=document_refnumber, endpoint="detect", tenant_type=tenant_type, status="rejected",
            reason=str(exc.detail),
            documents=[
                outcome(number=number, names=[_file_name(item)], document_type=segment.document_type,
                        status="not_checked", files=[{"file": _file_name(item), "pages": segment.pages}],
                        report={}, error=None)
                for number, (item, segment) in enumerate(found, start=1)
            ],
        )
        raise

    labels = _labels(supported)
    origin = {label: (item, segment) for label, (item, segment) in zip(labels, supported)}
    batch_items = [
        _batch_item(item, segment, label, number)
        for number, (label, (item, segment)) in enumerate(zip(labels, supported), start=1)
    ]
    result = await process_batch(
        tenant_type=tenant_type, document_refnumber=document_refnumber, items=batch_items,
        started_at=started_at, detected_types=True,
    )

    by_name = result.get("tenant_ids_by_name") or {}
    documents = []
    reports = result["documents"]
    for report in reports:
        if report.get("ignored") and report.get("_kept_names") and report.get("ignored_reason"):
            where = _where(report["_kept_names"], origin, shared_files)
            report["note"] = _ignored_note(report["ignored_reason"], report["document_type"], where,
                                           original_failed=bool(report.get("_original_failed")))
            if report.get("_detail"):
                report["_detail"]["kept_where"] = where
        names = [source["document_name"] for source in report["sources"]]
        places = [
            (_file_name(origin[name][0]), origin[name][1].pages)
            for name in names if id(origin[name][0]) in shared_files
        ]
        documents.append(document_entry(
            tenant_ids=sorted({tenant_id for name in names for tenant_id in by_name.get(name, [])},
                              key=lambda tenant_id: int(tenant_id.rsplit("_", 1)[1])),
            document_type=report["document_type"],
            # A copy set aside for another (tenant_results) is reported as ignored - with no error.
            status="ignored" if report.get("ignored") else "invalid" if report["error"] else "valid",
            merged=report["merged"],
            sources=[
                _source_row(origin[name][0], origin[name][1].pages, source["ocr_file"])
                for name, source in zip(names, report["sources"])
            ],
            error=None if report.get("ignored") else locate_error(report["error"], report["document_type"], places),
            ignored_reason=report.get("ignored_reason"),
            note=report.get("note"),
        ))
    status = "failed" if unsupported else result["status"]
    all_documents = documents + unsupported
    log_document_results(
        request_id=document_refnumber, endpoint="detect", tenant_type=tenant_type, status=status,
        tenants=result.get("tenants") or [],
        documents=[
            outcome(
                number=number,
                names=([source["document_name"] for source in report["sources"]] if report is not None
                       else [entry["document_type"]]),
                document_type=entry["document_type"], status=entry["status"], files=_log_files(entry),
                report=report if report is not None else {"_stage": "unsupported_document"},
                error=entry["error"], tenant_ids=entry["tenant_ids"],
            )
            for number, (entry, report) in enumerate(
                zip(all_documents, list(reports) + [None] * len(unsupported)), start=1,
            )
        ],
    )
    return finish(status, all_documents, tenants=result.get("tenants") or [])


async def process_detected_sources(
    *, tenant_type: str, document_refnumber: str, sources: list[str],
) -> dict[str, Any]:
    """Downloads every link (blob path or allowed URL), then runs `process_detected`."""
    started_at = time.perf_counter()
    storage = blob_storage.get_blob_storage()
    items: list[dict[str, Any]] = [{"source": source} for source in sources]

    async def fetch(client: httpx.AsyncClient, item: dict[str, Any]) -> None:
        name = posixpath.basename(urlparse(item["source"]).path) or "document"
        # The download helpers take a SourceDocument; only its source and name are used.
        source = SourceDocument(source=item["source"], document_id="-", document_type="-", document_name=name)
        blob_path = storage.blob_path(item["source"]) if storage else None
        try:
            if blob_path:
                content, filename, content_type = await source_fetch.download_blob(storage, blob_path, source)
                item["blob_path"] = blob_path
            elif not item["source"].lower().startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{item['source']}' is not a valid file link or file path. Please check it and try again.",
                )
            else:
                content, filename, content_type = await source_fetch.download_source(client, source)
        except HTTPException as exc:
            item["error"] = str(exc.detail)
            return
        item.update(content=content, filename=filename, content_type=content_type)

    async with httpx.AsyncClient() as client:
        await run_concurrently([partial(fetch, client, item) for item in items], stage="download")
    return await process_detected(
        tenant_type=tenant_type, document_refnumber=document_refnumber, items=items, started_at=started_at,
    )

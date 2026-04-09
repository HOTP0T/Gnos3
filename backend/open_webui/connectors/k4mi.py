"""
K4mi (Paperless-NGX) connector.

Syncs documents from a Paperless-NGX instance into Gnos3's RAG pipeline.
Supports both periodic polling and webhook-based real-time updates.

By default, uses Paperless's OCR text (already extracted during ingest),
avoiding redundant re-processing. Can optionally fetch raw PDFs for Gnos3's
own document loaders to process.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from open_webui.connectors.base import (
    BaseConnector,
    DocumentContent,
    ExternalDocument,
    WebhookAction,
)

log = logging.getLogger(__name__)

# Paperless-NGX API pagination default
PAGE_SIZE = 100


class K4miConnector(BaseConnector):
    """Connector for K4mi (Paperless-NGX) document management system."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip("/")
        self.api_token = config.get("api_token", "")
        self.public_url = config.get("public_url", self.base_url).rstrip("/")
        self.use_ocr_text = config.get("use_ocr_text", True)
        self.tag_filter = config.get("tag_filter") or []
        self.correspondent_filter = config.get("correspondent_filter") or []
        self.document_type_filter = config.get("document_type_filter") or []

    def _headers(self) -> dict:
        return {"Authorization": f"Token {self.api_token}"}

    def _doc_url(self, doc_id: int) -> str:
        return f"{self.public_url}/documents/{doc_id}/details"

    async def _api_get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated GET request to Paperless API."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.base_url}/api{path}",
                headers=self._headers(),
                params=params or {},
            )
            resp.raise_for_status()
            return resp.json()

    async def _api_get_bytes(self, path: str) -> bytes:
        """Download binary content from Paperless API."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(
                f"{self.base_url}/api{path}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.content

    def _build_filters(self) -> dict:
        """Build query params for Paperless document list filtering."""
        params: dict = {}
        if self.tag_filter:
            # Paperless uses tag IDs; if names are given, we'd need to resolve them.
            # For now, support tag IDs directly.
            params["tags__id__in"] = ",".join(str(t) for t in self.tag_filter)
        if self.correspondent_filter:
            params["correspondent__id__in"] = ",".join(
                str(c) for c in self.correspondent_filter
            )
        if self.document_type_filter:
            params["document_type__id__in"] = ",".join(
                str(d) for d in self.document_type_filter
            )
        return params

    async def list_documents(
        self, since: Optional[datetime] = None
    ) -> list[ExternalDocument]:
        docs = []
        page = 1

        while True:
            params = {
                "page": page,
                "page_size": PAGE_SIZE,
                "ordering": "-modified",
                **self._build_filters(),
            }
            if since:
                # Use datetime comparison (not date) to avoid losing same-day uploads.
                # The previous `modified__date__gt=YYYY-MM-DD` filter excluded ALL
                # documents from `since`'s date because Paperless treated it as
                # "strictly greater than the date" rather than "after the moment".
                # We also subtract a 1-hour safety buffer to handle clock skew
                # between Gnos3 and the Paperless server. The content_hash check
                # in the sync service will skip any docs that haven't actually
                # changed, so re-fetching a few extra docs is harmless.
                buffer = timedelta(hours=1)
                # Paperless expects ISO 8601 with timezone — use UTC.
                since_iso = (since - buffer).astimezone(timezone.utc).isoformat()
                params["modified__gt"] = since_iso

            data = await self._api_get("/documents/", params=params)
            results = data.get("results", [])

            for doc in results:
                content_text = doc.get("content", "")
                content_hash = (
                    hashlib.sha256(content_text.encode()).hexdigest()[:16]
                    if content_text
                    else None
                )
                docs.append(
                    ExternalDocument(
                        external_id=str(doc["id"]),
                        title=f"K4mi #{doc['id']} — {doc.get('title', 'Untitled')}",
                        content_hash=content_hash,
                        external_url=self._doc_url(doc["id"]),
                        metadata={
                            "correspondent": doc.get("correspondent"),
                            "document_type": doc.get("document_type"),
                            "tags": doc.get("tags", []),
                            "created_date": doc.get("created"),
                            "modified_date": doc.get("modified"),
                            "added": doc.get("added"),
                        },
                    )
                )

            # Pagination
            if data.get("next"):
                page += 1
            else:
                break

        return docs

    async def fetch_document(self, external_id: str) -> DocumentContent:
        doc_id = int(external_id)
        doc = await self._api_get(f"/documents/{doc_id}/")

        title = doc.get("title", f"Document {doc_id}")
        original_name = doc.get("original_file_name", "")

        # Build rich metadata block for maximum query versatility
        # Users may search by: ID, title, filename, correspondent, type, tags, date, ASN, content
        meta_lines = []
        meta_lines.append(f"{doc_id} K4mi Document {doc_id}")
        meta_lines.append(f"Title: {title}")
        if original_name and original_name != title:
            meta_lines.append(f"Original file: {original_name}")
        if doc.get("correspondent_name"):
            meta_lines.append(f"Correspondent: {doc['correspondent_name']}")
        if doc.get("document_type_name"):
            meta_lines.append(f"Document type: {doc['document_type_name']}")
        if doc.get("created"):
            meta_lines.append(f"Date: {doc['created'][:10]}")
        if doc.get("added"):
            meta_lines.append(f"Added: {doc['added'][:10]}")
        if doc.get("archive_serial_number"):
            meta_lines.append(f"Archive serial number (ASN): {doc['archive_serial_number']}")
        # Resolve tag names — Paperless returns tag IDs in doc, but tag_names might be available
        tag_names = doc.get("tag_names") or []
        if tag_names:
            meta_lines.append(f"Tags: {', '.join(tag_names)}")
        elif doc.get("tags"):
            meta_lines.append(f"Tag IDs: {', '.join(str(t) for t in doc['tags'])}")
        if doc.get("notes"):
            for note in doc["notes"][:3]:
                note_text = note if isinstance(note, str) else note.get("note", "")
                if note_text:
                    meta_lines.append(f"Note: {note_text[:200]}")
        meta_lines.append(f"Source URL: {self._doc_url(doc_id)}")
        meta_header = "\n".join(meta_lines)

        if self.use_ocr_text:
            content = doc.get("content", "")
            # Prepend metadata header for richer context in embeddings
            full_text = f"{meta_header}\n\n{content}" if content else meta_header
            return DocumentContent(
                external_id=external_id,
                title=title,
                content=full_text,
                file_name=f"{title}.txt",
                mime_type="text/plain",
                external_url=self._doc_url(doc_id),
                metadata={
                    "correspondent": doc.get("correspondent"),
                    "correspondent_name": doc.get("correspondent_name"),
                    "document_type": doc.get("document_type"),
                    "document_type_name": doc.get("document_type_name"),
                    "tags": doc.get("tags", []),
                    "tag_names": doc.get("tag_names", []),
                    "created_date": doc.get("created"),
                },
            )
        else:
            # Fetch the raw PDF for Gnos3 loaders to process
            file_bytes = await self._api_get_bytes(f"/documents/{doc_id}/download/")
            original_name = doc.get("original_file_name", f"doc-{doc_id}.pdf")
            return DocumentContent(
                external_id=external_id,
                title=title,
                file_bytes=file_bytes,
                file_name=original_name,
                mime_type="application/pdf",
                external_url=self._doc_url(doc_id),
                metadata={
                    "correspondent": doc.get("correspondent"),
                    "document_type": doc.get("document_type"),
                    "tags": doc.get("tags", []),
                },
            )

    async def get_deleted_ids(self, known_ids: list[str]) -> list[str]:
        """Check which known IDs no longer exist in Paperless."""
        deleted = []
        # Check in batches to avoid huge URLs
        batch_size = 50
        for i in range(0, len(known_ids), batch_size):
            batch = known_ids[i : i + batch_size]
            params = {
                "id__in": ",".join(batch),
                "page_size": batch_size,
            }
            try:
                data = await self._api_get("/documents/", params=params)
                existing_ids = {str(doc["id"]) for doc in data.get("results", [])}
                deleted.extend(eid for eid in batch if eid not in existing_ids)
            except Exception as e:
                log.warning("Error checking deleted docs batch: %s", e)
        return deleted

    def supports_webhooks(self) -> bool:
        return True

    async def handle_webhook(self, payload: dict) -> list[WebhookAction]:
        """
        Parse K4mi/Paperless-NGX webhook events.

        Paperless webhook payload format varies by version. We support:
        - {"document_id": int, "action": "added|updated|deleted"}
        - {"id": int, "type": "document_added|document_updated|document_deleted"}
        """
        actions = []

        # Format 1: explicit action field
        doc_id = payload.get("document_id") or payload.get("id")
        action_str = payload.get("action") or payload.get("type", "")

        if not doc_id:
            log.warning("K4mi webhook: no document_id in payload: %s", payload)
            return actions

        if "added" in action_str:
            actions.append(WebhookAction(
                action="added",
                external_id=str(doc_id),
                title=payload.get("title"),
            ))
        elif "updated" in action_str:
            actions.append(WebhookAction(
                action="updated",
                external_id=str(doc_id),
                title=payload.get("title"),
            ))
        elif "deleted" in action_str:
            actions.append(WebhookAction(
                action="deleted",
                external_id=str(doc_id),
            ))
        else:
            # Default: treat as update
            actions.append(WebhookAction(
                action="updated",
                external_id=str(doc_id),
                title=payload.get("title"),
            ))

        return actions

    @classmethod
    def connector_label(cls) -> str:
        return "K4mi (Paperless-NGX)"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "type": "object",
            "required": ["base_url", "api_token"],
            "properties": {
                "base_url": {
                    "type": "string",
                    "title": "K4mi API URL",
                    "description": "Internal URL of your Paperless-NGX instance (e.g. http://localhost:8000)",
                },
                "api_token": {
                    "type": "string",
                    "title": "API Token",
                    "description": "Paperless-NGX authentication token",
                    "format": "password",
                },
                "public_url": {
                    "type": "string",
                    "title": "Public URL",
                    "description": "Browser-accessible URL for document links (e.g. http://localhost:8000)",
                },
                "use_ocr_text": {
                    "type": "boolean",
                    "title": "Use OCR Text",
                    "description": "Use Paperless OCR text (faster) vs re-processing PDF (more control)",
                    "default": True,
                },
                "tag_filter": {
                    "type": "array",
                    "title": "Tag Filter (IDs)",
                    "description": "Only sync documents with these tag IDs (empty = all)",
                    "items": {"type": "integer"},
                },
                "correspondent_filter": {
                    "type": "array",
                    "title": "Correspondent Filter (IDs)",
                    "description": "Only sync from these correspondents",
                    "items": {"type": "integer"},
                },
                "document_type_filter": {
                    "type": "array",
                    "title": "Document Type Filter (IDs)",
                    "description": "Only sync these document types",
                    "items": {"type": "integer"},
                },
            },
        }

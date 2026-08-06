"""
Business Card Database connector.

Syncs structured contact data from the business-card-processor API into
Gnos3's RAG pipeline by serializing each business card (plus any per-document
notes the user added in K4mi) into a natural-language text document that
embeds well for semantic search.

Notes are first-class: the K4mi note field is the place users jot
context like "met at conference X" or "favorite color is blue", so the
serializer puts that text into the searchable body and gives notes their
own title prefix so retrieval can surface them.

Users can then ask the chat things like:
  - "Who did I meet whose favorite color is blue?"
  - "Show me everyone from ACME Corp"
  - "Which contact mentioned the Q3 partnership at the Berlin event?"
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional

import httpx

from open_webui.connectors.base import (
    BaseConnector,
    DocumentContent,
    ExternalDocument,
)

log = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Serializers: structured records → natural-language text ──────────


def _serialize_k4mi_notes(k4mi_notes) -> list[str]:
    """Flatten the JSONB notes array into readable lines.

    Stored shape (from sync task): [{"note_id": int, "text": str, "created": str}, ...]
    """
    if not k4mi_notes or not isinstance(k4mi_notes, list):
        return []
    lines = []
    for note in k4mi_notes:
        if isinstance(note, dict):
            text = (note.get("text") or "").strip()
            created = note.get("created") or ""
            if text:
                date_part = f" ({created[:10]})" if created else ""
                lines.append(f"Note{date_part}: {text}")
        elif isinstance(note, str) and note.strip():
            lines.append(f"Note: {note.strip()}")
    return lines


def _serialize_business_card(bc: dict) -> str:
    """Convert a business-card record into embedding-friendly text."""
    lines = []

    name = bc.get("full_name") or "Unknown"
    company = bc.get("company_name") or ""
    title = bc.get("job_title") or ""

    header_bits = [name]
    if title:
        header_bits.append(title)
    if company:
        header_bits.append(f"at {company}")
    lines.append("Business card: " + " — ".join(header_bits))

    contact = []
    if bc.get("email"):
        contact.append(f"Email: {bc['email']}")
    if bc.get("email_secondary"):
        contact.append(f"Email (alt): {bc['email_secondary']}")
    if bc.get("phone"):
        contact.append(f"Phone: {bc['phone']}")
    if bc.get("mobile"):
        contact.append(f"Mobile: {bc['mobile']}")
    if bc.get("fax"):
        contact.append(f"Fax: {bc['fax']}")
    if contact:
        lines.append(" | ".join(contact))

    if bc.get("website"):
        lines.append(f"Website: {bc['website']}")
    if bc.get("linkedin"):
        lines.append(f"LinkedIn: {bc['linkedin']}")
    if bc.get("address"):
        lines.append(f"Address: {bc['address']}")

    if bc.get("k4mi_tags"):
        tags = [str(t) for t in bc["k4mi_tags"] if t]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
    if bc.get("k4mi_correspondent"):
        lines.append(f"Correspondent: {bc['k4mi_correspondent']}")

    if bc.get("notes"):
        lines.append(f"Card notes: {bc['notes']}")

    note_lines = _serialize_k4mi_notes(bc.get("k4mi_notes"))
    if note_lines:
        lines.append("")
        lines.append("K4mi document notes:")
        lines.extend(note_lines)

    return "\n".join(lines)


# ── Connector ────────────────────────────────────────────────────────


class BusinessCardDBConnector(BaseConnector):
    """Connector for the business-card-processor contact database."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_base = config.get("api_base_url", "").rstrip("/")
        self.k4mi_public_url = (config.get("k4mi_public_url") or "").rstrip("/")
        # Paginate through every card. Cap to keep one sync round bounded.
        self.page_size = int(config.get("page_size", 200))
        self.max_pages = int(config.get("max_pages", 50))

    async def _api_get(self, path: str, params: Optional[dict] = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.api_base}{path}",
                params=params or {},
                headers=self._service_auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()

    def _extract_list(self, data) -> list:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("business_cards", "items", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []

    def _doc_url(self, bc: dict) -> Optional[str]:
        """Prefer the URL the BC service already stored; fall back to building
        one from k4mi_public_url + k4mi_document_id."""
        stored = bc.get("k4mi_document_url")
        if stored:
            return stored
        doc_id = bc.get("k4mi_document_id")
        if doc_id and self.k4mi_public_url:
            return f"{self.k4mi_public_url}/documents/{doc_id}/details"
        return None

    def _title(self, bc: dict) -> str:
        name = bc.get("full_name") or "Unknown contact"
        company = bc.get("company_name")
        return f"{name} — {company}" if company else name

    async def _reconcile_notes(self) -> None:
        """Ask bc-api to refresh k4mi_notes for every BC before we list.

        Workaround for K4mi not firing document-updated webhooks on
        note-only changes — without this call, notes added in K4mi
        never reach bc-api's DB, and we'd index stale rows here.
        Best-effort: failure logs and we proceed with whatever bc-api has.
        """
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self.api_base}/api/business-cards/reconcile-notes",
                    headers=self._service_auth_headers(),
                )
                if resp.status_code == 200:
                    summary = resp.json()
                    log.info(
                        "BC reconcile-notes: total=%s updated=%s errors=%s",
                        summary.get("total"),
                        summary.get("updated"),
                        summary.get("errors"),
                    )
                else:
                    log.warning(
                        "BC reconcile-notes returned %s: %s",
                        resp.status_code, resp.text[:200],
                    )
        except Exception as e:
            log.warning("BC reconcile-notes failed (continuing): %s", e)

    async def list_documents(
        self, since: Optional[datetime] = None
    ) -> list[ExternalDocument]:
        # Pull the latest notes from K4mi into bc-api first. Notes don't
        # arrive via the document-updated webhook (Paperless doesn't fire
        # it on note-only writes), so if we skip this the connector indexes
        # whatever stale notes bc-api happens to have.
        await self._reconcile_notes()

        docs: list[ExternalDocument] = []

        offset = 0
        for _ in range(self.max_pages):
            try:
                data = await self._api_get(
                    "/api/business-cards",
                    params={"limit": self.page_size, "offset": offset},
                )
            except Exception as e:
                log.warning("Failed to list business cards: %s", e)
                break

            cards = self._extract_list(data)
            if not cards:
                break

            for bc in cards:
                bc_id = bc.get("id")
                if bc_id is None:
                    continue
                text = _serialize_business_card(bc)
                doc_url = self._doc_url(bc)
                docs.append(ExternalDocument(
                    external_id=f"business_card:{bc_id}",
                    title=self._title(bc),
                    content_hash=_hash(text),
                    external_url=doc_url,
                    metadata={
                        "type": "business_card",
                        "source_id": bc_id,
                        "k4mi_document_id": bc.get("k4mi_document_id"),
                        "company_name": bc.get("company_name"),
                        "full_name": bc.get("full_name"),
                    },
                ))

            total = data.get("total") if isinstance(data, dict) else None
            offset += len(cards)
            if total is not None and offset >= total:
                break
            if len(cards) < self.page_size:
                break

        return docs

    async def fetch_document(self, external_id: str) -> DocumentContent:
        parts = external_id.split(":")
        if parts[0] != "business_card":
            raise ValueError(f"Unknown record type in external_id: {external_id}")
        bc_id = parts[1]
        bc = await self._api_get(f"/api/business-cards/{bc_id}")
        text = _serialize_business_card(bc)
        return DocumentContent(
            external_id=external_id,
            title=self._title(bc),
            content=text,
            file_name=f"business-card-{bc_id}.txt",
            mime_type="text/plain",
            external_url=self._doc_url(bc),
            metadata={
                "type": "business_card",
                "source_id": int(bc_id),
                "k4mi_document_id": bc.get("k4mi_document_id"),
                "company_name": bc.get("company_name"),
                "full_name": bc.get("full_name"),
            },
        )

    async def get_deleted_ids(self, known_ids: list[str]) -> list[str]:
        """Probe each known id; 404 means the card was removed (BC tag dropped
        in K4mi or row deleted directly)."""
        deleted = []
        for ext_id in known_ids:
            parts = ext_id.split(":")
            if parts[0] != "business_card" or len(parts) < 2:
                continue
            try:
                await self._api_get(f"/api/business-cards/{parts[1]}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    deleted.append(ext_id)
                else:
                    log.warning("Error checking %s: %s", ext_id, e)
            except Exception as e:
                log.warning("Error checking %s: %s", ext_id, e)
        return deleted

    @classmethod
    def connector_label(cls) -> str:
        return "Business Card Database"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "type": "object",
            "required": ["api_base_url"],
            "properties": {
                "api_base_url": {
                    "type": "string",
                    "title": "Business Card Processor API URL",
                    "description": "Base URL of the business-card-processor API (e.g. http://localhost:8002)",
                },
                "k4mi_public_url": {
                    "type": "string",
                    "title": "K4mi Public URL",
                    "description": "Browser-accessible K4mi base URL for document links (optional; falls back to the value already stored on each card)",
                },
                "page_size": {
                    "type": "integer",
                    "title": "Page size",
                    "default": 200,
                    "minimum": 1,
                    "maximum": 500,
                },
                "max_pages": {
                    "type": "integer",
                    "title": "Max pages per sync",
                    "description": "Safety cap to keep a single sync round bounded",
                    "default": 50,
                    "minimum": 1,
                },
            },
        }

"""
Data Connector sync orchestration service.

Bridges the connector adapters (which fetch external documents) with Gnos3's
existing RAG pipeline (Files, Knowledge Bases, vector DB).

Key flow:
  1. Connector adapter fetches docs from external source
  2. This service creates Gnos3 File records + links them to the connector's Knowledge Base
  3. Content is chunked/embedded via save_docs_to_vector_db() (existing pipeline)
  4. DataConnectorDocument tracks the mapping for change detection + cleanup
"""

import asyncio
import hashlib
import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request
from langchain_core.documents import Document

from open_webui.connectors import get_connector_class
from open_webui.connectors.base import BaseConnector, DocumentContent, WebhookAction
from open_webui.models.data_connector import (
    DataConnectorDocuments,
    DataConnectorModel,
    DataConnectors,
)
from open_webui.models.files import FileForm, Files
from open_webui.models.knowledge import KnowledgeForm, Knowledges
from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
from open_webui.routers.retrieval import save_docs_to_vector_db

log = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """SHA-256 of text content for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _ensure_knowledge_base(
    connector: DataConnectorModel, user_id: str
) -> str:
    """Create a Knowledge Base for the connector if it doesn't have one yet."""
    if connector.knowledge_id:
        kb = Knowledges.get_knowledge_by_id(connector.knowledge_id)
        if kb:
            return connector.knowledge_id

    kb = Knowledges.insert_new_knowledge(
        user_id=user_id,
        form_data=KnowledgeForm(
            name=f"[Connector] {connector.name}",
            description=f"Auto-synced from {connector.connector_type}: {connector.description or connector.name}",
        ),
    )
    if not kb:
        raise RuntimeError(f"Failed to create knowledge base for connector {connector.id}")

    DataConnectors.set_connector_knowledge_id(connector.id, kb.id)
    return kb.id


def _create_or_update_file(
    user_id: str,
    file_id: Optional[str],
    title: str,
    content: str,
    filename: str,
    source_url: Optional[str] = None,
) -> str:
    """Create (or update) a Gnos3 File record with text content. Returns file_id."""
    data = {"content": content}
    meta = {
        "name": title,
        "source": source_url or filename,
        "content_type": "text/plain",
    }

    if file_id:
        existing = Files.get_file_by_id(file_id)
        if existing:
            from open_webui.models.files import FileUpdateForm
            Files.update_file_by_id(
                file_id,
                FileUpdateForm(data=data, meta=meta),
            )
            return file_id

    new_id = str(uuid.uuid4())
    Files.insert_new_file(
        user_id=user_id,
        form_data=FileForm(
            id=new_id,
            filename=filename,
            path="",
            data=data,
            meta=meta,
        ),
    )
    return new_id


_sync_thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dc-sync")


def _index_file_content_sync(
    request: Request,
    file_id: str,
    collection_name: str,
    content: str,
    title: str,
    user_id: str,
    external_url: Optional[str] = None,
) -> None:
    """Chunk, embed, and store content in the vector DB via the existing pipeline.
    Must run in a thread — save_docs_to_vector_db uses run_coroutine_threadsafe
    which deadlocks if called from the event loop."""
    docs = [
        Document(
            page_content=content,
            metadata={
                "name": title,
                "file_id": file_id,
                "source": external_url or title,
                "created_by": user_id,
            },
        )
    ]
    save_docs_to_vector_db(
        request=request,
        docs=docs,
        collection_name=collection_name,
        metadata={"file_id": file_id},
        overwrite=False,
        split=True,
        add=True,
    )


async def _index_file_content(
    request: Request,
    file_id: str,
    collection_name: str,
    content: str,
    title: str,
    user_id: str,
    external_url: Optional[str] = None,
) -> None:
    """Async wrapper that runs embedding in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _sync_thread_pool,
        _index_file_content_sync,
        request, file_id, collection_name, content, title, user_id, external_url,
    )


def _remove_file_from_vector_db(file_id: str, collection_name: str) -> None:
    """Remove a file's vectors from the collection."""
    try:
        result = VECTOR_DB_CLIENT.query(
            collection_name=collection_name,
            filter={"file_id": file_id},
        )
        if result and result.ids and result.ids[0]:
            VECTOR_DB_CLIENT.delete(
                collection_name=collection_name,
                ids=result.ids[0],
            )
    except Exception as e:
        log.warning("Failed to remove vectors for file %s: %s", file_id, e)


def _instantiate_connector(connector: DataConnectorModel) -> BaseConnector:
    """Create a connector adapter instance from the stored config."""
    cls = get_connector_class(connector.connector_type)
    return cls(config=connector.config or {})


# ── Public API ───────────────────────────────────────────────────────


async def sync_connector(
    request: Request,
    connector_id: str,
    user_id: str,
) -> dict:
    """
    Run a full or incremental sync for a connector.

    Returns a summary dict: {added, updated, deleted, errors, total}.
    """
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise ValueError(f"Connector {connector_id} not found")

    DataConnectors.update_connector_sync_status(connector_id, status="running")

    stats = {"added": 0, "updated": 0, "deleted": 0, "errors": 0, "total": 0}

    try:
        adapter = _instantiate_connector(connector)
        knowledge_id = _ensure_knowledge_base(connector, user_id)

        # Determine incremental sync start time.
        # Use timezone-aware UTC so connector adapters can safely convert to ISO 8601.
        since = None
        if connector.last_sync_at:
            since = datetime.fromtimestamp(connector.last_sync_at, tz=timezone.utc)

        # 1. List new/changed documents from source
        external_docs = await adapter.list_documents(since=since)
        log.info(
            "Connector %s: found %d documents to process (since=%s)",
            connector.name, len(external_docs), since,
        )

        # 2. Process each document
        for ext_doc in external_docs:
            try:
                existing = DataConnectorDocuments.get_document_by_external_id(
                    connector_id, ext_doc.external_id
                )

                # Skip if content hash unchanged
                if (
                    existing
                    and existing.content_hash
                    and ext_doc.content_hash
                    and existing.content_hash == ext_doc.content_hash
                ):
                    continue

                # Fetch full content
                doc_content = await adapter.fetch_document(ext_doc.external_id)

                # Get text content
                text = doc_content.content or ""
                if not text and doc_content.file_bytes:
                    text = f"[Binary file: {doc_content.file_name or ext_doc.title}]"

                if not text.strip():
                    log.warning("Empty content for %s, skipping", ext_doc.external_id)
                    continue

                content_hash = ext_doc.content_hash or _content_hash(text)
                filename = doc_content.file_name or f"{ext_doc.title}.txt"

                # Create/update Gnos3 file
                file_id = _create_or_update_file(
                    user_id=user_id,
                    file_id=existing.file_id if existing else None,
                    title=ext_doc.title,
                    content=text,
                    filename=filename,
                    source_url=ext_doc.external_url,
                )

                # If updating, remove old vectors first
                if existing and existing.file_id:
                    _remove_file_from_vector_db(existing.file_id, knowledge_id)

                # Index into vector DB (runs in thread pool)
                await _index_file_content(
                    request=request,
                    file_id=file_id,
                    collection_name=knowledge_id,
                    content=text,
                    title=ext_doc.title,
                    user_id=user_id,
                    external_url=ext_doc.external_url,
                )

                # Link file to knowledge base (idempotent — unique constraint)
                Knowledges.add_file_to_knowledge_by_id(
                    knowledge_id=knowledge_id,
                    file_id=file_id,
                    user_id=user_id,
                )

                # Track in connector documents table
                DataConnectorDocuments.upsert_document(
                    connector_id=connector_id,
                    external_id=ext_doc.external_id,
                    file_id=file_id,
                    title=ext_doc.title,
                    content_hash=content_hash,
                    external_url=ext_doc.external_url,
                    meta=doc_content.metadata or {},
                )

                if existing:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1

            except Exception as e:
                log.exception("Error processing doc %s: %s", ext_doc.external_id, e)
                stats["errors"] += 1

        # 3. Detect and remove deleted documents
        known_ids = DataConnectorDocuments.get_external_ids_for_connector(connector_id)
        if known_ids:
            try:
                deleted_ids = await adapter.get_deleted_ids(known_ids)
                for ext_id in deleted_ids:
                    try:
                        file_id = DataConnectorDocuments.delete_document(
                            connector_id, ext_id
                        )
                        if file_id:
                            _remove_file_from_vector_db(file_id, knowledge_id)
                        stats["deleted"] += 1
                    except Exception as e:
                        log.exception("Error deleting doc %s: %s", ext_id, e)
                        stats["errors"] += 1
            except Exception as e:
                log.exception("Error checking deletions: %s", e)

        # 4. Update connector status
        doc_count = DataConnectorDocuments.count_by_connector(connector_id)
        stats["total"] = doc_count
        DataConnectors.update_connector_sync_status(
            connector_id,
            status="success",
            doc_count=doc_count,
            meta={"last_sync_stats": stats},
        )

    except Exception as e:
        log.exception("Sync failed for connector %s: %s", connector_id, e)
        DataConnectors.update_connector_sync_status(
            connector_id,
            status="error",
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}",
        )
        raise

    return stats


async def handle_webhook_event(
    request: Request,
    connector_id: str,
    user_id: str,
    payload: dict,
) -> list[dict]:
    """
    Process an incoming webhook event for a connector.
    Returns a list of actions taken.
    """
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise ValueError(f"Connector {connector_id} not found")
    if not connector.enabled:
        return [{"status": "skipped", "reason": "connector disabled"}]

    adapter = _instantiate_connector(connector)
    if not adapter.supports_webhooks():
        raise ValueError(f"Connector type {connector.connector_type} does not support webhooks")

    knowledge_id = _ensure_knowledge_base(connector, user_id)
    actions = await adapter.handle_webhook(payload)
    results = []

    for action in actions:
        try:
            if action.action in ("added", "updated"):
                doc_content = await adapter.fetch_document(action.external_id)
                text = doc_content.content or ""
                if not text.strip():
                    results.append({"action": action.action, "external_id": action.external_id, "status": "skipped_empty"})
                    continue

                existing = DataConnectorDocuments.get_document_by_external_id(
                    connector_id, action.external_id
                )
                content_hash = _content_hash(text)
                filename = doc_content.file_name or f"{action.title or action.external_id}.txt"

                file_id = _create_or_update_file(
                    user_id=user_id,
                    file_id=existing.file_id if existing else None,
                    title=action.title or action.external_id,
                    content=text,
                    filename=filename,
                    source_url=doc_content.external_url,
                )

                if existing and existing.file_id:
                    _remove_file_from_vector_db(existing.file_id, knowledge_id)

                await _index_file_content(
                    request=request,
                    file_id=file_id,
                    collection_name=knowledge_id,
                    content=text,
                    title=action.title or action.external_id,
                    user_id=user_id,
                    external_url=doc_content.external_url,
                )

                Knowledges.add_file_to_knowledge_by_id(
                    knowledge_id=knowledge_id, file_id=file_id, user_id=user_id
                )

                DataConnectorDocuments.upsert_document(
                    connector_id=connector_id,
                    external_id=action.external_id,
                    file_id=file_id,
                    title=action.title,
                    content_hash=content_hash,
                    external_url=doc_content.external_url,
                    meta=doc_content.metadata or {},
                )

                results.append({"action": action.action, "external_id": action.external_id, "status": "ok"})

            elif action.action == "deleted":
                file_id = DataConnectorDocuments.delete_document(
                    connector_id, action.external_id
                )
                if file_id:
                    _remove_file_from_vector_db(file_id, knowledge_id)
                results.append({"action": "deleted", "external_id": action.external_id, "status": "ok"})

        except Exception as e:
            log.exception("Webhook action failed: %s", e)
            results.append({"action": action.action, "external_id": action.external_id, "status": "error", "error": str(e)})

    # Update doc count
    doc_count = DataConnectorDocuments.count_by_connector(connector_id)
    DataConnectors.update_connector_sync_status(
        connector_id, status="success", doc_count=doc_count
    )

    return results


async def delete_connector_data(connector_id: str) -> None:
    """Clean up all data associated with a connector (files, vectors, KB)."""
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        return

    # Remove all tracked documents' vectors
    if connector.knowledge_id:
        file_ids = DataConnectorDocuments.delete_all_for_connector(connector_id)
        for file_id in file_ids:
            _remove_file_from_vector_db(file_id, connector.knowledge_id)

        # Delete the vector DB collection
        try:
            VECTOR_DB_CLIENT.delete_collection(collection_name=connector.knowledge_id)
        except Exception as e:
            log.warning("Failed to delete collection %s: %s", connector.knowledge_id, e)

    # Delete the connector record itself
    DataConnectors.delete_connector_by_id(connector_id)

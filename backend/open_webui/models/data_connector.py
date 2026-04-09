"""
Data Connector models — track external data source connections and their synced documents.

Each DataConnector owns a Gnos3 Knowledge Base. Documents fetched from the external source
are stored as Gnos3 Files, linked to that KB, and indexed in the vector DB via the
standard RAG pipeline.

DataConnectorDocument tracks the mapping between external IDs (e.g. K4mi doc #142,
invoice-processor invoice #89) and Gnos3 file IDs, plus content hashes for
change detection.
"""

import logging
import time
from typing import Optional
import uuid

from pydantic import BaseModel, ConfigDict
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Integer,
    String,
    Text,
    JSON,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Session

from open_webui.internal.db import Base, get_db_context

log = logging.getLogger(__name__)


# ── SQLAlchemy Models ────────────────────────────────────────────────


class DataConnector(Base):
    __tablename__ = "data_connector"

    id = Column(Text, primary_key=True, unique=True)
    user_id = Column(Text, nullable=False)

    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    connector_type = Column(String(50), nullable=False)  # "k4mi", "invoice_db", ...

    config = Column(JSON, nullable=True)       # type-specific config (URLs, tokens, filters)
    knowledge_id = Column(Text, nullable=True)  # linked Gnos3 knowledge base

    enabled = Column(Boolean, nullable=False, default=True)
    sync_interval = Column(Integer, nullable=False, default=3600)  # seconds

    last_sync_at = Column(BigInteger, nullable=True)
    last_sync_status = Column(String(20), nullable=True)   # "success", "error", "running"
    last_sync_error = Column(Text, nullable=True)
    doc_count = Column(Integer, nullable=False, default=0)

    meta = Column(JSON, nullable=True)  # sync cursor, progress, stats

    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)


class DataConnectorDocument(Base):
    __tablename__ = "data_connector_document"

    id = Column(Text, primary_key=True, unique=True)
    connector_id = Column(Text, nullable=False, index=True)

    external_id = Column(Text, nullable=False)   # source system ID
    file_id = Column(Text, nullable=True)        # Gnos3 file ID after processing
    title = Column(Text, nullable=True)
    content_hash = Column(Text, nullable=True)   # detect content changes
    external_url = Column(Text, nullable=True)   # link back to source

    meta = Column(JSON, nullable=True)           # extra metadata (updated_at from source, etc.)

    last_synced_at = Column(BigInteger, nullable=False)
    created_at = Column(BigInteger, nullable=False)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint("connector_id", "external_id", name="uq_connector_external_id"),
        Index("ix_data_connector_doc_connector", "connector_id"),
        Index("ix_data_connector_doc_external", "external_id"),
    )


# ── Pydantic Schemas ─────────────────────────────────────────────────


class DataConnectorModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    connector_type: str
    config: Optional[dict] = None
    knowledge_id: Optional[str] = None
    enabled: bool = True
    sync_interval: int = 3600
    last_sync_at: Optional[int] = None
    last_sync_status: Optional[str] = None
    last_sync_error: Optional[str] = None
    doc_count: int = 0
    meta: Optional[dict] = None
    created_at: int
    updated_at: int


class DataConnectorDocumentModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    connector_id: str
    external_id: str
    file_id: Optional[str] = None
    title: Optional[str] = None
    content_hash: Optional[str] = None
    external_url: Optional[str] = None
    meta: Optional[dict] = None
    last_synced_at: int
    created_at: int
    updated_at: int


# ── Form Schemas ─────────────────────────────────────────────────────


class DataConnectorForm(BaseModel):
    name: str
    description: Optional[str] = None
    connector_type: str
    config: Optional[dict] = None
    enabled: bool = True
    sync_interval: int = 3600


class DataConnectorUpdateForm(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[dict] = None
    enabled: Optional[bool] = None
    sync_interval: Optional[int] = None


# ── Service Class ────────────────────────────────────────────────────


class DataConnectorTable:
    def insert_new_connector(
        self,
        user_id: str,
        form_data: DataConnectorForm,
        db: Optional[Session] = None,
    ) -> Optional[DataConnectorModel]:
        with get_db_context(db) as db:
            connector = DataConnectorModel(
                **{
                    **form_data.model_dump(),
                    "id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "knowledge_id": None,
                    "last_sync_at": None,
                    "last_sync_status": None,
                    "last_sync_error": None,
                    "doc_count": 0,
                    "meta": None,
                    "created_at": int(time.time()),
                    "updated_at": int(time.time()),
                }
            )
            try:
                result = DataConnector(**connector.model_dump())
                db.add(result)
                db.commit()
                db.refresh(result)
                return DataConnectorModel.model_validate(result)
            except Exception as e:
                db.rollback()
                log.exception("Error creating data connector: %s", e)
                return None

    def get_connectors(
        self, db: Optional[Session] = None
    ) -> list[DataConnectorModel]:
        with get_db_context(db) as db:
            return [
                DataConnectorModel.model_validate(c)
                for c in db.query(DataConnector).order_by(DataConnector.created_at.desc()).all()
            ]

    def get_connector_by_id(
        self, connector_id: str, db: Optional[Session] = None
    ) -> Optional[DataConnectorModel]:
        with get_db_context(db) as db:
            result = db.query(DataConnector).filter_by(id=connector_id).first()
            return DataConnectorModel.model_validate(result) if result else None

    def get_enabled_connectors(
        self, db: Optional[Session] = None
    ) -> list[DataConnectorModel]:
        with get_db_context(db) as db:
            return [
                DataConnectorModel.model_validate(c)
                for c in db.query(DataConnector)
                .filter_by(enabled=True)
                .order_by(DataConnector.created_at.asc())
                .all()
            ]

    def update_connector_by_id(
        self,
        connector_id: str,
        form_data: DataConnectorUpdateForm,
        db: Optional[Session] = None,
    ) -> Optional[DataConnectorModel]:
        with get_db_context(db) as db:
            connector = db.query(DataConnector).filter_by(id=connector_id).first()
            if not connector:
                return None
            update_data = form_data.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(connector, key, value)
            connector.updated_at = int(time.time())
            db.commit()
            db.refresh(connector)
            return DataConnectorModel.model_validate(connector)

    def update_connector_sync_status(
        self,
        connector_id: str,
        status: str,
        error: Optional[str] = None,
        doc_count: Optional[int] = None,
        meta: Optional[dict] = None,
        db: Optional[Session] = None,
    ) -> Optional[DataConnectorModel]:
        with get_db_context(db) as db:
            connector = db.query(DataConnector).filter_by(id=connector_id).first()
            if not connector:
                return None
            connector.last_sync_status = status
            connector.last_sync_error = error
            if status == "success":
                connector.last_sync_at = int(time.time())
            if doc_count is not None:
                connector.doc_count = doc_count
            if meta is not None:
                connector.meta = {**(connector.meta or {}), **meta}
            connector.updated_at = int(time.time())
            db.commit()
            db.refresh(connector)
            return DataConnectorModel.model_validate(connector)

    def set_connector_knowledge_id(
        self,
        connector_id: str,
        knowledge_id: str,
        db: Optional[Session] = None,
    ) -> None:
        with get_db_context(db) as db:
            connector = db.query(DataConnector).filter_by(id=connector_id).first()
            if connector:
                connector.knowledge_id = knowledge_id
                connector.updated_at = int(time.time())
                db.commit()

    def delete_connector_by_id(
        self, connector_id: str, db: Optional[Session] = None
    ) -> bool:
        with get_db_context(db) as db:
            result = db.query(DataConnector).filter_by(id=connector_id).delete()
            db.commit()
            return result > 0


class DataConnectorDocumentTable:
    def upsert_document(
        self,
        connector_id: str,
        external_id: str,
        file_id: Optional[str] = None,
        title: Optional[str] = None,
        content_hash: Optional[str] = None,
        external_url: Optional[str] = None,
        meta: Optional[dict] = None,
        db: Optional[Session] = None,
    ) -> DataConnectorDocumentModel:
        with get_db_context(db) as db:
            existing = (
                db.query(DataConnectorDocument)
                .filter_by(connector_id=connector_id, external_id=external_id)
                .first()
            )
            now = int(time.time())
            if existing:
                if file_id is not None:
                    existing.file_id = file_id
                if title is not None:
                    existing.title = title
                if content_hash is not None:
                    existing.content_hash = content_hash
                if external_url is not None:
                    existing.external_url = external_url
                if meta is not None:
                    existing.meta = {**(existing.meta or {}), **meta}
                existing.last_synced_at = now
                existing.updated_at = now
                db.commit()
                db.refresh(existing)
                return DataConnectorDocumentModel.model_validate(existing)
            else:
                doc = DataConnectorDocument(
                    id=str(uuid.uuid4()),
                    connector_id=connector_id,
                    external_id=external_id,
                    file_id=file_id,
                    title=title,
                    content_hash=content_hash,
                    external_url=external_url,
                    meta=meta,
                    last_synced_at=now,
                    created_at=now,
                    updated_at=now,
                )
                db.add(doc)
                db.commit()
                db.refresh(doc)
                return DataConnectorDocumentModel.model_validate(doc)

    def get_documents_by_connector(
        self, connector_id: str, db: Optional[Session] = None
    ) -> list[DataConnectorDocumentModel]:
        with get_db_context(db) as db:
            return [
                DataConnectorDocumentModel.model_validate(d)
                for d in db.query(DataConnectorDocument)
                .filter_by(connector_id=connector_id)
                .order_by(DataConnectorDocument.created_at.desc())
                .all()
            ]

    def get_document_by_external_id(
        self,
        connector_id: str,
        external_id: str,
        db: Optional[Session] = None,
    ) -> Optional[DataConnectorDocumentModel]:
        with get_db_context(db) as db:
            result = (
                db.query(DataConnectorDocument)
                .filter_by(connector_id=connector_id, external_id=external_id)
                .first()
            )
            return DataConnectorDocumentModel.model_validate(result) if result else None

    def get_external_ids_for_connector(
        self, connector_id: str, db: Optional[Session] = None
    ) -> list[str]:
        with get_db_context(db) as db:
            rows = (
                db.query(DataConnectorDocument.external_id)
                .filter_by(connector_id=connector_id)
                .all()
            )
            return [r[0] for r in rows]

    def delete_document(
        self,
        connector_id: str,
        external_id: str,
        db: Optional[Session] = None,
    ) -> Optional[str]:
        """Delete tracking record. Returns the file_id so caller can clean up."""
        with get_db_context(db) as db:
            doc = (
                db.query(DataConnectorDocument)
                .filter_by(connector_id=connector_id, external_id=external_id)
                .first()
            )
            if not doc:
                return None
            file_id = doc.file_id
            db.delete(doc)
            db.commit()
            return file_id

    def delete_all_for_connector(
        self, connector_id: str, db: Optional[Session] = None
    ) -> list[str]:
        """Delete all docs for a connector. Returns file_ids for cleanup."""
        with get_db_context(db) as db:
            docs = (
                db.query(DataConnectorDocument)
                .filter_by(connector_id=connector_id)
                .all()
            )
            file_ids = [d.file_id for d in docs if d.file_id]
            db.query(DataConnectorDocument).filter_by(connector_id=connector_id).delete()
            db.commit()
            return file_ids

    def count_by_connector(
        self, connector_id: str, db: Optional[Session] = None
    ) -> int:
        with get_db_context(db) as db:
            return (
                db.query(DataConnectorDocument)
                .filter_by(connector_id=connector_id)
                .count()
            )


# Singleton instances
DataConnectors = DataConnectorTable()
DataConnectorDocuments = DataConnectorDocumentTable()

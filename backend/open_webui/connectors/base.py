"""
Abstract base class for all data source connectors.

Every connector must implement list_documents(), fetch_document(), and get_deleted_ids().
Connectors that support real-time push (webhooks) override supports_webhooks() and
handle_webhook().

The sync service calls these methods — connectors don't touch the RAG pipeline directly.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel


class ExternalDocument(BaseModel):
    """Lightweight listing entry returned by list_documents()."""
    external_id: str
    title: str
    content_hash: Optional[str] = None
    external_url: Optional[str] = None
    metadata: dict = {}


class DocumentContent(BaseModel):
    """Full document content returned by fetch_document()."""
    external_id: str
    title: str
    content: Optional[str] = None       # extracted text (if source provides it)
    file_bytes: Optional[bytes] = None  # raw file (for Gnos3 loaders to process)
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    external_url: Optional[str] = None
    metadata: dict = {}

    class Config:
        arbitrary_types_allowed = True


class WebhookAction(BaseModel):
    """Result of processing an incoming webhook event."""
    action: Literal["added", "updated", "deleted"]
    external_id: str
    title: Optional[str] = None


class BaseConnector(ABC):
    """Abstract interface that all data source connectors must implement."""

    def __init__(self, config: dict):
        self.config = config
        # Set by the sync service after instantiation (see
        # data_connector._instantiate_connector). The id of the Gnos3 user the
        # connector runs as — its owner. Used to mint service auth for calls to
        # Gnos3-authenticated module APIs.
        self.user_id: Optional[str] = None

    def _service_auth_headers(self) -> dict:
        """Mint a short-lived service JWT for the connector's owner so calls to
        Gnos3-authenticated module APIs (invoice-processor, business-card-
        processor) are authorized as that user.

        Signed with WEBUI_SECRET_KEY — the same key the modules verify with —
        and carries only the owner's `id`, matching the JWT shape the modules
        expect. Minted fresh per call and short-lived, so there is no stored
        secret to manage or rotate. Returns {} when no owner is set, which
        preserves the pre-auth behavior rather than crashing (the sync worker
        always sets `user_id`, so this only guards direct/test instantiation).
        """
        if not self.user_id:
            return {}
        # Lazy import: utils.auth pulls in models/config, and importing it at
        # module load could risk cycles during connector registration.
        from open_webui.utils.auth import create_token

        token = create_token({"id": self.user_id}, expires_delta=timedelta(minutes=10))
        return {"Authorization": f"Bearer {token}"}

    @abstractmethod
    async def list_documents(
        self, since: Optional[datetime] = None
    ) -> list[ExternalDocument]:
        """
        List documents available in the source.
        If `since` is provided, return only documents added/modified after that time.
        """

    @abstractmethod
    async def fetch_document(self, external_id: str) -> DocumentContent:
        """Fetch full document content + metadata by external ID."""

    @abstractmethod
    async def get_deleted_ids(self, known_ids: list[str]) -> list[str]:
        """
        Given a list of external IDs we currently track, return the subset
        that no longer exist in the source (i.e. were deleted).
        """

    def supports_webhooks(self) -> bool:
        """Override to return True if this connector can receive push events."""
        return False

    async def handle_webhook(self, payload: dict) -> list[WebhookAction]:
        """
        Process an incoming webhook payload. Returns a list of actions to take.
        Override if supports_webhooks() returns True.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support webhooks"
        )

    @classmethod
    def config_schema(cls) -> dict:
        """
        Return a JSON Schema dict describing the connector's config fields.
        Used by the frontend to render type-specific config forms.
        Override in subclasses.
        """
        return {}

    @classmethod
    def connector_label(cls) -> str:
        """Human-readable label for this connector type."""
        return cls.__name__

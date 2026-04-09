"""
REST API for managing Data Connectors — external data sources synced into Gnos3's RAG pipeline.

Endpoints:
  GET    /                    List all connectors
  POST   /create              Create a new connector (+ auto-creates linked Knowledge Base)
  GET    /types               List available connector types + config schemas
  GET    /{id}                Get connector details + sync stats
  POST   /{id}/update         Update connector config
  DELETE /{id}/delete         Delete connector + cleanup
  POST   /{id}/sync           Trigger manual sync
  GET    /{id}/documents      List synced documents for a connector
  POST   /webhook/{id}        Receive push events from external source
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from pydantic import BaseModel

from open_webui.connectors import get_available_types
from open_webui.models.data_connector import (
    DataConnectorDocuments,
    DataConnectorForm,
    DataConnectorModel,
    DataConnectorUpdateForm,
    DataConnectors,
)
from open_webui.services.data_connector import (
    delete_connector_data,
    handle_webhook_event,
    sync_connector,
)
from open_webui.utils.auth import get_admin_user, get_verified_user

log = logging.getLogger(__name__)

router = APIRouter()


# ── List / Types ─────────────────────────────────────────────────────


@router.get("/", response_model=list[DataConnectorModel])
async def list_connectors(user=Depends(get_verified_user)):
    return DataConnectors.get_connectors()


@router.get("/types")
async def list_connector_types(user=Depends(get_verified_user)):
    return get_available_types()


# ── CRUD ─────────────────────────────────────────────────────────────


@router.post("/create", response_model=Optional[DataConnectorModel])
async def create_connector(
    request: Request,
    form_data: DataConnectorForm,
    user=Depends(get_admin_user),
):
    connector = DataConnectors.insert_new_connector(
        user_id=user.id,
        form_data=form_data,
    )
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create connector",
        )
    return connector


@router.get("/{connector_id}", response_model=Optional[DataConnectorModel])
async def get_connector(
    connector_id: str,
    user=Depends(get_verified_user),
):
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    return connector


@router.post("/{connector_id}/update", response_model=Optional[DataConnectorModel])
async def update_connector(
    connector_id: str,
    form_data: DataConnectorUpdateForm,
    user=Depends(get_admin_user),
):
    connector = DataConnectors.update_connector_by_id(connector_id, form_data)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    return connector


@router.delete("/{connector_id}/delete")
async def delete_connector(
    connector_id: str,
    user=Depends(get_admin_user),
):
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    await delete_connector_data(connector_id)
    return {"status": "ok", "message": f"Connector {connector.name} deleted"}


# ── Sync ─────────────────────────────────────────────────────────────


@router.post("/{connector_id}/sync")
async def trigger_sync(
    request: Request,
    connector_id: str,
    user=Depends(get_admin_user),
):
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    if connector.last_sync_status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Sync already in progress",
        )

    # Run sync in background thread to avoid blocking the event loop
    # (save_docs_to_vector_db does synchronous embedding)
    app = request.app

    async def _run_sync():
        try:
            mock_request = Request(
                scope={
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "headers": [],
                    "path": "/",
                    "root_path": "",
                    "scheme": "http",
                    "query_string": b"",
                    "server": ("localhost", 8080),
                    "app": app,
                }
            )
            await sync_connector(
                request=mock_request,
                connector_id=connector_id,
                user_id=user.id,
            )
        except Exception as e:
            log.exception("Background sync failed for %s: %s", connector_id, e)

    asyncio.create_task(_run_sync())
    return {"status": "ok", "message": "Sync started in background"}


# ── Documents ────────────────────────────────────────────────────────


@router.get("/{connector_id}/documents")
async def list_connector_documents(
    connector_id: str,
    user=Depends(get_verified_user),
):
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )
    return DataConnectorDocuments.get_documents_by_connector(connector_id)


# ── Webhook ──────────────────────────────────────────────────────────


@router.post("/webhook/{connector_id}")
async def receive_webhook(
    request: Request,
    connector_id: str,
):
    """
    Receive push events from external data sources.
    No auth required — connectors use their own verification (e.g. shared secret in config).
    """
    connector = DataConnectors.get_connector_by_id(connector_id)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connector not found",
        )

    payload = await request.json()
    try:
        results = await handle_webhook_event(
            request=request,
            connector_id=connector_id,
            user_id=connector.user_id,
            payload=payload,
        )
        return {"status": "ok", "actions": results}
    except Exception as e:
        log.exception("Webhook processing failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing failed: {str(e)}",
        )

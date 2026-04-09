"""
Background sync worker for Data Connectors.

Started as an asyncio task in main.py lifespan(). Polls every 60s for connectors
whose sync interval has elapsed, and runs sync_connector() for each one.
"""

import asyncio
import logging
import time

from fastapi import Request

from open_webui.models.data_connector import DataConnectors
from open_webui.services.data_connector import sync_connector

log = logging.getLogger(__name__)

POLL_INTERVAL = 60  # seconds between checks


async def data_connector_sync_worker(app) -> None:
    """
    Periodic background task that checks for connectors needing sync.

    Runs in the main event loop alongside FastAPI. Each sync is executed
    sequentially (one connector at a time) to avoid overwhelming the
    embedding service.
    """
    log.info("Data connector sync worker started")

    # Wait a bit for the app to fully initialize
    await asyncio.sleep(10)

    while True:
        try:
            connectors = DataConnectors.get_enabled_connectors()
            now = int(time.time())

            for connector in connectors:
                # Skip if sync interval hasn't elapsed
                if connector.last_sync_at:
                    next_sync = connector.last_sync_at + connector.sync_interval
                    if now < next_sync:
                        continue

                # Skip if already running
                if connector.last_sync_status == "running":
                    continue

                log.info(
                    "Auto-sync starting for connector: %s (%s)",
                    connector.name, connector.connector_type,
                )

                try:
                    # Build a minimal mock Request for save_docs_to_vector_db
                    # which needs request.app.state for config access
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

                    stats = await sync_connector(
                        request=mock_request,
                        connector_id=connector.id,
                        user_id=connector.user_id,
                    )
                    log.info(
                        "Auto-sync completed for %s: +%d /%d -%d (%d errors)",
                        connector.name,
                        stats["added"], stats["updated"],
                        stats["deleted"], stats["errors"],
                    )
                except Exception as e:
                    log.exception("Auto-sync failed for %s: %s", connector.name, e)

        except Exception as e:
            log.exception("Data connector worker error: %s", e)

        await asyncio.sleep(POLL_INTERVAL)

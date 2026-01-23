"""
title: K4mi Document Assistant
description: Search and retrieve documents from Paperless-ngx (K4mi) for RAG
author: Claude
version: 2.0.0
license: MIT
requirements: aiohttp
"""

import aiohttp
import json
import re
from typing import Callable, Any, Optional, List
from pydantic import BaseModel, Field


class Tools:
    """
    A streamlined tool for searching Paperless-ngx documents.

    Key differences from v1:
    - Async HTTP with aiohttp (faster, non-blocking)
    - Simpler API with just 2 main methods
    - Better context formatting for RAG
    - Automatic relevance scoring
    """

    class Valves(BaseModel):
        paperless_url: str = Field(
            default="",
            description="Paperless-ngx base URL (e.g., https://k4mi.example.com)"
        )
        api_token: str = Field(
            default="",
            description="Paperless-ngx API token"
        )
        default_limit: int = Field(
            default=5,
            description="Default number of documents to return"
        )
        content_preview_length: int = Field(
            default=2000,
            description="Characters of content to include in search results (0 = full)"
        )
        timeout_seconds: int = Field(
            default=30,
            description="API request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_base_url(self) -> str:
        """Get normalized base URL"""
        url = self.valves.paperless_url.strip().rstrip('/')
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        return url

    async def _api_get(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None) -> dict:
        """Make authenticated GET request to Paperless API"""
        url = f"{self._get_base_url()}/api/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Token {self.valves.api_token}",
            "Accept": "application/json"
        }

        timeout = aiohttp.ClientTimeout(total=self.valves.timeout_seconds)
        async with session.get(url, headers=headers, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"API error {resp.status}: {error_text[:300]}")
            return await resp.json()

    async def _emit_status(self, emitter: Callable, message: str, status: str = "in_progress", done: bool = False):
        """Emit a status update"""
        if emitter:
            await emitter({
                "type": "status",
                "data": {"status": status, "description": message, "done": done}
            })

    async def _emit_citation(self, emitter: Callable, doc: dict):
        """Emit a citation for RAG"""
        if emitter:
            await emitter({
                "type": "citation",
                "data": {
                    "document": [doc.get('content', '')[:500] or doc.get('title', '')],
                    "metadata": [{"source": doc.get('title', 'Document')}],
                    "source": {"name": doc.get('url', '')}
                }
            })

    async def search(
        self,
        query: str,
        limit: int = 5,
        tags: str = "",
        correspondent: str = "",
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Search documents in K4mi/Paperless-ngx and return relevant content for RAG.

        Use this tool when the user asks about their documents, invoices, receipts,
        or any information that might be stored in their document management system.

        Examples of when to use this tool:
        - "What invoices do I have from Amazon?"
        - "Find documents about insurance"
        - "Show me my tax documents"
        - "What did I pay for electricity last month?"

        :param query: Search terms to find in documents (required)
        :param limit: Maximum number of documents to return (default: 5)
        :param tags: Filter by tag names, comma-separated (optional)
        :param correspondent: Filter by correspondent/vendor name (optional)
        :return: JSON with matching documents and their content
        """
        await self._emit_status(__event_emitter__, f"Searching K4mi for: {query}")

        if not self.valves.paperless_url or not self.valves.api_token:
            return json.dumps({
                "error": "K4mi not configured. Please set paperless_url and api_token in Valves."
            })

        try:
            async with aiohttp.ClientSession() as session:
                # Build search parameters
                params = {"query": query, "page_size": min(limit, 25)}

                # Resolve tag filter
                if tags:
                    await self._emit_status(__event_emitter__, "Resolving tags...")
                    tags_data = await self._api_get(session, "tags/")
                    tag_names_lower = [t.strip().lower() for t in tags.split(',')]
                    matching_ids = [
                        str(t['id']) for t in tags_data.get('results', [])
                        if t['name'].lower() in tag_names_lower
                    ]
                    if matching_ids:
                        params['tags__id__in'] = ','.join(matching_ids)

                # Resolve correspondent filter
                if correspondent:
                    await self._emit_status(__event_emitter__, "Resolving correspondent...")
                    corr_data = await self._api_get(session, "correspondents/")
                    corr_lower = correspondent.lower()
                    for c in corr_data.get('results', []):
                        if corr_lower in c['name'].lower():
                            params['correspondent__id'] = c['id']
                            break

                # Search documents
                await self._emit_status(__event_emitter__, f"Fetching documents...")
                search_data = await self._api_get(session, "documents/", params)
                raw_docs = search_data.get('results', [])

                if not raw_docs:
                    await self._emit_status(__event_emitter__, "No documents found", "info", True)
                    return json.dumps({
                        "message": f"No documents found matching '{query}'",
                        "query": query,
                        "count": 0,
                        "documents": []
                    })

                # Fetch full document details
                await self._emit_status(__event_emitter__, f"Processing {len(raw_docs)} documents...")
                documents = []

                for raw in raw_docs:
                    doc_data = await self._api_get(session, f"documents/{raw['id']}/")

                    # Get content
                    content = doc_data.get('content', '') or ''
                    preview_len = self.valves.content_preview_length
                    if preview_len > 0 and len(content) > preview_len:
                        content = content[:preview_len] + f"\n[...truncated, {len(doc_data.get('content', ''))} total chars]"

                    doc = {
                        "id": raw['id'],
                        "title": doc_data.get('title', 'Untitled'),
                        "content": content,
                        "correspondent": doc_data.get('correspondent_name'),
                        "document_type": doc_data.get('document_type_name'),
                        "created": doc_data.get('created', '')[:10] if doc_data.get('created') else None,
                        "tags": [t.get('name', t) if isinstance(t, dict) else t for t in doc_data.get('tags', [])],
                        "url": f"{self._get_base_url()}/documents/{raw['id']}/details",
                        "relevance_score": raw.get('__search_hit__', {}).get('score', 0)
                    }

                    documents.append(doc)
                    await self._emit_citation(__event_emitter__, doc)

                # Sort by relevance
                documents.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

                await self._emit_status(__event_emitter__, f"Found {len(documents)} documents", "success", True)

                return json.dumps({
                    "query": query,
                    "count": len(documents),
                    "documents": documents,
                    "hint": "Use the 'content' field to find specific information the user asked about."
                }, ensure_ascii=False, indent=2)

        except Exception as e:
            await self._emit_status(__event_emitter__, f"Error: {str(e)}", "error", True)
            return json.dumps({"error": str(e), "query": query})

    async def get_document(
        self,
        document_id: int,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Get full details of a specific document by ID.

        Use this when you need complete information about a document you found in search,
        or when the user references a specific document ID.

        :param document_id: The numeric ID of the document
        :return: Complete document data including full content
        """
        await self._emit_status(__event_emitter__, f"Fetching document {document_id}...")

        if not self.valves.paperless_url or not self.valves.api_token:
            return json.dumps({
                "error": "K4mi not configured. Please set paperless_url and api_token in Valves."
            })

        try:
            async with aiohttp.ClientSession() as session:
                doc_data = await self._api_get(session, f"documents/{document_id}/")

                # Get tag names
                tag_ids = doc_data.get('tags', [])
                tag_names = []
                if tag_ids:
                    tags_data = await self._api_get(session, "tags/")
                    tags_map = {t['id']: t['name'] for t in tags_data.get('results', [])}
                    tag_names = [tags_map.get(tid, str(tid)) for tid in tag_ids]

                # Get correspondent name
                correspondent_name = None
                if doc_data.get('correspondent'):
                    try:
                        corr_data = await self._api_get(session, f"correspondents/{doc_data['correspondent']}/")
                        correspondent_name = corr_data.get('name')
                    except:
                        pass

                # Get document type name
                doc_type_name = None
                if doc_data.get('document_type'):
                    try:
                        type_data = await self._api_get(session, f"document_types/{doc_data['document_type']}/")
                        doc_type_name = type_data.get('name')
                    except:
                        pass

                # Get notes
                notes = []
                try:
                    notes_data = await self._api_get(session, f"documents/{document_id}/notes/")
                    notes = [n.get('note', '') for n in notes_data if n.get('note')]
                except:
                    pass

                result = {
                    "id": document_id,
                    "title": doc_data.get('title', 'Untitled'),
                    "content": doc_data.get('content', ''),
                    "content_length": len(doc_data.get('content', '') or ''),
                    "correspondent": correspondent_name,
                    "document_type": doc_type_name,
                    "tags": tag_names,
                    "created": doc_data.get('created'),
                    "added": doc_data.get('added'),
                    "modified": doc_data.get('modified'),
                    "original_filename": doc_data.get('original_file_name'),
                    "notes": notes,
                    "url": f"{self._get_base_url()}/documents/{document_id}/details"
                }

                await self._emit_citation(__event_emitter__, result)
                await self._emit_status(__event_emitter__, f"Retrieved: {result['title']}", "success", True)

                return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            await self._emit_status(__event_emitter__, f"Error: {str(e)}", "error", True)
            return json.dumps({"error": str(e), "document_id": document_id})

    async def browse(
        self,
        filter_type: str = "",
        filter_value: str = "",
        limit: int = 10,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        Browse documents by category without a search query.

        Use this when the user wants to see documents by tag, correspondent, or type
        without searching for specific content.

        Examples:
        - "Show me all invoices" -> filter_type="document_type", filter_value="invoice"
        - "Documents from Amazon" -> filter_type="correspondent", filter_value="Amazon"
        - "Documents tagged 'taxes'" -> filter_type="tag", filter_value="taxes"
        - "List recent documents" -> no filter, just limit

        :param filter_type: One of: "tag", "correspondent", "document_type", or empty for recent
        :param filter_value: The name to filter by (e.g., tag name, correspondent name)
        :param limit: Maximum documents to return (default: 10)
        :return: List of matching documents
        """
        await self._emit_status(__event_emitter__, "Browsing documents...")

        if not self.valves.paperless_url or not self.valves.api_token:
            return json.dumps({
                "error": "K4mi not configured. Please set paperless_url and api_token in Valves."
            })

        try:
            async with aiohttp.ClientSession() as session:
                params = {"page_size": min(limit, 50), "ordering": "-created"}

                # Apply filter
                filter_applied = None
                if filter_type and filter_value:
                    filter_value_lower = filter_value.lower()

                    if filter_type == "tag":
                        tags_data = await self._api_get(session, "tags/")
                        for t in tags_data.get('results', []):
                            if filter_value_lower in t['name'].lower():
                                params['tags__id'] = t['id']
                                filter_applied = f"tag='{t['name']}'"
                                break

                    elif filter_type == "correspondent":
                        corr_data = await self._api_get(session, "correspondents/")
                        for c in corr_data.get('results', []):
                            if filter_value_lower in c['name'].lower():
                                params['correspondent__id'] = c['id']
                                filter_applied = f"correspondent='{c['name']}'"
                                break

                    elif filter_type == "document_type":
                        types_data = await self._api_get(session, "document_types/")
                        for dt in types_data.get('results', []):
                            if filter_value_lower in dt['name'].lower():
                                params['document_type__id'] = dt['id']
                                filter_applied = f"type='{dt['name']}'"
                                break

                # Fetch documents
                await self._emit_status(__event_emitter__, f"Fetching documents...")
                docs_data = await self._api_get(session, "documents/", params)
                raw_docs = docs_data.get('results', [])

                if not raw_docs:
                    await self._emit_status(__event_emitter__, "No documents found", "info", True)
                    return json.dumps({
                        "message": "No documents found",
                        "filter": filter_applied or "none",
                        "count": 0,
                        "documents": []
                    })

                # Build summary list (no full content for browse)
                documents = []
                for raw in raw_docs:
                    documents.append({
                        "id": raw['id'],
                        "title": raw.get('title', 'Untitled'),
                        "correspondent": raw.get('correspondent_name'),
                        "created": raw.get('created', '')[:10] if raw.get('created') else None,
                        "url": f"{self._get_base_url()}/documents/{raw['id']}/details"
                    })

                await self._emit_status(__event_emitter__, f"Found {len(documents)} documents", "success", True)

                return json.dumps({
                    "filter": filter_applied or "recent",
                    "count": len(documents),
                    "documents": documents,
                    "hint": "Use get_document(id) to retrieve full content of any document."
                }, ensure_ascii=False, indent=2)

        except Exception as e:
            await self._emit_status(__event_emitter__, f"Error: {str(e)}", "error", True)
            return json.dumps({"error": str(e)})

    async def list_categories(
        self,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> str:
        """
        List all available tags, correspondents, and document types.

        Use this when the user wants to know what categories exist for filtering,
        or when they're unsure what to search for.

        :return: Lists of all tags, correspondents (vendors), and document types
        """
        await self._emit_status(__event_emitter__, "Fetching categories...")

        if not self.valves.paperless_url or not self.valves.api_token:
            return json.dumps({
                "error": "K4mi not configured. Please set paperless_url and api_token in Valves."
            })

        try:
            async with aiohttp.ClientSession() as session:
                # Fetch all categories in parallel
                tags_task = self._api_get(session, "tags/")
                corr_task = self._api_get(session, "correspondents/")
                types_task = self._api_get(session, "document_types/")

                tags_data, corr_data, types_data = await asyncio.gather(
                    tags_task, corr_task, types_task,
                    return_exceptions=True
                )

                # Process results
                tags = []
                if not isinstance(tags_data, Exception):
                    tags = [{"name": t['name'], "count": t.get('document_count', 0)}
                            for t in tags_data.get('results', [])]
                    tags.sort(key=lambda x: x['count'], reverse=True)

                correspondents = []
                if not isinstance(corr_data, Exception):
                    correspondents = [{"name": c['name'], "count": c.get('document_count', 0)}
                                     for c in corr_data.get('results', [])]
                    correspondents.sort(key=lambda x: x['count'], reverse=True)

                document_types = []
                if not isinstance(types_data, Exception):
                    document_types = [{"name": dt['name'], "count": dt.get('document_count', 0)}
                                     for dt in types_data.get('results', [])]
                    document_types.sort(key=lambda x: x['count'], reverse=True)

                await self._emit_status(__event_emitter__, "Categories loaded", "success", True)

                return json.dumps({
                    "tags": tags,
                    "correspondents": correspondents,
                    "document_types": document_types,
                    "hint": "Use these names with search() or browse() to filter documents."
                }, ensure_ascii=False, indent=2)

        except Exception as e:
            await self._emit_status(__event_emitter__, f"Error: {str(e)}", "error", True)
            return json.dumps({"error": str(e)})


# Required for asyncio.gather
import asyncio

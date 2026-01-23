"""
title: K4mi Document Search Tool
author: Max R
funding_url: N/A
version: 6.0.0
license: MIT
description: Improved RAG tool for Paperless-ngx with async support and date filtering.
"""

import json
import asyncio
import aiohttp
from datetime import datetime
from urllib.parse import urlparse, urlencode
from pydantic import BaseModel, Field
from typing import Callable, Any, Optional, List, Dict


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class AsyncPaperlessClient:
    """Async client to interact with Paperless-ngx API with optimized performance."""

    def __init__(self, base_url: str, token: str):
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"

        self.base_url = base_url
        self.token = token
        self.headers = {
            "Authorization": f"Token {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Caches
        self._tags_cache: Optional[Dict] = None
        self._correspondents_cache: Optional[Dict] = None
        self._document_types_cache: Optional[Dict] = None
        self._custom_fields_cache: Optional[Dict] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get(self, endpoint: str, params: dict = None) -> dict:
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/api/{endpoint}"

        session = await self._get_session()
        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"API error {response.status}: {text[:500]}")
            return await response.json()

    async def _get_all_pages(
        self, endpoint: str, params: dict = None, max_results: int = None
    ) -> List[dict]:
        """Fetch all pages with optional result limit."""
        results = []
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/api/{endpoint}"

        session = await self._get_session()

        while url:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API error {response.status}: {text[:500]}")

                data = await response.json()
                page_results = data.get("results", [])

                # Preserve search relevance score if present
                for item in page_results:
                    if "__search_hit__" in data:
                        hit = data["__search_hit__"]
                        if str(item.get("id")) in hit:
                            item["_relevance_score"] = hit[str(item["id"])].get("score", 0)

                results.extend(page_results)

                # Check result limit
                if max_results and len(results) >= max_results:
                    results = results[:max_results]
                    break

                url = data.get("next")
                params = None  # Params are included in next URL

        return results

    async def _load_metadata_caches(self):
        """Load all metadata caches concurrently."""
        if self._tags_cache is not None:
            return  # Already loaded

        tasks = [
            self._get_all_pages("tags/"),
            self._get_all_pages("correspondents/"),
            self._get_all_pages("document_types/"),
            self._get_all_pages("custom_fields/"),
        ]

        try:
            tags, correspondents, doc_types, custom_fields = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            self._tags_cache = {t["id"]: t for t in tags} if isinstance(tags, list) else {}
            self._correspondents_cache = {c["id"]: c for c in correspondents} if isinstance(correspondents, list) else {}
            self._document_types_cache = {d["id"]: d for d in doc_types} if isinstance(doc_types, list) else {}
            self._custom_fields_cache = {f["id"]: f for f in custom_fields} if isinstance(custom_fields, list) else {}
        except Exception:
            self._tags_cache = {}
            self._correspondents_cache = {}
            self._document_types_cache = {}
            self._custom_fields_cache = {}

    def get_tags(self) -> dict:
        return self._tags_cache or {}

    def get_correspondents(self) -> dict:
        return self._correspondents_cache or {}

    def get_document_types(self) -> dict:
        return self._document_types_cache or {}

    def get_custom_fields(self) -> dict:
        return self._custom_fields_cache or {}

    async def get_document(self, doc_id: int) -> dict:
        return await self._get(f"documents/{doc_id}/")

    async def get_document_notes(self, doc_id: int) -> list:
        try:
            data = await self._get(f"documents/{doc_id}/notes/")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    async def search_documents(
        self,
        query: str = None,
        tag_id: int = None,
        correspondent_id: int = None,
        document_type_id: int = None,
        created_after: str = None,
        created_before: str = None,
        ordering: str = None,
        max_results: int = None,
    ) -> List[dict]:
        """
        Search documents with various filters.

        Args:
            query: Full-text search query
            tag_id: Filter by tag ID
            correspondent_id: Filter by correspondent ID
            document_type_id: Filter by document type ID
            created_after: Filter documents created after this date (YYYY-MM-DD)
            created_before: Filter documents created before this date (YYYY-MM-DD)
            ordering: Sort order (e.g., '-created', 'title', '-added')
            max_results: Maximum number of results to return
        """
        params = {}

        if query:
            params["query"] = query
        if tag_id:
            params["tags__id"] = tag_id
        if correspondent_id:
            params["correspondent__id"] = correspondent_id
        if document_type_id:
            params["document_type__id"] = document_type_id
        if created_after:
            params["created__date__gte"] = created_after
        if created_before:
            params["created__date__lte"] = created_before
        if ordering:
            params["ordering"] = ordering

        return await self._get_all_pages("documents/", params, max_results)

    def find_tag_by_name(self, name: str) -> Optional[int]:
        tags = self.get_tags()
        name_lower = name.lower().strip()

        # Exact match first
        for tag_id, tag_info in tags.items():
            if tag_info["name"].lower() == name_lower:
                return tag_id

        # Partial match fallback
        for tag_id, tag_info in tags.items():
            if name_lower in tag_info["name"].lower():
                return tag_id
        return None

    def find_correspondent_by_name(self, name: str) -> Optional[int]:
        correspondents = self.get_correspondents()
        name_lower = name.lower().strip()

        # Exact match first
        for c_id, c_info in correspondents.items():
            if c_info["name"].lower() == name_lower:
                return c_id

        # Partial match fallback
        for c_id, c_info in correspondents.items():
            if name_lower in c_info["name"].lower():
                return c_id
        return None

    def find_document_type_by_name(self, name: str) -> Optional[int]:
        doc_types = self.get_document_types()
        name_lower = name.lower().strip()

        for dt_id, dt_info in doc_types.items():
            if dt_info["name"].lower() == name_lower:
                return dt_id

        for dt_id, dt_info in doc_types.items():
            if name_lower in dt_info["name"].lower():
                return dt_id
        return None

    async def enrich_documents_batch(
        self,
        docs: List[dict],
        include_content: bool = True,
        include_notes: bool = False,
        max_concurrent: int = 10
    ) -> List[dict]:
        """
        Enrich multiple documents concurrently for better performance.

        Args:
            docs: List of document dicts from search
            include_content: Whether to fetch full content (requires individual API calls)
            include_notes: Whether to fetch notes (additional API calls)
            max_concurrent: Max concurrent requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def enrich_single(doc: dict) -> dict:
            async with semaphore:
                return await self._enrich_document(doc, include_content, include_notes)

        tasks = [enrich_single(doc) for doc in docs]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _enrich_document(
        self,
        doc: dict,
        fetch_full: bool = True,
        fetch_notes: bool = False
    ) -> dict:
        """Enrich a single document with resolved names and optionally full data."""
        tags_lookup = self.get_tags()
        correspondents_lookup = self.get_correspondents()
        doc_types_lookup = self.get_document_types()
        custom_fields_lookup = self.get_custom_fields()

        enriched = dict(doc)

        # Fetch full document for content and custom_fields if needed
        if fetch_full:
            try:
                full_doc = await self.get_document(doc["id"])
                enriched.update(full_doc)
            except Exception as e:
                enriched["_fetch_error"] = str(e)

        # Resolve tags
        tag_ids = enriched.get("tags", [])
        enriched["tag_names"] = [
            tags_lookup[tid]["name"]
            for tid in tag_ids
            if tid in tags_lookup
        ]

        # Resolve correspondent
        correspondent_id = enriched.get("correspondent")
        if correspondent_id and correspondent_id in correspondents_lookup:
            enriched["correspondent_name"] = correspondents_lookup[correspondent_id].get("name")
        else:
            enriched["correspondent_name"] = None

        # Resolve document type
        doc_type_id = enriched.get("document_type")
        if doc_type_id and doc_type_id in doc_types_lookup:
            enriched["document_type_name"] = doc_types_lookup[doc_type_id].get("name")
        else:
            enriched["document_type_name"] = None

        # Resolve custom fields
        custom_fields_raw = enriched.get("custom_fields", [])
        enriched["custom_fields_dict"] = {}
        for cf in custom_fields_raw:
            field_id = cf.get("field")
            field_value = cf.get("value")
            if field_id in custom_fields_lookup:
                field_name = custom_fields_lookup[field_id].get("name")
                enriched["custom_fields_dict"][field_name] = field_value

        # Fetch notes if requested
        if fetch_notes:
            try:
                notes = await self.get_document_notes(doc["id"])
                enriched["notes"] = [n.get("note", "") for n in notes if n.get("note")]
            except Exception:
                enriched["notes"] = []
        else:
            enriched["notes"] = []

        # Add URL
        enriched["url"] = f"{self.base_url}/documents/{doc['id']}/details"

        return enriched


def smart_truncate(text: str, max_length: int, preserve_end: bool = True) -> str:
    """
    Intelligently truncate text while trying to preserve complete sentences.

    Args:
        text: The text to truncate
        max_length: Maximum length
        preserve_end: If True, also include the end of the text (useful for totals/summaries)
    """
    if not text or len(text) <= max_length:
        return text

    if not preserve_end:
        # Simple truncation at sentence boundary
        truncated = text[:max_length]
        # Try to end at a sentence
        for sep in ['. ', '.\n', '\n\n', '\n']:
            last_sep = truncated.rfind(sep)
            if last_sep > max_length * 0.7:  # Don't cut too much
                return truncated[:last_sep + 1] + "\n[...truncated...]"
        return truncated + "\n[...truncated...]"

    # Preserve both start and end (useful for invoices where totals are at the end)
    start_length = int(max_length * 0.6)
    end_length = int(max_length * 0.35)

    start_text = text[:start_length]
    end_text = text[-end_length:]

    # Try to cut at sentence boundaries
    for sep in ['. ', '.\n', '\n\n', '\n']:
        last_sep = start_text.rfind(sep)
        if last_sep > start_length * 0.7:
            start_text = start_text[:last_sep + 1]
            break

    for sep in ['. ', '.\n', '\n\n', '\n']:
        first_sep = end_text.find(sep)
        if first_sep != -1 and first_sep < end_length * 0.3:
            end_text = end_text[first_sep + 1:]
            break

    return f"{start_text}\n\n[...content truncated...]\n\n{end_text}"


def extract_key_content(text: str, max_length: int = 4000) -> str:
    """
    Extract the most relevant content from OCR text, prioritizing:
    - Headers and titles
    - Amounts, totals, and financial data
    - Dates
    - Names and addresses
    """
    if not text or len(text) <= max_length:
        return text

    lines = text.split('\n')

    # Scoring patterns for important lines
    important_patterns = [
        ('total', 10),
        ('amount', 8),
        ('subtotal', 8),
        ('tax', 7),
        ('invoice', 7),
        ('date', 6),
        ('due', 6),
        ('payment', 6),
        ('balance', 6),
        ('$', 5),
        ('€', 5),
        ('£', 5),
    ]

    scored_lines = []
    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = 0

        # Score based on patterns
        for pattern, points in important_patterns:
            if pattern in line_lower:
                score += points

        # Boost lines with numbers (likely amounts)
        if any(c.isdigit() for c in line):
            score += 2

        # Boost header-like lines (short, possibly uppercase)
        if len(line.strip()) < 50 and line.strip().isupper():
            score += 3

        scored_lines.append((score, i, line))

    # Sort by score (descending) then by position (ascending) for ties
    scored_lines.sort(key=lambda x: (-x[0], x[1]))

    # Take top-scoring lines up to max_length
    selected_indices = set()
    current_length = 0

    for score, idx, line in scored_lines:
        if current_length + len(line) + 1 > max_length:
            continue
        selected_indices.add(idx)
        current_length += len(line) + 1

    # Rebuild text preserving original order
    result_lines = [lines[i] for i in sorted(selected_indices)]

    if len(selected_indices) < len(lines):
        result_lines.append("\n[...some content omitted for brevity...]")

    return '\n'.join(result_lines)


class Tools:
    class Valves(BaseModel):
        K4MI_URL: str = Field(
            default="https://k4mi.yourdomain.com/",
            description="The base URL of your K4mi/Paperless-ngx instance",
        )
        K4MI_TOKEN: str = Field(
            default="",
            description="API token for K4mi/Paperless authentication",
        )
        DEFAULT_MAX_RESULTS: int = Field(
            default=25,
            description="Default maximum number of documents to return (prevents overwhelming the LLM)",
        )
        MAX_CONTENT_LENGTH: int = Field(
            default=5000,
            description="Maximum characters of content per document",
        )
        ENABLE_SMART_EXTRACTION: bool = Field(
            default=True,
            description="Use intelligent content extraction to prioritize important text",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_documents(
        self,
        query: str = None,
        tag: str = None,
        vendor: str = None,
        document_type: str = None,
        date_from: str = None,
        date_to: str = None,
        max_results: int = None,
        include_full_content: bool = False,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search for documents in K4mi/Paperless-ngx. Use when user asks about documents, invoices, receipts, or files.

        :param query: Full-text search in document content (e.g. "insurance claim", "order 12345").
        :param tag: Filter by tag name (e.g. "2024", "expenses"). Partial match supported.
        :param vendor: Filter by vendor/correspondent name (e.g. "Amazon", "Office Depot").
        :param document_type: Filter by type (e.g. "Invoice", "Receipt", "Contract").
        :param date_from: Documents created on or after this date (YYYY-MM-DD format).
        :param date_to: Documents created on or before this date (YYYY-MM-DD format).
        :param max_results: Maximum documents to return. Default is 25.
        :param include_full_content: Set True for detailed content analysis with more text.
        :return: JSON with document data including content, metadata, and custom fields.
        """
        emitter = EventEmitter(__event_emitter__)

        # Use default max if not specified
        if max_results is None:
            max_results = self.valves.DEFAULT_MAX_RESULTS

        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            # Build filter description
            filters = []
            if query:
                filters.append(f"search='{query}'")
            if tag:
                filters.append(f"tag='{tag}'")
            if vendor:
                filters.append(f"vendor='{vendor}'")
            if document_type:
                filters.append(f"type='{document_type}'")
            if date_from:
                filters.append(f"from={date_from}")
            if date_to:
                filters.append(f"to={date_to}")

            filter_str = ", ".join(filters) if filters else "no filters"
            await emitter.progress_update(f"Searching: {filter_str}")

            # Load metadata caches concurrently
            await emitter.progress_update("Loading metadata...")
            await client._load_metadata_caches()

            # Resolve filter names to IDs
            tag_id = None
            if tag:
                tag_id = client.find_tag_by_name(tag)
                if tag_id is None:
                    available = [t["name"] for t in client.get_tags().values()][:30]
                    await emitter.error_update(f"Tag '{tag}' not found")
                    return json.dumps({
                        "error": f"Tag '{tag}' not found",
                        "available_tags": available,
                        "suggestion": "Use list_tags() to see all available tags"
                    })

            correspondent_id = None
            if vendor:
                correspondent_id = client.find_correspondent_by_name(vendor)
                if correspondent_id is None:
                    available = [c["name"] for c in client.get_correspondents().values()][:30]
                    await emitter.error_update(f"Vendor '{vendor}' not found")
                    return json.dumps({
                        "error": f"Vendor '{vendor}' not found",
                        "available_vendors": available,
                        "suggestion": "Use list_vendors() to see all available vendors"
                    })

            doc_type_id = None
            if document_type:
                doc_type_id = client.find_document_type_by_name(document_type)
                if doc_type_id is None:
                    available = [d["name"] for d in client.get_document_types().values()]
                    await emitter.error_update(f"Document type '{document_type}' not found")
                    return json.dumps({
                        "error": f"Document type '{document_type}' not found",
                        "available_types": available
                    })

            # Search with ordering by relevance/date
            await emitter.progress_update("Searching documents...")
            ordering = None if query else "-created"  # Let search handle relevance, otherwise newest first

            raw_documents = await client.search_documents(
                query=query,
                tag_id=tag_id,
                correspondent_id=correspondent_id,
                document_type_id=doc_type_id,
                created_after=date_from,
                created_before=date_to,
                ordering=ordering,
                max_results=max_results,
            )

            if not raw_documents:
                await emitter.error_update("No documents found")
                return json.dumps({
                    "error": "No documents found matching your criteria",
                    "filters_applied": filter_str,
                    "suggestions": [
                        "Try broader search terms",
                        "Check tag/vendor spelling with list_tags() or list_vendors()",
                        "Expand date range if filtering by date"
                    ]
                })

            # Enrich documents concurrently
            await emitter.progress_update(f"Processing {len(raw_documents)} documents...")
            enriched_docs = await client.enrich_documents_batch(
                raw_documents,
                include_content=True,
                include_notes=False,  # Skip notes for performance unless needed
                max_concurrent=10
            )

            # Format results
            documents = []
            for enriched in enriched_docs:
                content = enriched.get("content", "")

                # Process content based on settings
                if content:
                    max_len = self.valves.MAX_CONTENT_LENGTH
                    if include_full_content:
                        max_len = max_len * 2  # Allow more for full content

                    if self.valves.ENABLE_SMART_EXTRACTION and not include_full_content:
                        content = extract_key_content(content, max_len)
                    elif len(content) > max_len:
                        content = smart_truncate(content, max_len, preserve_end=True)

                doc_data = {
                    "id": enriched.get("id"),
                    "title": enriched.get("title"),
                    "file_name": enriched.get("original_file_name") or enriched.get("archived_file_name"),
                    "vendor": enriched.get("correspondent_name"),
                    "document_type": enriched.get("document_type_name"),
                    "tags": enriched.get("tag_names", []),
                    "created": enriched.get("created"),
                    "added": enriched.get("added"),
                    "custom_fields": enriched.get("custom_fields_dict", {}),
                    "content": content,
                    "url": enriched.get("url"),
                }

                # Include relevance score if present
                if "_relevance_score" in enriched:
                    doc_data["relevance_score"] = enriched["_relevance_score"]

                documents.append(doc_data)

                # Emit citation for Open WebUI
                if __event_emitter__:
                    snippet = content[:500] if content else doc_data["title"] or "Document"
                    await __event_emitter__({
                        "type": "citation",
                        "data": {
                            "document": [snippet],
                            "metadata": [{"source": doc_data["file_name"] or doc_data["title"]}],
                            "source": {"name": doc_data["url"]},
                        },
                    })

            await emitter.success_update(f"Found {len(documents)} documents")

            result = {
                "success": True,
                "filters_applied": filter_str,
                "document_count": len(documents),
                "documents": documents,
                "analysis_instructions": (
                    "IMPORTANT: Analyze the 'content' field in each document - this contains the OCR text with actual data like: "
                    "Invoice amounts, line items, totals, and taxes; Dates, due dates, invoice numbers; "
                    "Vendor details, addresses, contact info; Product descriptions and quantities. "
                    "The 'custom_fields' may contain pre-extracted structured data. "
                    "Create the output format the user requested (table, summary, list, etc.) by extracting relevant information."
                )
            }

            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e), "type": type(e).__name__})

        finally:
            await client.close()

    async def get_document_by_id(
        self,
        document_id: int,
        include_notes: bool = True,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Get complete details for a specific document by its ID.
        Use this when you need full information about a document found in search results.

        :param document_id: The numeric ID of the document.
        :param include_notes: Whether to fetch document notes.
        :return: Complete document data with all fields, full content, and notes.
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update(f"Fetching document {document_id}...")

            await client._load_metadata_caches()
            doc = await client.get_document(document_id)
            enriched = await client._enrich_document(doc, fetch_full=False, fetch_notes=include_notes)

            await emitter.success_update(f"Retrieved: {enriched.get('title', 'Untitled')}")
            return json.dumps(enriched, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

    async def quick_search(
        self,
        query: str,
        max_results: int = 10,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Fast search that returns basic document info without fetching full content.
        Use this for quick lookups or when you just need to list matching documents.

        For detailed content analysis, use search_documents() with include_full_content=True.

        :param query: Search query for document titles and content.
        :param max_results: Maximum results to return (default: 10).
        :return: List of matching documents with basic metadata (no content).
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update(f"Quick search: '{query}'")

            await client._load_metadata_caches()

            raw_docs = await client.search_documents(
                query=query,
                max_results=max_results
            )

            if not raw_docs:
                await emitter.error_update("No documents found")
                return json.dumps({"documents": [], "count": 0})

            # Basic enrichment without fetching full document
            documents = []
            tags_lookup = client.get_tags()
            correspondents_lookup = client.get_correspondents()

            for doc in raw_docs:
                tag_names = [
                    tags_lookup[tid]["name"]
                    for tid in doc.get("tags", [])
                    if tid in tags_lookup
                ]

                correspondent_name = None
                if doc.get("correspondent") in correspondents_lookup:
                    correspondent_name = correspondents_lookup[doc["correspondent"]]["name"]

                documents.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "vendor": correspondent_name,
                    "tags": tag_names,
                    "created": doc.get("created"),
                    "url": f"{client.base_url}/documents/{doc['id']}/details"
                })

            await emitter.success_update(f"Found {len(documents)} documents")

            return json.dumps({
                "documents": documents,
                "count": len(documents),
                "note": "Use search_documents() or get_document_by_id() for full content"
            }, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

    async def list_tags(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all available tags in K4mi/Paperless.
        Use this to discover what tags exist before searching.

        :return: List of all tag names with document counts.
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update("Fetching tags...")

            tags = await client._get_all_pages("tags/")
            tag_info = [
                {"name": t["name"], "document_count": t.get("document_count", 0)}
                for t in sorted(tags, key=lambda x: x.get("document_count", 0), reverse=True)
            ]

            await emitter.success_update(f"Found {len(tags)} tags")
            return json.dumps({
                "tags": tag_info,
                "count": len(tags)
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

    async def list_vendors(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all vendors/correspondents in K4mi/Paperless.
        Use this to discover what vendors exist before searching.

        :return: List of all vendor names with document counts.
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update("Fetching vendors...")

            correspondents = await client._get_all_pages("correspondents/")
            vendor_info = [
                {"name": c["name"], "document_count": c.get("document_count", 0)}
                for c in sorted(correspondents, key=lambda x: x.get("document_count", 0), reverse=True)
            ]

            await emitter.success_update(f"Found {len(correspondents)} vendors")
            return json.dumps({
                "vendors": vendor_info,
                "count": len(correspondents)
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

    async def list_document_types(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all document types in K4mi/Paperless.
        Use this to discover what document types exist (Invoice, Receipt, Contract, etc.).

        :return: List of all document type names.
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update("Fetching document types...")

            doc_types = await client._get_all_pages("document_types/")
            type_info = [
                {"name": d["name"], "document_count": d.get("document_count", 0)}
                for d in sorted(doc_types, key=lambda x: x.get("document_count", 0), reverse=True)
            ]

            await emitter.success_update(f"Found {len(doc_types)} document types")
            return json.dumps({
                "document_types": type_info,
                "count": len(doc_types)
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

    async def list_custom_fields(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all custom fields defined in K4mi/Paperless.
        Custom fields may contain extracted data like invoice amounts, tax, etc.

        :return: List of custom field names and their data types.
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update("Fetching custom fields...")

            custom_fields = await client._get_all_pages("custom_fields/")
            fields_info = [
                {
                    "name": cf["name"],
                    "data_type": cf.get("data_type"),
                }
                for cf in custom_fields
            ]

            await emitter.success_update(f"Found {len(custom_fields)} custom fields")
            return json.dumps({
                "custom_fields": fields_info,
                "count": len(custom_fields)
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

    async def get_statistics(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Get an overview of the K4mi/Paperless document collection.
        Shows total documents, tags, vendors, and document types.

        :return: Statistics about the document collection.
        """
        emitter = EventEmitter(__event_emitter__)
        client = AsyncPaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

        try:
            await emitter.progress_update("Gathering statistics...")

            # Fetch counts concurrently
            await client._load_metadata_caches()

            # Get document count from a minimal search
            docs = await client.search_documents(max_results=1)

            # Try to get statistics endpoint if available
            try:
                stats = await client._get("statistics/")
            except Exception:
                stats = {}

            result = {
                "tags_count": len(client.get_tags()),
                "vendors_count": len(client.get_correspondents()),
                "document_types_count": len(client.get_document_types()),
                "custom_fields_count": len(client.get_custom_fields()),
            }

            if stats:
                result.update({
                    "total_documents": stats.get("documents_total"),
                    "documents_in_inbox": stats.get("documents_inbox"),
                })

            await emitter.success_update("Statistics retrieved")
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as e:
            await emitter.error_update(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        finally:
            await client.close()

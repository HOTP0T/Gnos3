"""
title: K4mi Document Search Tool
author: Max R
funding_url: N/A
version: 5.1.0
license: MIT
"""

import json
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from typing import Callable, Any, Optional


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


class PaperlessClient:
    """Client to interact with K4mi/Paperless API"""

    def __init__(self, base_url: str, token: str):
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"

        self.base_url = base_url
        self.token = token

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.session.max_redirects = 10

        # Caches
        self._tags_cache = None
        self._correspondents_cache = None
        self._document_types_cache = None
        self._custom_fields_cache = None
        self._storage_paths_cache = None

    def _make_request(self, url: str, params: dict = None) -> requests.Response:
        """Make a GET request, handling redirects manually to preserve auth headers"""
        max_redirects = 5
        current_url = url

        for _ in range(max_redirects):
            response = self.session.get(
                current_url, params=params, allow_redirects=False
            )
            if response.status_code not in (301, 302, 303, 307, 308):
                return response

            redirect_url = response.headers.get("Location")
            if not redirect_url:
                return response

            if redirect_url.startswith("/"):
                parsed = urlparse(current_url)
                redirect_url = f"{parsed.scheme}://{parsed.netloc}{redirect_url}"

            current_url = redirect_url
            params = None

        return response

    def _get(self, endpoint: str, params: dict = None) -> dict:
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/api/{endpoint}"
        response = self._make_request(url, params)

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text[:500]}")

        return response.json()

    def _get_all_pages(self, endpoint: str, params: dict = None) -> list:
        results = []
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/api/{endpoint}"

        while url:
            response = self._make_request(url, params)
            if response.status_code != 200:
                raise Exception(
                    f"API error {response.status_code}: {response.text[:500]}"
                )

            data = response.json()
            results.extend(data.get("results", []))
            url = data.get("next")
            params = None

        return results

    def get_tags(self) -> dict:
        if self._tags_cache is None:
            tags = self._get_all_pages("tags/")
            self._tags_cache = {tag["id"]: tag for tag in tags}
        return self._tags_cache

    def get_correspondents(self) -> dict:
        if self._correspondents_cache is None:
            correspondents = self._get_all_pages("correspondents/")
            self._correspondents_cache = {c["id"]: c for c in correspondents}
        return self._correspondents_cache

    def get_document_types(self) -> dict:
        if self._document_types_cache is None:
            doc_types = self._get_all_pages("document_types/")
            self._document_types_cache = {dt["id"]: dt for dt in doc_types}
        return self._document_types_cache

    def get_custom_fields(self) -> dict:
        if self._custom_fields_cache is None:
            try:
                custom_fields = self._get_all_pages("custom_fields/")
                self._custom_fields_cache = {cf["id"]: cf for cf in custom_fields}
            except Exception:
                self._custom_fields_cache = {}
        return self._custom_fields_cache

    def get_storage_paths(self) -> dict:
        if self._storage_paths_cache is None:
            try:
                storage_paths = self._get_all_pages("storage_paths/")
                self._storage_paths_cache = {sp["id"]: sp for sp in storage_paths}
            except Exception:
                self._storage_paths_cache = {}
        return self._storage_paths_cache

    def get_document_notes(self, doc_id: int) -> list:
        try:
            data = self._get(f"documents/{doc_id}/notes/")
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_document(self, doc_id: int) -> dict:
        return self._get(f"documents/{doc_id}/")

    def search_documents(
        self,
        query: str = None,
        tag_id: int = None,
        correspondent_id: int = None,
        document_type_id: int = None,
    ) -> list:
        params = {}
        if query:
            params["query"] = query
        if tag_id:
            params["tags__id"] = tag_id
        if correspondent_id:
            params["correspondent__id"] = correspondent_id
        if document_type_id:
            params["document_type__id"] = document_type_id

        return self._get_all_pages("documents/", params)

    def find_tag_by_name(self, name: str) -> Optional[int]:
        tags = self.get_tags()
        name_lower = name.lower()
        for tag_id, tag_info in tags.items():
            if name_lower in tag_info["name"].lower():
                return tag_id
        return None

    def find_correspondent_by_name(self, name: str) -> Optional[int]:
        correspondents = self.get_correspondents()
        name_lower = name.lower()
        for c_id, c_info in correspondents.items():
            if name_lower in c_info["name"].lower():
                return c_id
        return None

    def find_document_type_by_name(self, name: str) -> Optional[int]:
        doc_types = self.get_document_types()
        name_lower = name.lower()
        for dt_id, dt_info in doc_types.items():
            if name_lower in dt_info["name"].lower():
                return dt_id
        return None

    def enrich_document(self, doc: dict, fetch_full: bool = True) -> dict:
        """
        Enrich a document with resolved names and full data.

        IMPORTANT: The search API only returns partial data. We must fetch
        the full document to get custom_fields and content.
        """
        tags_lookup = self.get_tags()
        correspondents_lookup = self.get_correspondents()
        doc_types_lookup = self.get_document_types()
        custom_fields_lookup = self.get_custom_fields()

        # Start with search result data
        enriched = dict(doc)

        # CRITICAL: Fetch full document to get custom_fields and content
        # The search API does NOT return these fields!
        if fetch_full:
            try:
                full_doc = self.get_document(doc["id"])
                enriched.update(full_doc)
            except Exception:
                pass

        # Resolve tags
        tag_ids = enriched.get("tags", [])
        enriched["tag_names"] = []
        for tid in tag_ids:
            if tid in tags_lookup:
                enriched["tag_names"].append(tags_lookup[tid]["name"])

        # Resolve correspondent
        correspondent_id = enriched.get("correspondent")
        if correspondent_id and correspondent_id in correspondents_lookup:
            enriched["correspondent_name"] = correspondents_lookup[
                correspondent_id
            ].get("name")
        else:
            enriched["correspondent_name"] = None

        # Resolve document type
        doc_type_id = enriched.get("document_type")
        if doc_type_id and doc_type_id in doc_types_lookup:
            enriched["document_type_name"] = doc_types_lookup[doc_type_id].get("name")
        else:
            enriched["document_type_name"] = None

        # Resolve custom fields - these come from the FULL document fetch
        custom_fields_raw = enriched.get("custom_fields", [])
        enriched["custom_fields_dict"] = {}
        for cf in custom_fields_raw:
            field_id = cf.get("field")
            field_value = cf.get("value")
            if field_id in custom_fields_lookup:
                field_def = custom_fields_lookup[field_id]
                enriched["custom_fields_dict"][field_def.get("name")] = field_value
            else:
                # Include even if we don't have the field definition
                enriched["custom_fields_dict"][f"field_{field_id}"] = field_value

        # Fetch notes
        try:
            notes = self.get_document_notes(doc["id"])
            enriched["notes"] = [
                note.get("note", "") for note in notes if note.get("note")
            ]
        except Exception:
            enriched["notes"] = []

        # Add URL
        enriched["url"] = f"{self.base_url}/documents/{doc['id']}/details"

        return enriched


class Tools:
    class Valves(BaseModel):
        K4MI_URL: str = Field(
            default="https://k4mi.yourdomain.com/",
            description="The base URL of your K4mi/Paperless instance",
        )
        K4MI_TOKEN: str = Field(
            default="",
            description="API token for K4mi authentication",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def search_documents(
        self,
        tag: str = None,
        vendor: str = None,
        search_text: str = None,
        include_content: bool = False,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search for documents in K4mi/Paperless. Returns ALL available data for each document.

        The LLM should analyze the returned data and extract whatever fields the user asked for.
        Each document includes: file_name, title, correspondent (vendor), tags, dates, notes,
        ALL custom fields, and optionally the full OCR content.

        IMPORTANT - Parameter usage:
        - To find documents with a specific tag like "project1", pass: tag="project1"
        - To find documents from a specific vendor, pass: vendor="Company Name"
        - To search document contents, pass: search_text="search terms"
        - To include full OCR text content, pass: include_content=True

        Example calls:
        - User asks "invoices tagged project1" → call with tag="project1"
        - User asks "invoices from Amazon" → call with vendor="Amazon"
        - User asks "find documents mentioning insurance" → call with search_text="insurance"

        :param tag: Filter documents by tag name (e.g., "project1", "expenses", "2024").
        :param vendor: Filter by vendor/correspondent name (e.g., "Amazon", "Office Depot").
        :param search_text: Search within document contents.
        :param include_content: If True, includes full OCR text content for each document.
        :return: JSON with ALL available data for each document. The LLM should extract relevant fields based on user request.
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            # Build description of applied filters
            filters_applied = []
            if tag:
                filters_applied.append(f"tag='{tag}'")
            if vendor:
                filters_applied.append(f"vendor='{vendor}'")
            if search_text:
                filters_applied.append(f"search='{search_text}'")

            filter_str = (
                ", ".join(filters_applied)
                if filters_applied
                else "NO FILTERS (returning all documents)"
            )
            await emitter.progress_update(f"Searching with: {filter_str}")

            # Initialize client
            client = PaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

            # Resolve tag name to ID
            tag_id = None
            if tag:
                tag_id = client.find_tag_by_name(tag)
                if tag_id is None:
                    available_tags = [t["name"] for t in client.get_tags().values()]
                    error_msg = f"ERROR: Tag '{tag}' not found. Available tags: {', '.join(available_tags[:20])}"
                    await emitter.error_update(error_msg)
                    return json.dumps(
                        {"error": error_msg, "available_tags": available_tags}
                    )

            # Resolve vendor name to ID
            correspondent_id = None
            if vendor:
                correspondent_id = client.find_correspondent_by_name(vendor)
                if correspondent_id is None:
                    available = [
                        c["name"] for c in client.get_correspondents().values()
                    ]
                    error_msg = f"ERROR: Vendor '{vendor}' not found. Available vendors: {', '.join(available[:20])}"
                    await emitter.error_update(error_msg)
                    return json.dumps(
                        {"error": error_msg, "available_vendors": available}
                    )

            # Search documents with filters
            await emitter.progress_update("Fetching documents from K4mi...")
            raw_documents = client.search_documents(
                query=search_text, tag_id=tag_id, correspondent_id=correspondent_id
            )

            if not raw_documents:
                msg = f"No documents found matching: {filter_str}"
                await emitter.error_update(msg)
                return json.dumps(
                    {"error": msg, "filters_applied": filter_str, "document_count": 0}
                )

            # Process documents - return ALL available data
            await emitter.progress_update(
                f"Processing {len(raw_documents)} documents..."
            )

            documents = []
            for doc in raw_documents:
                enriched = client.enrich_document(doc)

                # Build complete document data - let LLM figure out what to use
                doc_data = {
                    # Basic info
                    "id": enriched.get("id"),
                    "file_name": enriched.get("original_file_name")
                    or enriched.get("archived_file_name"),
                    "title": enriched.get("title"),
                    # Correspondent/Vendor
                    "correspondent": enriched.get("correspondent_name"),
                    # Classification
                    "document_type": enriched.get("document_type_name"),
                    "tags": enriched.get("tag_names", []),
                    # Dates
                    "created": enriched.get("created"),
                    "added": enriched.get("added"),
                    "modified": enriched.get("modified"),
                    # Custom fields - ALL of them with their names and values
                    "custom_fields": enriched.get("custom_fields_dict", {}),
                    # Notes
                    "notes": enriched.get("notes", []),
                    # URL to view in Paperless
                    "url": enriched.get("url"),
                }

                # Always include OCR content - this is where the actual data lives!
                content = enriched.get("content", "")
                if content:
                    # Limit content length to avoid overwhelming the LLM
                    max_len = 8000 if include_content else 3000
                    if len(content) > max_len:
                        content = content[:max_len] + "\n\n[... content truncated ...]"
                    doc_data["content"] = content

                documents.append(doc_data)

                # Emit citation
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [
                                    (
                                        enriched.get("content", "")[:300]
                                        if enriched.get("content")
                                        else doc_data["title"]
                                    )
                                ],
                                "metadata": [
                                    {
                                        "source": doc_data["file_name"]
                                        or doc_data["title"]
                                    }
                                ],
                                "source": {"name": doc_data["url"]},
                            },
                        }
                    )

            await emitter.success_update(f"Found {len(documents)} documents")

            # Return ALL data - let the LLM be smart about extracting what user needs
            result = {
                "filters_applied": filter_str,
                "document_count": len(documents),
                "documents": documents,
                "instructions": """
You have received ALL available data for each document including the OCR content.
Analyze the data and create the output the user requested (table, summary, etc.).

IMPORTANT - Where to find data:
1. 'content' - The OCR text from the scanned document. This contains the actual invoice data like amounts, totals, tax, vendor names, dates, etc. ALWAYS look here first!
2. 'custom_fields' - Structured fields that may contain extracted data.
3. 'correspondent' - The vendor/company name if assigned in Paperless.
4. 'title' and 'file_name' - May contain vendor names or invoice info.

Parse the 'content' field to extract amounts, taxes, totals, and other financial data that the user requested.
Use your intelligence to find and extract the relevant information from wherever it exists.
""",
            }

            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            return json.dumps({"error": error_msg})

    async def get_document_details(
        self,
        document_id: int,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Get full details for a specific document by ID.

        :param document_id: The numeric ID of the document.
        :return: Complete document data including content, custom fields, and notes.
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            await emitter.progress_update(f"Fetching document {document_id}...")
            client = PaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

            doc = client.get_document(document_id)
            enriched = client.enrich_document(doc)

            await emitter.success_update(
                f"Retrieved: {enriched.get('title', 'Untitled')}"
            )
            return json.dumps(enriched, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            return json.dumps({"error": error_msg})

    async def list_tags(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all available tags in K4mi. Use this to see what tags exist.

        :return: List of all tag names.
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            await emitter.progress_update("Fetching tags...")
            client = PaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

            tags = list(client.get_tags().values())
            tag_names = [t["name"] for t in tags]

            await emitter.success_update(f"Found {len(tags)} tags")
            return json.dumps(
                {"tags": tag_names, "count": len(tags)}, ensure_ascii=False
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            return json.dumps({"error": error_msg})

    async def list_vendors(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all vendors/correspondents in K4mi. Use this to see what vendors exist.

        :return: List of all vendor names.
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            await emitter.progress_update("Fetching vendors...")
            client = PaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

            correspondents = list(client.get_correspondents().values())
            vendor_names = [c["name"] for c in correspondents]

            await emitter.success_update(f"Found {len(correspondents)} vendors")
            return json.dumps(
                {"vendors": vendor_names, "count": len(correspondents)},
                ensure_ascii=False,
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            return json.dumps({"error": error_msg})

    async def list_custom_fields(
        self,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        List all custom fields defined in K4mi (e.g., invoice amounts, tax fields).

        :return: List of custom field names and types.
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            await emitter.progress_update("Fetching custom fields...")
            client = PaperlessClient(self.valves.K4MI_URL, self.valves.K4MI_TOKEN)

            custom_fields = list(client.get_custom_fields().values())
            fields_info = [
                {"name": cf["name"], "type": cf.get("data_type")}
                for cf in custom_fields
            ]

            await emitter.success_update(f"Found {len(custom_fields)} custom fields")
            return json.dumps(
                {"custom_fields": fields_info, "count": len(custom_fields)},
                ensure_ascii=False,
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            return json.dumps({"error": error_msg})

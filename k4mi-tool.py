"""
title: K4mi Document Search Tool
author: Max R
funding_url: N/A
version: 6.0.0
license: MIT
"""

import json
import requests
import time
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from typing import Callable, Any, Optional, List


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None, debug: bool = False):
        self.event_emitter = event_emitter
        self.debug = debug
        self.debug_logs: List[str] = []

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def debug_log(self, message: str):
        """Log debug message - collected and returned in response"""
        self.debug_logs.append(f"[DEBUG] {message}")
        if self.debug:
            await self.emit(f"[DEBUG] {message}")

    def get_debug_logs(self) -> List[str]:
        return self.debug_logs

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

    def __init__(self, base_url: str, token: str, debug_callback=None):
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"

        self.base_url = base_url
        self.token = token
        self.debug_callback = debug_callback  # Async function to log debug info

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self.session.max_redirects = 10

        # Caches
        self._tags_cache = None
        self._correspondents_cache = None
        self._document_types_cache = None
        self._custom_fields_cache = None
        self._storage_paths_cache = None

    def _log(self, message: str):
        """Synchronous debug log - stores for later async emission"""
        if self.debug_callback:
            self.debug_callback(message)

    def _make_request(self, url: str, params: dict = None, retries: int = 3) -> requests.Response:
        """Make a GET request with retry logic, handling redirects manually to preserve auth headers"""
        max_redirects = 5
        current_url = url
        last_error = None

        for attempt in range(retries):
            try:
                for _ in range(max_redirects):
                    response = self.session.get(current_url, params=params, allow_redirects=False, timeout=30)
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
            except requests.exceptions.RequestException as e:
                last_error = e
                self._log(f"Request attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff

        raise Exception(f"Request failed after {retries} attempts: {last_error}")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/api/{endpoint}"
        self._log(f"GET {url} params={params}")
        response = self._make_request(url, params)

        if response.status_code != 200:
            error_msg = f"API error {response.status_code}: {response.text[:500]}"
            self._log(f"ERROR: {error_msg}")
            raise Exception(error_msg)

        return response.json()

    def _get_all_pages(self, endpoint: str, params: dict = None) -> list:
        results = []
        endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/api/{endpoint}"
        page = 1

        while url:
            self._log(f"GET {url} (page {page}) params={params}")
            response = self._make_request(url, params)
            if response.status_code != 200:
                error_msg = f"API error {response.status_code}: {response.text[:500]}"
                self._log(f"ERROR: {error_msg}")
                raise Exception(error_msg)

            data = response.json()
            page_results = data.get("results", [])
            results.extend(page_results)
            self._log(f"Page {page}: got {len(page_results)} results, total: {len(results)}")
            url = data.get("next")
            params = None
            page += 1

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

    def search_documents(self, query: str = None, tag_id: int = None,
                        correspondent_id: int = None, document_type_id: int = None) -> list:
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
        enriched["_fetch_errors"] = []  # Track any issues

        # CRITICAL: Fetch full document to get custom_fields and content
        # The search API does NOT return these fields!
        if fetch_full:
            try:
                self._log(f"Fetching full document {doc['id']} for content...")
                full_doc = self.get_document(doc["id"])

                # Check if content was retrieved
                content = full_doc.get("content", "")
                content_len = len(content) if content else 0
                self._log(f"Document {doc['id']}: content length = {content_len} chars")

                if not content:
                    enriched["_fetch_errors"].append("Content field is empty - OCR may not have processed this document")

                enriched.update(full_doc)
            except Exception as e:
                error_msg = f"Failed to fetch full document {doc['id']}: {str(e)}"
                self._log(f"ERROR: {error_msg}")
                enriched["_fetch_errors"].append(error_msg)

        # Resolve tags
        tag_ids = enriched.get("tags", [])
        enriched["tag_names"] = []
        for tid in tag_ids:
            if tid in tags_lookup:
                enriched["tag_names"].append(tags_lookup[tid]["name"])

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
            enriched["notes"] = [note.get("note", "") for note in notes if note.get("note")]
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
        DEBUG_MODE: bool = Field(
            default=False,
            description="Enable debug mode to see detailed API calls and responses in status updates",
        )
        CONTENT_MAX_LENGTH: int = Field(
            default=0,
            description="Maximum characters of content to return per document (0 = unlimited)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._debug_logs: List[str] = []

    def _collect_debug(self, message: str):
        """Callback to collect debug logs from PaperlessClient"""
        self._debug_logs.append(message)

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
        :param include_content: If True, forces full OCR content regardless of CONTENT_MAX_LENGTH setting.
        :return: JSON with ALL available data for each document including OCR content. The LLM should extract relevant fields based on user request.
        """
        # Reset debug logs for this request
        self._debug_logs = []
        emitter = EventEmitter(__event_emitter__, debug=self.valves.DEBUG_MODE)

        try:
            # Build description of applied filters
            filters_applied = []
            if tag:
                filters_applied.append(f"tag='{tag}'")
            if vendor:
                filters_applied.append(f"vendor='{vendor}'")
            if search_text:
                filters_applied.append(f"search='{search_text}'")

            filter_str = ", ".join(filters_applied) if filters_applied else "NO FILTERS (returning all documents)"
            await emitter.progress_update(f"Searching with: {filter_str}")

            # Initialize client with debug callback
            client = PaperlessClient(
                self.valves.K4MI_URL,
                self.valves.K4MI_TOKEN,
                debug_callback=self._collect_debug if self.valves.DEBUG_MODE else None
            )

            # Resolve tag name to ID
            tag_id = None
            if tag:
                tag_id = client.find_tag_by_name(tag)
                if tag_id is None:
                    available_tags = [t["name"] for t in client.get_tags().values()]
                    error_msg = f"ERROR: Tag '{tag}' not found. Available tags: {', '.join(available_tags[:20])}"
                    await emitter.error_update(error_msg)
                    return json.dumps({"error": error_msg, "available_tags": available_tags})

            # Resolve vendor name to ID
            correspondent_id = None
            if vendor:
                correspondent_id = client.find_correspondent_by_name(vendor)
                if correspondent_id is None:
                    available = [c["name"] for c in client.get_correspondents().values()]
                    error_msg = f"ERROR: Vendor '{vendor}' not found. Available vendors: {', '.join(available[:20])}"
                    await emitter.error_update(error_msg)
                    return json.dumps({"error": error_msg, "available_vendors": available})

            # Search documents with filters
            await emitter.progress_update("Fetching documents from K4mi...")
            raw_documents = client.search_documents(
                query=search_text,
                tag_id=tag_id,
                correspondent_id=correspondent_id
            )

            if not raw_documents:
                msg = f"No documents found matching: {filter_str}"
                await emitter.error_update(msg)
                return json.dumps({"error": msg, "filters_applied": filter_str, "document_count": 0})

            # Process documents - return ALL available data
            await emitter.progress_update(f"Processing {len(raw_documents)} documents...")

            documents = []
            for idx, doc in enumerate(raw_documents, 1):
                enriched = client.enrich_document(doc)

                # Build complete document data - let LLM figure out what to use
                doc_data = {
                    # Document number for easy reference
                    "doc_number": idx,
                    # Basic info
                    "id": enriched.get("id"),
                    "file_name": enriched.get("original_file_name") or enriched.get("archived_file_name"),
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
                    # Apply content length limit only if configured (0 = unlimited)
                    # include_content=True overrides the limit to return full content
                    max_len = self.valves.CONTENT_MAX_LENGTH if not include_content else 0
                    if max_len > 0 and len(content) > max_len:
                        doc_data["content"] = content[:max_len] + f"\n\n[... content truncated at {max_len} chars, full length: {len(content)} ...]"
                    else:
                        doc_data["content"] = content
                    doc_data["content_length"] = len(content)
                else:
                    doc_data["content"] = None
                    doc_data["content_length"] = 0

                # Include any fetch errors for debugging
                fetch_errors = enriched.get("_fetch_errors", [])
                if fetch_errors:
                    doc_data["_warnings"] = fetch_errors

                documents.append(doc_data)

                # Emit citation
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "citation",
                        "data": {
                            "document": [enriched.get("content", "")[:300] if enriched.get("content") else doc_data["title"]],
                            "metadata": [{"source": doc_data["file_name"] or doc_data["title"]}],
                            "source": {"name": doc_data["url"]},
                        },
                    })

            # Analyze content status
            docs_with_content = sum(1 for d in documents if d.get("content"))
            docs_without_content = len(documents) - docs_with_content
            total_content_chars = sum(d.get("content_length", 0) for d in documents)

            status_msg = f"Found {len(documents)} documents ({docs_with_content} with content, {docs_without_content} without)"
            await emitter.success_update(status_msg)

            # Create a summary section to help LLM avoid duplicates
            summary_lines = [f"TOTAL: {len(documents)} UNIQUE documents found"]
            summary_lines.append(f"CONTENT STATUS: {docs_with_content} with OCR content, {docs_without_content} without")
            summary_lines.append(f"TOTAL CONTENT: {total_content_chars:,} characters")
            summary_lines.append("")
            for i, doc in enumerate(documents, 1):
                file_name = doc.get("file_name") or doc.get("title") or f"doc_{doc.get('id')}"
                content_status = f"({doc.get('content_length', 0):,} chars)" if doc.get("content") else "(NO CONTENT)"
                summary_lines.append(f"  [{i}] ID={doc.get('id')}: {file_name} {content_status}")

            # Return ALL data - let the LLM be smart about extracting what user needs
            result = {
                "instructions": """
## IMPORTANT: READ THIS FIRST

You have received UNIQUE documents. Each document has a different ID.
DO NOT create duplicate entries. Each document should appear ONCE in your output.

### Document Content:
- The 'content' field contains the full OCR-extracted text from each document
- If 'content' is null/empty, OCR may not have processed that document yet
- Parse information from the 'content' field - this is the actual document text

### Chinese Invoice (发票) Parsing:
- 销售方/销方 = SELLER/VENDOR (this is who the user means by "vendor")
- 购买方/购方 = BUYER (not the vendor!)
- 金额 = SUBTOTAL (before tax)
- 税额 = TAX AMOUNT
- 价税合计 = TOTAL (MUST = Subtotal + Tax)

### Parse amounts from 'content' field. Remove spaces in numbers.
""",
                "summary": "\n".join(summary_lines),
                "filters_applied": filter_str,
                "document_count": len(documents),
                "documents_with_content": docs_with_content,
                "documents_without_content": docs_without_content,
                "documents": documents,
            }

            # Include debug logs if debug mode is enabled
            if self.valves.DEBUG_MODE and self._debug_logs:
                result["_debug_logs"] = self._debug_logs

            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            result = {"error": error_msg}
            if self.valves.DEBUG_MODE and self._debug_logs:
                result["_debug_logs"] = self._debug_logs
            return json.dumps(result)

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
        self._debug_logs = []
        emitter = EventEmitter(__event_emitter__, debug=self.valves.DEBUG_MODE)

        try:
            await emitter.progress_update(f"Fetching document {document_id}...")
            client = PaperlessClient(
                self.valves.K4MI_URL,
                self.valves.K4MI_TOKEN,
                debug_callback=self._collect_debug if self.valves.DEBUG_MODE else None
            )

            doc = client.get_document(document_id)
            enriched = client.enrich_document(doc)

            # Report content status
            content_len = len(enriched.get("content", "") or "")
            status = f"Retrieved: {enriched.get('title', 'Untitled')} ({content_len:,} chars of content)"
            await emitter.success_update(status)

            result = dict(enriched)
            result["content_length"] = content_len
            if self.valves.DEBUG_MODE and self._debug_logs:
                result["_debug_logs"] = self._debug_logs

            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            result = {"error": error_msg}
            if self.valves.DEBUG_MODE and self._debug_logs:
                result["_debug_logs"] = self._debug_logs
            return json.dumps(result)

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
            return json.dumps({"tags": tag_names, "count": len(tags)}, ensure_ascii=False)

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
            return json.dumps({"vendors": vendor_names, "count": len(correspondents)}, ensure_ascii=False)

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
            fields_info = [{"name": cf["name"], "type": cf.get("data_type")} for cf in custom_fields]

            await emitter.success_update(f"Found {len(custom_fields)} custom fields")
            return json.dumps({"custom_fields": fields_info, "count": len(custom_fields)}, ensure_ascii=False)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            await emitter.error_update(error_msg)
            return json.dumps({"error": error_msg})

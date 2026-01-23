"""
title: K4mi RAG Filter
description: Automatically searches Paperless-ngx documents and injects relevant context into every conversation
author: Claude
version: 1.0.0
license: MIT
"""

import aiohttp
import asyncio
import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Filter:
    """
    A filter that automatically searches Paperless-ngx (K4mi) for relevant documents
    and injects their content into the conversation context.

    This runs BEFORE the LLM sees your message (inlet) and can also process
    the response AFTER the LLM replies (outlet).
    """

    class Valves(BaseModel):
        """Admin-configurable settings"""
        priority: int = Field(
            default=0,
            description="Filter priority (lower = runs first)"
        )
        paperless_url: str = Field(
            default="",
            description="Paperless-ngx base URL (e.g., https://paperless.example.com)"
        )
        paperless_token: str = Field(
            default="",
            description="Paperless-ngx API token"
        )
        enabled: bool = Field(
            default=True,
            description="Enable/disable the RAG filter"
        )
        max_documents: int = Field(
            default=5,
            description="Maximum number of documents to retrieve"
        )
        max_content_length: int = Field(
            default=4000,
            description="Maximum characters of content per document"
        )
        min_query_length: int = Field(
            default=3,
            description="Minimum query length to trigger search"
        )
        search_mode: str = Field(
            default="smart",
            description="Search mode: 'smart' (extract keywords), 'full' (use entire message), 'last_sentence' (use last sentence only)"
        )
        include_tags: str = Field(
            default="",
            description="Only search documents with these tags (comma-separated, empty = all)"
        )
        exclude_system_messages: bool = Field(
            default=True,
            description="Don't modify system messages from context injection"
        )

    class UserValves(BaseModel):
        """Per-user settings (users can override these)"""
        enabled: bool = Field(
            default=True,
            description="Enable/disable RAG for your conversations"
        )
        show_sources: bool = Field(
            default=True,
            description="Show which documents were used in the response"
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _fetch_json(self, session: aiohttp.ClientSession, url: str, params: dict = None) -> dict:
        """Make an authenticated GET request to Paperless API"""
        headers = {
            "Authorization": f"Token {self.valves.paperless_token}",
            "Accept": "application/json"
        }
        async with session.get(url, headers=headers, params=params, timeout=30) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"API error {response.status}: {text[:200]}")
            return await response.json()

    def _extract_search_query(self, message: str) -> str:
        """Extract a search query from the user's message based on search_mode"""
        if not message:
            return ""

        message = message.strip()

        if self.valves.search_mode == "full":
            # Use the entire message
            return message[:500]  # Limit length

        elif self.valves.search_mode == "last_sentence":
            # Use only the last sentence
            sentences = re.split(r'[.!?]+', message)
            sentences = [s.strip() for s in sentences if s.strip()]
            return sentences[-1] if sentences else message

        else:  # "smart" mode - extract key terms
            # Remove common filler words and extract meaningful terms
            stop_words = {
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'the', 'a', 'an',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'can', 'may', 'might', 'must', 'shall', 'to', 'of', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while',
                'about', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                'those', 'am', 'it', 'its', 'itself', 'they', 'them', 'their',
                'find', 'show', 'tell', 'give', 'get', 'know', 'think', 'see',
                'want', 'look', 'use', 'make', 'go', 'need', 'feel', 'try', 'ask',
                'please', 'help', 'thanks', 'thank', 'hello', 'hi', 'hey'
            }

            # Extract words, keeping numbers and preserving case for proper nouns
            words = re.findall(r'\b[\w\d]+\b', message.lower())
            keywords = [w for w in words if w not in stop_words and len(w) > 2]

            # Take first 10 keywords
            return ' '.join(keywords[:10])

    async def _search_paperless(self, query: str) -> List[Dict[str, Any]]:
        """Search Paperless-ngx for documents matching the query"""
        if not query or len(query) < self.valves.min_query_length:
            return []

        base_url = self.valves.paperless_url.rstrip('/')

        async with aiohttp.ClientSession() as session:
            # Build search parameters
            params = {
                "query": query,
                "page_size": self.valves.max_documents
            }

            # Add tag filter if configured
            if self.valves.include_tags:
                # First, get tag IDs by name
                tags_url = f"{base_url}/api/tags/"
                tags_data = await self._fetch_json(session, tags_url)
                tag_names = [t.strip().lower() for t in self.valves.include_tags.split(',')]
                tag_ids = []
                for tag in tags_data.get('results', []):
                    if tag['name'].lower() in tag_names:
                        tag_ids.append(str(tag['id']))
                if tag_ids:
                    params['tags__id__in'] = ','.join(tag_ids)

            # Search documents
            search_url = f"{base_url}/api/documents/"
            data = await self._fetch_json(session, search_url, params)

            results = []
            for doc in data.get('results', [])[:self.valves.max_documents]:
                # Fetch full document to get content
                doc_url = f"{base_url}/api/documents/{doc['id']}/"
                full_doc = await self._fetch_json(session, doc_url)

                content = full_doc.get('content', '') or ''
                if len(content) > self.valves.max_content_length:
                    content = content[:self.valves.max_content_length] + "..."

                results.append({
                    'id': doc['id'],
                    'title': full_doc.get('title', 'Untitled'),
                    'content': content,
                    'created': full_doc.get('created', ''),
                    'correspondent': full_doc.get('correspondent_name') or full_doc.get('correspondent'),
                    'tags': full_doc.get('tags', []),
                    'url': f"{base_url}/documents/{doc['id']}/details",
                    'score': doc.get('__search_hit__', {}).get('score', 0)
                })

            return results

    def _build_context_message(self, documents: List[Dict[str, Any]], query: str) -> str:
        """Build a context message from retrieved documents"""
        if not documents:
            return ""

        context_parts = [
            "## Relevant Documents from K4mi",
            f"*Found {len(documents)} document(s) matching your query*\n"
        ]

        for i, doc in enumerate(documents, 1):
            context_parts.append(f"### Document {i}: {doc['title']}")
            if doc.get('correspondent'):
                context_parts.append(f"**Source/Correspondent:** {doc['correspondent']}")
            if doc.get('created'):
                context_parts.append(f"**Date:** {doc['created'][:10]}")
            context_parts.append(f"\n{doc['content']}\n")
            context_parts.append("---")

        return '\n'.join(context_parts)

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None
    ) -> dict:
        """
        Pre-process the request BEFORE it reaches the LLM.

        This is where we:
        1. Extract the user's query
        2. Search Paperless-ngx for relevant documents
        3. Inject the document content into the conversation context
        """
        # Check if filter is enabled
        if not self.valves.enabled:
            return body

        # Check user-level setting
        if __user__:
            user_valves = __user__.get('valves')
            if user_valves and hasattr(user_valves, 'enabled') and not user_valves.enabled:
                return body

        # Validate configuration
        if not self.valves.paperless_url or not self.valves.paperless_token:
            return body

        messages = body.get('messages', [])
        if not messages:
            return body

        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    last_user_message = content
                elif isinstance(content, list):
                    # Handle multimodal messages
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            last_user_message = part.get('text', '')
                            break
                break

        if not last_user_message:
            return body

        # Extract search query
        search_query = self._extract_search_query(last_user_message)
        if len(search_query) < self.valves.min_query_length:
            return body

        try:
            # Search Paperless-ngx
            documents = await self._search_paperless(search_query)

            if not documents:
                return body

            # Build context message
            context = self._build_context_message(documents, search_query)

            if not context:
                return body

            # Inject context into the conversation
            # Strategy: Add as a system message or prepend to existing system message

            rag_instruction = f"""
Use the following documents from the user's document management system to help answer their question.
If the documents contain relevant information, use it in your response and cite the source.
If the documents don't contain relevant information, you can still answer based on your knowledge.

{context}

Now respond to the user's message:
"""

            # Find or create system message
            if messages and messages[0].get('role') == 'system':
                # Append to existing system message
                if not self.valves.exclude_system_messages:
                    messages[0]['content'] = messages[0]['content'] + "\n\n" + rag_instruction
                else:
                    # Insert as a new message after system
                    messages.insert(1, {
                        'role': 'system',
                        'content': rag_instruction
                    })
            else:
                # Insert new system message at the beginning
                messages.insert(0, {
                    'role': 'system',
                    'content': rag_instruction
                })

            body['messages'] = messages

            # Store document info for outlet (to show sources)
            body['__k4mi_documents__'] = documents

        except Exception as e:
            # Log error but don't break the request
            print(f"K4mi RAG Filter error: {e}")

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None
    ) -> dict:
        """
        Post-process the response AFTER the LLM generates it.

        This is where we can:
        1. Append source citations
        2. Clean up the response
        3. Add metadata
        """
        # Check if user wants to see sources
        if __user__:
            user_valves = __user__.get('valves')
            if user_valves and hasattr(user_valves, 'show_sources') and not user_valves.show_sources:
                return body

        # Get stored documents from inlet
        documents = body.pop('__k4mi_documents__', None)

        if not documents:
            return body

        # Find the assistant's response and append sources
        messages = body.get('messages', [])
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if isinstance(content, str) and content:
                    # Append source citations
                    sources = "\n\n---\n**Sources:**\n"
                    for doc in documents:
                        sources += f"- [{doc['title']}]({doc['url']})\n"
                    msg['content'] = content + sources
                break

        return body

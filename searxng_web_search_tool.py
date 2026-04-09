"""
title: SearXNG Web Search
author: Gnos3-K4mi
version: 1.1.0
license: MIT
description: Fast web search via local SearXNG instance with source citations
"""

import requests
from typing import Callable, Any
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description):
        await self.emit(description)

    async def error_update(self, description):
        await self.emit(description, "error", True)

    async def success_update(self, description):
        await self.emit(description, "success", True)

    async def citation(self, title: str, url: str, content: str):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [content],
                        "metadata": [{"source": url, "name": title}],
                        "source": {"name": title, "url": url},
                    },
                }
            )

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


class Tools:
    class Valves(BaseModel):
        SEARXNG_BASE_URL: str = Field(
            default="http://localhost:8888",
            description="Base URL of your SearXNG instance",
        )
        MAX_RESULTS: int = Field(
            default=10,
            description="Maximum number of search results to return",
        )
        REQUEST_TIMEOUT: int = Field(
            default=15,
            description="HTTP request timeout in seconds",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _format_results(self, results: list) -> str:
        """Format search results as readable markdown text for the LLM."""
        if not results:
            return "No results found."

        lines = [f"Found {len(results)} results:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"## Result {i}: {r['title']}")
            lines.append(f"**URL:** {r['url']}")
            if r.get("published_date"):
                lines.append(f"**Date:** {r['published_date']}")
            lines.append(f"{r['content']}")
            lines.append("")
        lines.append("---")
        lines.append("Use ALL the results above to write a comprehensive answer. Cite sources with their URLs.")
        return "\n".join(lines)

    async def web_search(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search the web using SearXNG. Returns multiple results with titles, URLs, and content snippets.
        IMPORTANT: After receiving results, you MUST use ALL relevant results to compose your answer, not just one. Always cite sources with [Title](URL) format.

        :param query: The search query string
        :return: Formatted search results with titles, URLs, and snippets
        """
        emitter = EventEmitter(__event_emitter__)
        await emitter.progress_update(f"Searching: {query}")

        try:
            response = requests.get(
                f"{self.valves.SEARXNG_BASE_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                },
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            await emitter.error_update(f"Search failed: {e}")
            return f"Search failed: {e}"

        results = []
        for r in data.get("results", [])[: self.valves.MAX_RESULTS]:
            result = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            results.append(result)
            await emitter.citation(
                title=result["title"],
                url=result["url"],
                content=result["content"],
            )

        await emitter.success_update(f"Found {len(results)} results for: {query}")
        return self._format_results(results)

    async def web_search_news(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Search for recent news articles using SearXNG.
        Use this when the user asks about current events, breaking news, or recent developments.
        IMPORTANT: After receiving results, you MUST use ALL relevant results to compose your answer. Always cite sources with [Title](URL) format.

        :param query: The news search query
        :return: Formatted news results with titles, URLs, dates, and snippets
        """
        emitter = EventEmitter(__event_emitter__)
        await emitter.progress_update(f"Searching news: {query}")

        try:
            response = requests.get(
                f"{self.valves.SEARXNG_BASE_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "news",
                    "time_range": "month",
                },
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            await emitter.error_update(f"News search failed: {e}")
            return f"News search failed: {e}"

        results = []
        for r in data.get("results", [])[: self.valves.MAX_RESULTS]:
            result = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "published_date": r.get("publishedDate", ""),
            }
            results.append(result)
            await emitter.citation(
                title=result["title"],
                url=result["url"],
                content=result["content"],
            )

        await emitter.success_update(f"Found {len(results)} news results for: {query}")
        return self._format_results(results)

    async def fetch_page(
        self,
        url: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Fetch a webpage and return its text content. Use this after web_search to read a specific page in detail.

        :param url: The full URL of the page to fetch
        :return: The page text content (truncated to ~8000 chars)
        """
        emitter = EventEmitter(__event_emitter__)
        await emitter.progress_update(f"Fetching: {url}")

        try:
            response = requests.get(
                url,
                timeout=self.valves.REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                },
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "html" in content_type:
                try:
                    from html2text import HTML2Text

                    h = HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = True
                    h.body_width = 0
                    text = h.handle(response.text)
                except ImportError:
                    import re

                    text = re.sub(r"<script[^>]*>.*?</script>", "", response.text, flags=re.DOTALL)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()
            else:
                text = response.text

            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[... truncated]"

            await emitter.citation(title=url, url=url, content=text[:500])
            await emitter.success_update(f"Fetched {len(text)} chars from {url}")
            return f"# Content from {url}\n\n{text}"

        except requests.RequestException as e:
            await emitter.error_update(f"Fetch failed: {e}")
            return f"Failed to fetch {url}: {e}"

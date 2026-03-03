"""
title: K4mi Invoice Query Tool
author: gnos3
version: 0.1.0
description: Query structured invoice data extracted from K4mi documents. Use this tool for any questions about invoices, payments, vendors, or spending.
license: MIT
"""

import json
import logging
from typing import Optional

import aiohttp
from fastapi import Request
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Tools:
    class Valves(BaseModel):
        """Admin-configurable settings for the invoice tool."""

        invoice_api_base_url: str = Field(
            default="http://localhost:8001",
            description="Base URL of the Invoice Processor API",
        )
        invoice_api_token: str = Field(
            default="",
            description="Authentication token for the Invoice Processor API (optional)",
        )
        k4mi_base_url: str = Field(
            default="http://localhost:8000",
            description="Base URL of K4mi (Paperless-NGX) for document links",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    async def _api_get(self, path: str, params: dict = None) -> dict | list:
        """Internal helper to call the invoice API."""
        url = f"{self.valves.invoice_api_base_url}{path}"
        headers = {"Accept": "application/json"}
        if self.valves.invoice_api_token:
            headers["Authorization"] = f"Bearer {self.valves.invoice_api_token}"

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"error": f"API returned status {resp.status}: {text[:200]}"}
                return await resp.json()

    def _add_k4mi_links(self, data):
        """Add K4mi document links to invoice results."""
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("k4mi_document_id"):
                item["k4mi_url"] = (
                    f"{self.valves.k4mi_base_url}/documents/{item['k4mi_document_id']}/details"
                )
        return data

    async def search_invoices(
        self,
        vendor: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tag: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        notes_search: Optional[str] = None,
        limit: int = 20,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        Search invoices with filters. Use this for questions like "show me invoices from vendor X",
        "what invoices do we have for January 2025", or "invoices tagged project1".

        :param vendor: Filter by vendor/company name (partial match)
        :param date_from: Start date filter in YYYY-MM-DD format
        :param date_to: End date filter in YYYY-MM-DD format
        :param tag: Filter by K4mi tag name
        :param min_amount: Minimum total amount filter
        :param max_amount: Maximum total amount filter
        :param notes_search: Search text within invoice notes
        :param limit: Maximum number of results (default 20)
        :return: JSON array of matching invoices with vendor, date, amount, invoice number
        """
        params = {
            k: v
            for k, v in {
                "vendor": vendor,
                "date_from": date_from,
                "date_to": date_to,
                "tag": tag,
                "min_amount": min_amount,
                "max_amount": max_amount,
                "notes_search": notes_search,
                "limit": limit,
            }.items()
            if v is not None
        }

        result = await self._api_get("/api/invoices", params)
        result = self._add_k4mi_links(result)
        return json.dumps(result, ensure_ascii=False, default=str)

    async def get_vendor_total(
        self,
        vendor: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        Get the exact total amount paid to a specific vendor, optionally within a date range.
        Use this for questions like "how much did we pay vendor X in January 2025" or
        "total spending with CompanyName this year". Returns precise SQL-calculated sums.

        :param vendor: Vendor/company name to calculate total for
        :param date_from: Start date in YYYY-MM-DD format (optional)
        :param date_to: End date in YYYY-MM-DD format (optional)
        :return: JSON with vendor name, total amount, currency, and invoice count
        """
        params = {"vendor": vendor}
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        result = await self._api_get("/api/invoices/vendor-total", params)
        return json.dumps(result, ensure_ascii=False, default=str)

    async def get_invoices_by_tag(
        self,
        tag: str,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        Get all invoices associated with a specific K4mi tag.
        Use this for questions like "show all invoices for project X" or "invoices tagged Q1-2025".

        :param tag: The K4mi tag name to filter by
        :return: JSON array of invoices with full details for that tag
        """
        result = await self._api_get("/api/invoices/by-tag", {"tag": tag})
        result = self._add_k4mi_links(result)
        return json.dumps(result, ensure_ascii=False, default=str)

    async def get_invoice_details(
        self,
        invoice_id: int,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        Get full details of a specific invoice including line items and notes.
        Use this when the user asks about a specific invoice by ID or wants a detailed breakdown.

        :param invoice_id: The invoice ID from the database
        :return: JSON with complete invoice details including line items and notes
        """
        result = await self._api_get(f"/api/invoices/{invoice_id}")
        result = self._add_k4mi_links(result)
        return json.dumps(result, ensure_ascii=False, default=str)

    async def get_spending_summary(
        self,
        period: str = "monthly",
        year: Optional[int] = None,
        vendor: Optional[str] = None,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        Get aggregated spending summary. Use this for questions like "show me monthly spending
        for 2025", "spending breakdown by vendor this year", or "quarterly totals".

        :param period: Grouping period - "monthly", "quarterly", "yearly", or "by_vendor"
        :param year: Filter to a specific year (optional)
        :param vendor: Filter to a specific vendor (optional)
        :return: JSON with aggregated spending data grouped by the specified period
        """
        params = {"period": period}
        if year:
            params["year"] = year
        if vendor:
            params["vendor"] = vendor

        result = await self._api_get("/api/invoices/spending-summary", params)
        return json.dumps(result, ensure_ascii=False, default=str)

    async def list_vendors(
        self,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        List all known vendors/companies with their invoice counts and total amounts.
        Use this when the user asks "who are our vendors" or needs vendor name suggestions.

        :return: JSON array of vendors with name, invoice_count, and total_amount
        """
        result = await self._api_get("/api/invoices/vendors")
        return json.dumps(result, ensure_ascii=False, default=str)

    async def search_invoice_notes(
        self,
        query: str,
        __request__: Request = None,
        __user__: dict = None,
    ) -> str:
        """
        Search across all invoice notes for specific text.
        Use this when the user asks about notes or comments on invoices.

        :param query: Search text to find in invoice notes
        :return: JSON array of invoices whose notes match the search query
        """
        result = await self._api_get("/api/invoices/search-notes", {"query": query})
        result = self._add_k4mi_links(result)
        return json.dumps(result, ensure_ascii=False, default=str)

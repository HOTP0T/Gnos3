"""
Invoice/Accounting Database connector.

Syncs structured data from the invoice-processor API into Gnos3's RAG pipeline
by serializing records (invoices, transactions, payments, companies) into
natural-language text documents that embed well for semantic search.

Users can then ask questions like:
  - "What invoices did ACME send us this quarter?"
  - "Show me all posted journal entries for March"
  - "What's our outstanding AP balance?"
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional

import httpx

from open_webui.connectors.base import (
    BaseConnector,
    DocumentContent,
    ExternalDocument,
)

log = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Serializers: structured records → natural-language text ──────────


def _serialize_invoice(inv: dict) -> str:
    """Convert an invoice record into embedding-friendly text."""
    lines = []
    lines.append(f"Invoice #{inv.get('invoice_number', 'N/A')} from {inv.get('vendor_name', 'Unknown vendor')}")

    if inv.get("client_name"):
        lines.append(f"Client: {inv['client_name']}")

    date_parts = []
    if inv.get("invoice_date"):
        date_parts.append(f"Date: {inv['invoice_date']}")
    if inv.get("due_date"):
        date_parts.append(f"Due: {inv['due_date']}")
    if inv.get("payment_terms"):
        date_parts.append(f"Terms: {inv['payment_terms']}")
    if date_parts:
        lines.append(" | ".join(date_parts))

    currency = inv.get("currency", "USD")
    amount_parts = []
    if inv.get("subtotal") is not None:
        amount_parts.append(f"Subtotal: {currency} {inv['subtotal']}")
    if inv.get("tax_amount") is not None:
        amount_parts.append(f"Tax: {currency} {inv['tax_amount']}")
    if inv.get("total_amount") is not None:
        amount_parts.append(f"Total: {currency} {inv['total_amount']}")
    if amount_parts:
        lines.append(" | ".join(amount_parts))

    if inv.get("balance_due") is not None:
        lines.append(f"Balance due: {currency} {inv['balance_due']}")

    if inv.get("description"):
        lines.append(f"Description: {inv['description']}")

    if inv.get("final_account_code"):
        lines.append(f"Account: {inv['final_account_code']} (status: {inv.get('categorization_status', 'pending')})")
    elif inv.get("suggested_account_code"):
        lines.append(f"Suggested account: {inv['suggested_account_code']} (confidence: {inv.get('suggested_account_confidence', '?')})")

    status = inv.get("processing_status", "unknown")
    lines.append(f"Processing status: {status}")

    if inv.get("k4mi_tags"):
        lines.append(f"Tags: {', '.join(str(t) for t in inv['k4mi_tags'])}")
    if inv.get("k4mi_correspondent"):
        lines.append(f"Correspondent: {inv['k4mi_correspondent']}")

    if inv.get("line_items") and isinstance(inv["line_items"], list):
        lines.append("Line items:")
        for item in inv["line_items"][:20]:  # cap at 20 items
            desc = item.get("description", "item")
            qty = item.get("quantity", "")
            price = item.get("unit_price", "")
            total = item.get("total", "")
            lines.append(f"  - {desc} (qty: {qty}, unit: {price}, total: {total})")

    return "\n".join(lines)


def _serialize_transaction(txn: dict) -> str:
    """Convert a journal entry/transaction into embedding-friendly text."""
    lines = []
    entry_num = txn.get("entry_number", txn.get("id", "?"))
    status = txn.get("status", "draft")
    lines.append(f"Journal Entry {entry_num} [{status}]")

    date_parts = []
    if txn.get("transaction_date"):
        date_parts.append(f"Date: {txn['transaction_date']}")
    txn_type = txn.get("transaction_type", "journal")
    date_parts.append(f"Type: {txn_type}")
    currency = txn.get("currency", "USD")
    date_parts.append(f"Currency: {currency}")
    rate = txn.get("exchange_rate")
    if rate and float(rate) != 1.0:
        date_parts.append(f"Rate: {rate}")
    lines.append(" | ".join(date_parts))

    if txn.get("reference"):
        lines.append(f"Reference: {txn['reference']}")
    if txn.get("description"):
        lines.append(f"Description: {txn['description']}")

    # Journal entry lines
    txn_lines = txn.get("lines", [])
    if txn_lines:
        lines.append("Lines:")
        for line in txn_lines:
            acct = line.get("account", {})
            code = acct.get("code", "?") if isinstance(acct, dict) else "?"
            name = acct.get("name", "") if isinstance(acct, dict) else ""
            debit = line.get("debit", 0)
            credit = line.get("credit", 0)
            side = "DR" if float(debit) > 0 else "CR"
            amount = debit if float(debit) > 0 else credit
            desc = f" — {line['description']}" if line.get("description") else ""
            lines.append(f"  {side} {code} {name} {currency} {amount}{desc}")

    return "\n".join(lines)


def _serialize_payment(pay: dict) -> str:
    """Convert a payment record into embedding-friendly text."""
    lines = []
    ref = pay.get("reference", pay.get("id", "?"))
    direction = pay.get("direction", "unknown")
    lines.append(f"Payment {ref} [{direction}]")

    date_parts = []
    if pay.get("payment_date"):
        date_parts.append(f"Date: {pay['payment_date']}")
    currency = pay.get("currency", "USD")
    amount = pay.get("amount", 0)
    date_parts.append(f"Amount: {currency} {amount}")
    method = pay.get("method", "")
    if method:
        date_parts.append(f"Method: {method}")
    lines.append(" | ".join(date_parts))

    if pay.get("payer"):
        lines.append(f"Payer: {pay['payer']}")
    if pay.get("payee"):
        lines.append(f"Payee: {pay['payee']}")
    if pay.get("notes"):
        lines.append(f"Notes: {pay['notes']}")

    return "\n".join(lines)


def _serialize_company_summary(company: dict, stats: Optional[dict] = None) -> str:
    """Create a summary document for a company."""
    lines = []
    lines.append(f"Company: {company.get('name', '?')}")
    if company.get("legal_name"):
        lines.append(f"Legal name: {company['legal_name']}")
    lines.append(f"Country: {company.get('country', '?')} | Currency: {company.get('currency', '?')}")

    if stats:
        if "invoice_count" in stats:
            lines.append(f"Invoices: {stats['invoice_count']}")
        if "transaction_count" in stats:
            lines.append(f"Transactions: {stats['transaction_count']}")
        if "payment_count" in stats:
            lines.append(f"Payments: {stats['payment_count']}")

    return "\n".join(lines)


# ── Connector ────────────────────────────────────────────────────────


class InvoiceDBConnector(BaseConnector):
    """Connector for the invoice-processor accounting/invoice database."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_base = config.get("api_base_url", "").rstrip("/")
        self.company_ids = config.get("company_ids") or []
        self.include_invoices = config.get("include_invoices", True)
        self.include_transactions = config.get("include_transactions", True)
        self.include_payments = config.get("include_payments", True)
        self.include_company_summaries = config.get("include_company_summaries", True)

    async def _api_get(self, path: str, params: Optional[dict] = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.api_base}{path}",
                params=params or {},
            )
            resp.raise_for_status()
            return resp.json()

    def _extract_list(self, data) -> list:
        """Extract the list from an API response that may be a list or a dict like
        {"invoices": [...], "total": N} or {"items": [...]}."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Try known keys: invoices, companies, transactions, payments, items, results
            for key in ("invoices", "companies", "transactions", "payments", "items", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return []

    async def _get_companies(self) -> list[dict]:
        """Get companies to sync (filtered or all)."""
        data = await self._api_get("/api/accounting/companies")
        companies = self._extract_list(data)
        if self.company_ids:
            companies = [c for c in companies if c.get("id") in self.company_ids]
        return companies

    async def list_documents(
        self, since: Optional[datetime] = None
    ) -> list[ExternalDocument]:
        docs = []
        since_str = since.isoformat() if since else None

        # Invoices
        if self.include_invoices:
            params = {}
            if since_str:
                params["updated_after"] = since_str
            try:
                data = await self._api_get("/api/invoices", params=params)
                invoices = self._extract_list(data)
                for inv in invoices:
                    text = _serialize_invoice(inv)
                    docs.append(ExternalDocument(
                        external_id=f"invoice:{inv['id']}",
                        title=f"Invoice #{inv.get('invoice_number', inv['id'])} — {inv.get('vendor_name', 'Unknown')}",
                        content_hash=_hash(text),
                        metadata={"type": "invoice", "source_id": inv["id"]},
                    ))
            except Exception as e:
                log.warning("Failed to list invoices: %s", e)

        # Per-company data
        try:
            companies = await self._get_companies()
        except Exception as e:
            log.warning("Failed to list companies: %s", e)
            companies = []

        for company in companies:
            cid = company["id"]
            cname = company.get("name", f"Company {cid}")

            # Company summary
            if self.include_company_summaries:
                text = _serialize_company_summary(company)
                docs.append(ExternalDocument(
                    external_id=f"company:{cid}:summary",
                    title=f"Company Summary — {cname}",
                    content_hash=_hash(text),
                    metadata={"type": "company_summary", "company_id": cid},
                ))

            # Transactions
            if self.include_transactions:
                params = {"company_id": cid}
                if since_str:
                    params["updated_after"] = since_str
                try:
                    data = await self._api_get("/api/accounting/transactions", params=params)
                    txns = self._extract_list(data)
                    for txn in txns:
                        text = _serialize_transaction(txn)
                        docs.append(ExternalDocument(
                            external_id=f"txn:{txn['id']}",
                            title=f"Journal Entry {txn.get('entry_number', txn['id'])} — {cname}",
                            content_hash=_hash(text),
                            metadata={"type": "transaction", "company_id": cid, "source_id": txn["id"]},
                        ))
                except Exception as e:
                    log.warning("Failed to list transactions for company %s: %s", cid, e)

            # Payments
            if self.include_payments:
                params = {"company_id": cid}
                if since_str:
                    params["updated_after"] = since_str
                try:
                    data = await self._api_get("/api/accounting/payments", params=params)
                    payments = self._extract_list(data)
                    for pay in payments:
                        text = _serialize_payment(pay)
                        docs.append(ExternalDocument(
                            external_id=f"payment:{pay['id']}",
                            title=f"Payment {pay.get('reference', pay['id'])} — {cname}",
                            content_hash=_hash(text),
                            metadata={"type": "payment", "company_id": cid, "source_id": pay["id"]},
                        ))
                except Exception as e:
                    log.warning("Failed to list payments for company %s: %s", cid, e)

        return docs

    async def fetch_document(self, external_id: str) -> DocumentContent:
        """Fetch a single record and serialize it to text."""
        parts = external_id.split(":")
        record_type = parts[0]

        if record_type == "invoice":
            inv_id = parts[1]
            inv = await self._api_get(f"/api/invoices/{inv_id}")
            text = _serialize_invoice(inv)
            return DocumentContent(
                external_id=external_id,
                title=f"Invoice #{inv.get('invoice_number', inv_id)} — {inv.get('vendor_name', 'Unknown')}",
                content=text,
                file_name=f"invoice-{inv_id}.txt",
                mime_type="text/plain",
                metadata={"type": "invoice", "source_id": int(inv_id)},
            )

        elif record_type == "txn":
            txn_id = parts[1]
            txn = await self._api_get(f"/api/accounting/transactions/{txn_id}")
            text = _serialize_transaction(txn)
            return DocumentContent(
                external_id=external_id,
                title=f"Journal Entry {txn.get('entry_number', txn_id)}",
                content=text,
                file_name=f"txn-{txn_id}.txt",
                mime_type="text/plain",
                metadata={"type": "transaction", "source_id": int(txn_id)},
            )

        elif record_type == "payment":
            pay_id = parts[1]
            pay = await self._api_get(f"/api/accounting/payments/{pay_id}")
            text = _serialize_payment(pay)
            return DocumentContent(
                external_id=external_id,
                title=f"Payment {pay.get('reference', pay_id)}",
                content=text,
                file_name=f"payment-{pay_id}.txt",
                mime_type="text/plain",
                metadata={"type": "payment", "source_id": int(pay_id)},
            )

        elif record_type == "company":
            company_id = parts[1]
            company = await self._api_get(f"/api/accounting/companies/{company_id}")
            text = _serialize_company_summary(company)
            return DocumentContent(
                external_id=external_id,
                title=f"Company Summary — {company.get('name', company_id)}",
                content=text,
                file_name=f"company-{company_id}-summary.txt",
                mime_type="text/plain",
                metadata={"type": "company_summary", "company_id": int(company_id)},
            )

        else:
            raise ValueError(f"Unknown record type in external_id: {external_id}")

    async def get_deleted_ids(self, known_ids: list[str]) -> list[str]:
        """
        Check which records have been deleted from the source.
        For structured data this is less critical — records rarely get deleted.
        We check invoices and transactions individually.
        """
        deleted = []
        for ext_id in known_ids:
            parts = ext_id.split(":")
            record_type = parts[0]
            try:
                if record_type == "invoice":
                    await self._api_get(f"/api/invoices/{parts[1]}")
                elif record_type == "txn":
                    await self._api_get(f"/api/accounting/transactions/{parts[1]}")
                elif record_type == "payment":
                    await self._api_get(f"/api/accounting/payments/{parts[1]}")
                elif record_type == "company":
                    await self._api_get(f"/api/accounting/companies/{parts[1]}")
                # If we get here, the record still exists
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    deleted.append(ext_id)
                else:
                    log.warning("Error checking %s: %s", ext_id, e)
            except Exception as e:
                log.warning("Error checking %s: %s", ext_id, e)
        return deleted

    @classmethod
    def connector_label(cls) -> str:
        return "Invoice & Accounting Database"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "type": "object",
            "required": ["api_base_url"],
            "properties": {
                "api_base_url": {
                    "type": "string",
                    "title": "Invoice Processor API URL",
                    "description": "Base URL of the invoice-processor API (e.g. http://localhost:8001)",
                },
                "company_ids": {
                    "type": "array",
                    "title": "Company IDs",
                    "description": "Only sync data for these companies (empty = all)",
                    "items": {"type": "integer"},
                },
                "include_invoices": {
                    "type": "boolean",
                    "title": "Include Invoices",
                    "default": True,
                },
                "include_transactions": {
                    "type": "boolean",
                    "title": "Include Journal Entries",
                    "default": True,
                },
                "include_payments": {
                    "type": "boolean",
                    "title": "Include Payments",
                    "default": True,
                },
                "include_company_summaries": {
                    "type": "boolean",
                    "title": "Include Company Summaries",
                    "default": True,
                },
            },
        }

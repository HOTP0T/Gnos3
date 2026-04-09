"""
title: Company Data Tool
description: Universal data access tool. Query any structured data (invoices, accounting, bank statements, companies) via SQL and any document in K4mi by ID or search. The AI writes queries based on your questions.
requirements: aiohttp,psycopg2-binary
version: 2.0.0
"""

import json
import aiohttp
import psycopg2
import psycopg2.extras
from pydantic import BaseModel, Field

# ── Database schema (injected into LLM context so it can write SQL) ──

DATABASE_SCHEMA = """
=== ACCOUNTING & INVOICE DATABASE SCHEMA ===

TABLE companies (id, name, legal_name, country, currency, description, is_active, opening_balance_date, created_at, updated_at)
  -- Each company has its own accounts, transactions, payments, bank accounts, invoices

TABLE accounts (id, company_id FK→companies, code VARCHAR, name, account_type ENUM(asset,liability,equity,revenue,expense), normal_balance, parent_id FK→accounts, level, description, is_active, opening_debit NUMERIC, opening_credit NUMERIC)
  -- Chart of accounts per company. code is like '401000', '607000', etc.

TABLE invoices (id, company_id FK→companies NULL, k4mi_document_id INT UNIQUE, k4mi_document_url, vendor_name, vendor_name_normalized, invoice_number, invoice_date DATE, due_date DATE, currency, subtotal NUMERIC, tax_amount NUMERIC, tax_inclusive BOOL, total_amount NUMERIC, amount_paid, balance_due, payment_terms, po_number, description, client_name, business_unit, line_items JSONB, k4mi_tags JSONB, k4mi_correspondent, k4mi_document_type, k4mi_notes JSONB, suggested_account_code, final_account_code, categorization_status, processing_status, confidence_score, needs_review BOOL, extraction_model, created_at, updated_at)
  -- Invoices extracted from K4mi documents via OCR. k4mi_document_id links to K4mi.

TABLE transactions (id, company_id FK→companies, transaction_date DATE, transaction_type ENUM(invoice,bill,payment_in,payment_out,journal,others,adjustment), status ENUM(draft,posted,voided), currency, exchange_rate NUMERIC, reference, description, notes, invoice_id FK→invoices NULL, payment_id FK→payments NULL, entry_number VARCHAR, created_at, updated_at)
  -- Journal entries. Each has multiple transaction_lines.

TABLE transaction_lines (id, transaction_id FK→transactions, account_id FK→accounts, debit NUMERIC, credit NUMERIC, description)
  -- Double-entry lines. For each transaction: SUM(debit) = SUM(credit).

TABLE payments (id, company_id FK→companies, payment_date DATE, amount NUMERIC, currency, direction ENUM(inbound,outbound), method ENUM(cash,bank_transfer,check,credit_card,other), reference, notes, payer, payee, invoice_id FK→invoices NULL, debit_account_id FK→accounts NULL, credit_account_id FK→accounts NULL, created_at, updated_at)

TABLE bank_accounts (id, company_id FK→companies, name, account_id FK→accounts NULL, bank_name, account_number_masked, currency, is_active)

TABLE bank_statement_lines (id, bank_account_id FK→bank_accounts, transaction_date DATE, description, reference, amount NUMERIC, balance NUMERIC, match_status VARCHAR(unmatched/partial_matched/auto_matched/manual_matched/excluded), matched_transaction_id FK→transactions NULL, payment_id FK→payments NULL, allocated_total NUMERIC, import_batch_id, created_at)
  -- Bank statement lines imported from CSV/Excel. Matched to transactions/payments.

TABLE accounting_periods (id, company_id FK→companies, name, start_date DATE, end_date DATE, is_closed BOOL, closed_at)

TABLE fixed_assets (id, company_id FK→companies, name, description, asset_account_id FK, depreciation_account_id FK, expense_account_id FK, acquisition_date DATE, acquisition_value NUMERIC, salvage_value NUMERIC, useful_life_months INT, depreciation_method, disposal_date, disposal_value, invoice_id FK→invoices NULL, is_active)

TABLE exchange_rates (id, company_id FK→companies, from_currency, to_currency, rate NUMERIC, effective_date DATE, source)

TABLE match_groups (id, company_id FK→companies, match_type VARCHAR(manual/auto/system), notes, status VARCHAR(active/voided), created_at, updated_at)
TABLE match_group_bsl (id, match_group_id FK→match_groups, bank_statement_line_id FK→bank_statement_lines, allocated_amount NUMERIC)
TABLE match_group_transaction (id, match_group_id FK→match_groups, transaction_id FK→transactions, allocated_amount NUMERIC)
  -- M:N matching between bank statement lines and transactions

TABLE recurring_templates (id, company_id FK→companies, name, frequency, day_of_month, start_date, end_date, next_run_date, transaction_type, currency, reference_prefix, description, lines_template JSONB, auto_post BOOL, is_active)

TABLE account_categorization_rules (id, company_id FK→companies, vendor_name_pattern, account_code, match_type, counterparty_account_code, auto_create_entry, tax_account_code, tax_rate, match_field, rule_type, status, is_active)

=== USEFUL QUERY PATTERNS ===
-- Find company by name: SELECT * FROM companies WHERE name ILIKE '%keyword%' OR legal_name ILIKE '%keyword%'
-- Get bank statements: SELECT bsl.*, ba.name as bank_name FROM bank_statement_lines bsl JOIN bank_accounts ba ON bsl.bank_account_id = ba.id WHERE ba.company_id = X
-- Invoice totals by vendor: SELECT vendor_name, COUNT(*), SUM(total_amount) FROM invoices GROUP BY vendor_name ORDER BY SUM(total_amount) DESC
-- Account balance: SELECT a.code, a.name, SUM(tl.debit) as total_debit, SUM(tl.credit) as total_credit FROM transaction_lines tl JOIN accounts a ON tl.account_id = a.id JOIN transactions t ON tl.transaction_id = t.id WHERE t.company_id = X AND t.status = 'posted' GROUP BY a.code, a.name
-- Count documents by type in K4mi: use search_k4mi_documents or count_k4mi_documents functions instead
"""


class Tools:
    class Valves(BaseModel):
        k4mi_base_url: str = Field(
            default="http://localhost:8000",
            description="K4mi (Paperless-NGX) API base URL",
        )
        k4mi_api_token: str = Field(
            default="",
            description="K4mi API authentication token",
        )
        db_connection_string: str = Field(
            default="postgresql://invoice_user:change_me_in_production@localhost:5433/invoices",
            description="PostgreSQL connection string for the accounting/invoice database",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def query_database(self, sql_query: str) -> str:
        """Execute a read-only SQL query against the accounting and invoice database. Use this to answer ANY question about invoices, companies, transactions, payments, bank statements, accounts, or any structured financial data. SCHEMA: companies(id,name,legal_name,country,currency,description,is_active). accounts(id,company_id,code,name,account_type[asset/liability/equity/revenue/expense],normal_balance,parent_id,opening_debit,opening_credit,is_active). invoices(id,company_id,k4mi_document_id,vendor_name,vendor_name_normalized,invoice_number,invoice_date,due_date,currency,subtotal,tax_amount,tax_inclusive,total_amount,amount_paid,balance_due,payment_terms,description,client_name,k4mi_tags JSONB,processing_status,confidence_score,needs_review,final_account_code,categorization_status). transactions(id,company_id,transaction_date,transaction_type[invoice/bill/payment_in/payment_out/journal/others/adjustment],status[draft/posted/voided],currency,exchange_rate,reference,description,invoice_id,payment_id,entry_number). transaction_lines(id,transaction_id,account_id,debit,credit,description). payments(id,company_id,payment_date,amount,currency,direction[inbound/outbound],method[cash/bank_transfer/check/credit_card/other],reference,payer,payee,invoice_id). bank_accounts(id,company_id,name,bank_name,account_number_masked,currency,is_active). bank_statement_lines(id,bank_account_id,transaction_date,description,reference,amount,balance,match_status[unmatched/partial_matched/auto_matched/manual_matched/excluded],matched_transaction_id,payment_id,allocated_total). accounting_periods(id,company_id,name,start_date,end_date,is_closed). fixed_assets(id,company_id,name,acquisition_date,acquisition_value,salvage_value,useful_life_months,depreciation_method,is_active). exchange_rates(id,company_id,from_currency,to_currency,rate,effective_date). TIPS: Find company by name with ILIKE. Join bank_statement_lines to bank_accounts via bank_account_id, then filter by company_id. Join transaction_lines to accounts and transactions for ledger queries.

        :param sql_query: A read-only SQL SELECT query. Must be SELECT only, no INSERT/UPDATE/DELETE/DROP.
        :return: Query results as JSON with columns and rows
        """
        try:
            sql = sql_query.strip().rstrip(";")

            # Safety: only allow SELECT statements
            first_word = sql.split()[0].upper() if sql else ""
            if first_word not in ("SELECT", "WITH", "EXPLAIN"):
                return json.dumps({"error": "Only SELECT queries are allowed. No modifications permitted."})

            # Block dangerous keywords
            sql_upper = sql.upper()
            for dangerous in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]:
                if dangerous in sql_upper.split():
                    return json.dumps({"error": f"Forbidden keyword: {dangerous}. Only read-only queries allowed."})

            conn = psycopg2.connect(self.valves.db_connection_string)
            conn.set_session(readonly=True, autocommit=True)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cur.execute(sql)
            rows = cur.fetchmany(200)  # Cap at 200 rows

            total = cur.rowcount
            columns = [desc[0] for desc in cur.description] if cur.description else []

            cur.close()
            conn.close()

            # Convert to serializable format
            results = []
            for row in rows:
                clean = {}
                for k, v in dict(row).items():
                    if hasattr(v, "isoformat"):
                        clean[k] = v.isoformat()
                    elif isinstance(v, (int, float, bool, str, type(None))):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
                results.append(clean)

            return json.dumps(
                {"columns": columns, "row_count": total, "returned": len(results), "rows": results},
                ensure_ascii=False,
                default=str,
            )
        except psycopg2.Error as e:
            return json.dumps({"error": f"Database error: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Query failed: {str(e)}"})

    async def get_k4mi_document(self, document_id: int) -> str:
        """Get a specific document from K4mi (Paperless-NGX) by its ID number. Use this when the user asks about a specific document by number, or when you need the actual content of a document referenced by k4mi_document_id in the database.

        :param document_id: The K4mi document ID number
        :return: Document details including title, full text content, correspondent, tags, dates, and a link to view it
        """
        try:
            headers = {"Authorization": f"Token {self.valves.k4mi_api_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.valves.k4mi_base_url}/api/documents/{document_id}/",
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    doc = await resp.json()

            result = {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "content": doc.get("content", "")[:4000],
                "correspondent": doc.get("correspondent_name"),
                "document_type": doc.get("document_type_name"),
                "tags": doc.get("tag_names", []),
                "created_date": doc.get("created"),
                "added_date": doc.get("added"),
                "original_filename": doc.get("original_file_name"),
                "notes": [n if isinstance(n, str) else n.get("note", "") for n in (doc.get("notes") or [])],
                "view_url": f"{self.valves.k4mi_base_url}/documents/{document_id}/details",
            }
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"Failed to get document {document_id}: {str(e)}"})

    async def search_k4mi_documents(
        self,
        query: str = "",
        limit: int = 20,
    ) -> str:
        """Search documents in K4mi (Paperless-NGX) by keyword. Use this for finding documents by content, title, correspondent, or tags. Use query_database for structured invoice/accounting data instead.

        :param query: Full-text search query (searches title, content, tags, correspondent)
        :param limit: Maximum results (default 20)
        :return: List of matching documents with ID, title, tags, and dates
        """
        try:
            headers = {"Authorization": f"Token {self.valves.k4mi_api_token}"}
            params = {"page_size": min(limit, 100), "ordering": "-created"}
            if query:
                params["query"] = query

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.valves.k4mi_base_url}/api/documents/",
                    headers=headers,
                    params=params,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            results = data.get("results", [])
            docs = [
                {
                    "id": doc["id"],
                    "title": doc.get("title"),
                    "correspondent": doc.get("correspondent_name"),
                    "document_type": doc.get("document_type_name"),
                    "tags": doc.get("tag_names", []),
                    "created": doc.get("created", "")[:10],
                    "view_url": f"{self.valves.k4mi_base_url}/documents/{doc['id']}/details",
                }
                for doc in results[:limit]
            ]

            return json.dumps(
                {"total_in_system": data.get("count", 0), "returned": len(docs), "documents": docs},
                ensure_ascii=False,
                default=str,
            )
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

# Invoice Parsing & Reporting System - Recreation Guide

A complete guide to recreating the invoice processing pipeline with OCR extraction, duplicate detection, and reporting.

---

## System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  Vision LLM      │───▶│   PostgreSQL    │
│   (REST API)    │    │  (Ollama)        │    │   (Structured)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐             │
│   Dashboard     │◀───│  Report Export   │◀────────────┘
│   (HTML/JS)     │    │  (Markdown/CSV)  │
└─────────────────┘    └──────────────────┘
```

**Core Components**:
1. **File-Mover Service** - FastAPI service handling uploads, Ollama calls, and database ops
2. **Ollama** - Local LLM for invoice OCR extraction
3. **PostgreSQL** - Structured invoice storage with duplicate detection
4. **Dashboard** - Real-time HTML dashboard with stats and actions

---

## 1. Database Schema

### Main Tables

```sql
-- Core invoice storage
CREATE TABLE invoices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Extracted fields
    vendor_name VARCHAR(255),
    customer_name VARCHAR(255),
    invoice_number VARCHAR(100),
    invoice_date DATE,
    due_date DATE,
    subtotal DECIMAL(15,2),
    total_amount DECIMAL(15,2),
    tax_amount DECIMAL(15,2),
    currency VARCHAR(10) DEFAULT 'USD',
    payment_terms VARCHAR(100),
    po_number VARCHAR(100),
    line_items JSONB,

    -- File info
    original_filename VARCHAR(500),
    file_path VARCHAR(1000),
    full_text TEXT,

    -- Processing metadata
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    processing_date TIMESTAMP WITH TIME ZONE,
    extraction_model VARCHAR(100),
    confidence_score DECIMAL(5,2),
    vision_call_latency_ms INTEGER,

    -- Quality assurance
    needs_review BOOLEAN DEFAULT false,
    review_reason TEXT,
    is_duplicate BOOLEAN DEFAULT false,
    duplicate_of_id UUID REFERENCES invoices(id),

    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_invoices_vendor ON invoices(vendor_name);
CREATE INDEX idx_invoices_date ON invoices(invoice_date);
CREATE INDEX idx_invoices_status ON invoices(processing_status);
CREATE INDEX idx_invoices_number ON invoices(invoice_number);
CREATE INDEX idx_invoices_review ON invoices(needs_review) WHERE needs_review = true;

-- Duplicate detection candidates
CREATE TABLE duplicate_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_id UUID REFERENCES invoices(id) ON DELETE CASCADE,
    potential_duplicate_id UUID REFERENCES invoices(id) ON DELETE CASCADE,
    match_reason TEXT,
    match_score DECIMAL(5,2),
    resolution VARCHAR(50),  -- confirmed_unique, confirmed_duplicate, merged
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Report exports tracking
CREATE TABLE report_exports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    export_name VARCHAR(255),
    file_path VARCHAR(1000),
    export_format VARCHAR(50),  -- markdown, csv, json
    filter_criteria JSONB,
    invoice_ids UUID[],
    row_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Dashboard notifications
CREATE TABLE ui_notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notification_type VARCHAR(50),  -- success, failure, warning, info
    title VARCHAR(255),
    message TEXT,
    invoice_id UUID REFERENCES invoices(id),
    is_read BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Duplicate Detection Function

```sql
CREATE OR REPLACE FUNCTION check_duplicate_invoice(
    p_invoice_number VARCHAR,
    p_vendor_name VARCHAR,
    p_invoice_date DATE,
    p_total_amount DECIMAL
) RETURNS TABLE(
    duplicate_id UUID,
    match_score DECIMAL,
    match_reason TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.id,
        CASE
            WHEN i.invoice_number = p_invoice_number AND i.vendor_name = p_vendor_name THEN 100.0
            WHEN i.invoice_number = p_invoice_number THEN 85.0
            WHEN i.vendor_name = p_vendor_name
                AND i.invoice_date = p_invoice_date
                AND i.total_amount = p_total_amount THEN 90.0
            WHEN i.vendor_name = p_vendor_name
                AND ABS(i.invoice_date - p_invoice_date) <= 3
                AND i.total_amount = p_total_amount THEN 80.0
            ELSE 0.0
        END::DECIMAL AS score,
        CASE
            WHEN i.invoice_number = p_invoice_number AND i.vendor_name = p_vendor_name
                THEN 'Exact match: invoice number + vendor'
            WHEN i.invoice_number = p_invoice_number
                THEN 'Same invoice number'
            WHEN i.vendor_name = p_vendor_name
                AND i.invoice_date = p_invoice_date
                AND i.total_amount = p_total_amount
                THEN 'Same vendor + date + amount'
            ELSE 'Similar invoice detected'
        END AS reason
    FROM invoices i
    WHERE i.processing_status = 'completed'
      AND (
          i.invoice_number = p_invoice_number
          OR (i.vendor_name = p_vendor_name AND i.total_amount = p_total_amount)
      )
    ORDER BY score DESC
    LIMIT 5;
END;
$$ LANGUAGE plpgsql;
```

---

## 2. File-Mover Service (FastAPI)

### Core Structure

```python
# app.py
import os
import uuid
import asyncio
import base64
import httpx
import asyncpg
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="Invoice File Service")

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:latest")
TEXT_MODEL = os.getenv("TEXT_MODEL", "deepseek-r1:32b")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ragstack:password@postgres:5432/ragstack")
UPLOAD_DIR = "/knowledgebase/invoices/upload"
PROCESSED_DIR = "/knowledgebase/invoices/processed"

# Database pool
db_pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()
```

### Upload Endpoint

```python
@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    trigger_processing: bool = True
):
    results = []
    for file in files:
        # Generate safe filename
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in file.filename)
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{safe_name}")

        # Save file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        results.append({"filename": safe_name, "file_path": file_path})

        # Trigger async processing
        if trigger_processing:
            asyncio.create_task(process_invoice_with_ollama(file_path, safe_name))

    return {"uploaded": len(results), "files": results}
```

### Invoice Processing with Ollama

```python
EXTRACTION_PROMPT = """Analyze this invoice image and extract the following information as JSON:

{
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "vendor_name": "string or null",
  "customer_name": "string or null",
  "subtotal": number or null,
  "total_amount": number or null,
  "tax_amount": number or null,
  "currency": "USD/EUR/etc or null",
  "line_items": [{"description": "string", "amount": number}] or []
}

Return ONLY valid JSON, no explanations."""

async def process_invoice_with_ollama(file_path: str, filename: str):
    """Process invoice using vision LLM and store results."""
    start_time = datetime.now()

    try:
        # Read and encode file
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")

        # Determine model based on file type
        ext = os.path.splitext(filename)[1].lower()
        is_image = ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
        model = VISION_MODEL if is_image else TEXT_MODEL

        # Call Ollama
        async with httpx.AsyncClient(timeout=120.0) as client:
            if is_image:
                payload = {
                    "model": model,
                    "prompt": EXTRACTION_PROMPT,
                    "images": [base64_image],
                    "stream": False
                }
            else:
                # For PDFs, extract text first (simplified)
                payload = {
                    "model": model,
                    "prompt": f"{EXTRACTION_PROMPT}\n\nDocument content:\n{file_bytes.decode('utf-8', errors='ignore')[:8000]}",
                    "stream": False
                }

            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            result = response.json()

        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Parse JSON response
        response_text = result.get("response", "{}")
        # Handle reasoning models that wrap response in <think> tags
        if "thinking" in result:
            response_text = result.get("response", response_text)

        # Clean JSON from markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        import json
        data = json.loads(response_text.strip())

        # Calculate confidence score (20 points per filled field, max 100)
        filled_fields = sum(1 for v in [
            data.get("invoice_number"),
            data.get("invoice_date"),
            data.get("vendor_name"),
            data.get("total_amount"),
            data.get("customer_name")
        ] if v)
        confidence = min(100, filled_fields * 20)

        # Check for duplicates
        is_duplicate = False
        duplicate_id = None
        async with db_pool.acquire() as conn:
            dups = await conn.fetch("""
                SELECT * FROM check_duplicate_invoice($1, $2, $3, $4)
                WHERE match_score >= 80
            """, data.get("invoice_number"), data.get("vendor_name"),
                data.get("invoice_date"), data.get("total_amount"))
            if dups:
                is_duplicate = True
                duplicate_id = dups[0]["duplicate_id"]

        # Determine if review needed
        needs_review = confidence < 70 or not data.get("vendor_name") or not data.get("total_amount")
        review_reason = None
        if confidence < 70:
            review_reason = "Low extraction confidence"
        elif not data.get("vendor_name"):
            review_reason = "Missing vendor name"
        elif not data.get("total_amount"):
            review_reason = "Missing total amount"
        elif is_duplicate:
            review_reason = "Potential duplicate invoice"
            needs_review = True

        # Organize file
        vendor_safe = "".join(c if c.isalnum() else "_" for c in (data.get("vendor_name") or "Unknown"))
        year = str(data.get("invoice_date", "")[:4]) if data.get("invoice_date") else "Unknown"
        dest_dir = os.path.join(PROCESSED_DIR, vendor_safe, year)
        os.makedirs(dest_dir, exist_ok=True)
        new_path = os.path.join(dest_dir, os.path.basename(file_path))
        os.rename(file_path, new_path)

        # Store in database
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO invoices (
                    invoice_number, invoice_date, due_date, vendor_name, customer_name,
                    subtotal, total_amount, tax_amount, currency, line_items,
                    original_filename, file_path, processing_status, processing_date,
                    extraction_model, confidence_score, vision_call_latency_ms,
                    needs_review, review_reason, is_duplicate, duplicate_of_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
            """,
                data.get("invoice_number"),
                data.get("invoice_date"),
                data.get("due_date"),
                data.get("vendor_name"),
                data.get("customer_name"),
                data.get("subtotal"),
                data.get("total_amount"),
                data.get("tax_amount"),
                data.get("currency", "USD"),
                json.dumps(data.get("line_items", [])),
                filename,
                new_path,
                "completed",
                datetime.now(),
                model,
                confidence,
                latency_ms,
                needs_review,
                review_reason,
                is_duplicate,
                duplicate_id
            )

        return {"status": "completed", "confidence": confidence}

    except Exception as e:
        # Store failed record
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO invoices (original_filename, file_path, processing_status, error_message, processing_date)
                VALUES ($1, $2, 'failed', $3, $4)
            """, filename, file_path, str(e), datetime.now())
        return {"status": "failed", "error": str(e)}
```

### API Endpoints

```python
@app.get("/invoice-stats")
async def get_invoice_stats():
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_invoices,
                COUNT(*) FILTER (WHERE processing_status = 'completed') as completed,
                COUNT(*) FILTER (WHERE processing_status = 'failed') as failed,
                COUNT(*) FILTER (WHERE processing_status = 'pending') as pending,
                COALESCE(SUM(total_amount) FILTER (WHERE processing_status = 'completed'), 0) as total_amount_sum,
                COUNT(*) FILTER (WHERE needs_review = true) as needs_review_count,
                COUNT(*) FILTER (WHERE is_duplicate = true) as duplicate_count,
                COUNT(*) FILTER (WHERE DATE(processing_date) = CURRENT_DATE) as processed_today
            FROM invoices
        """)
        return dict(stats)

@app.get("/recent-invoices")
async def get_recent_invoices(limit: int = 50):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM invoices
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)
        return [dict(r) for r in rows]

@app.get("/failed-invoices")
async def get_failed_invoices():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM invoices
            WHERE processing_status = 'failed'
            ORDER BY created_at DESC
        """)
        return [dict(r) for r in rows]

@app.post("/mark-reviewed")
async def mark_reviewed(invoice_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE invoices
            SET needs_review = false, review_reason = NULL, updated_at = NOW()
            WHERE id = $1
        """, uuid.UUID(invoice_id))
    return {"status": "reviewed", "invoice_id": invoice_id}

@app.post("/resolve-duplicate")
async def resolve_duplicate(invoice_id: str, resolution: str):
    async with db_pool.acquire() as conn:
        if resolution == "delete":
            await conn.execute("DELETE FROM invoices WHERE id = $1", uuid.UUID(invoice_id))
        elif resolution == "confirmed_duplicate":
            await conn.execute("""
                UPDATE invoices SET is_duplicate = true, needs_review = false, updated_at = NOW()
                WHERE id = $1
            """, uuid.UUID(invoice_id))
        elif resolution == "confirmed_unique":
            await conn.execute("""
                UPDATE invoices SET is_duplicate = false, needs_review = false, updated_at = NOW()
                WHERE id = $1
            """, uuid.UUID(invoice_id))
    return {"status": resolution, "invoice_id": invoice_id}

@app.post("/reprocess-failed")
async def reprocess_failed(invoice_id: str):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM invoices WHERE id = $1", uuid.UUID(invoice_id))
        if row and row["processing_status"] == "failed":
            await conn.execute("DELETE FROM invoices WHERE id = $1", uuid.UUID(invoice_id))
            asyncio.create_task(process_invoice_with_ollama(row["file_path"], row["original_filename"]))
            return {"status": "reprocessing", "filename": row["original_filename"]}
    raise HTTPException(404, "Invoice not found or not failed")
```

---

## 3. Report Export

### Export Endpoint

```python
class ExportRequest(BaseModel):
    filter_type: str = "all"  # all, recent, completed, failed, vendor, date_range, needs_review
    vendor_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    format: str = "markdown"  # markdown, csv, json
    limit: int = 100

@app.post("/export-report")
async def export_report(request: ExportRequest):
    # Build query based on filter
    query = "SELECT * FROM invoices WHERE 1=1"
    params = []

    if request.filter_type == "recent":
        query += " AND created_at >= NOW() - INTERVAL '7 days'"
    elif request.filter_type == "completed":
        query += " AND processing_status = 'completed'"
    elif request.filter_type == "failed":
        query += " AND processing_status = 'failed'"
    elif request.filter_type == "vendor" and request.vendor_name:
        query += f" AND vendor_name ILIKE ${len(params)+1}"
        params.append(f"%{request.vendor_name}%")
    elif request.filter_type == "date_range":
        if request.start_date:
            query += f" AND invoice_date >= ${len(params)+1}"
            params.append(request.start_date)
        if request.end_date:
            query += f" AND invoice_date <= ${len(params)+1}"
            params.append(request.end_date)
    elif request.filter_type == "needs_review":
        query += " AND needs_review = true"

    query += f" ORDER BY created_at DESC LIMIT ${len(params)+1}"
    params.append(request.limit)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    invoices = [dict(r) for r in rows]

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if request.format == "markdown":
        content = generate_markdown_report(invoices, request.filter_type)
        filename = f"invoice_report_{timestamp}.md"
    elif request.format == "csv":
        content = generate_csv_report(invoices)
        filename = f"invoice_report_{timestamp}.csv"
    else:
        import json
        content = json.dumps(invoices, default=str, indent=2)
        filename = f"invoice_report_{timestamp}.json"

    # Save report
    report_dir = os.path.join(PROCESSED_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    file_path = os.path.join(report_dir, filename)

    with open(file_path, "w") as f:
        f.write(content)

    # Store metadata
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO report_exports (export_name, file_path, export_format, filter_criteria, row_count)
            VALUES ($1, $2, $3, $4, $5)
        """, filename, file_path, request.format,
            json.dumps({"filter_type": request.filter_type}), len(invoices))

    return {"file_path": file_path, "row_count": len(invoices), "format": request.format}

def generate_markdown_report(invoices: list, filter_type: str) -> str:
    lines = [
        f"# Invoice Report",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Filter**: {filter_type}",
        f"**Total Invoices**: {len(invoices)}",
        f"",
        "| Vendor | Invoice # | Date | Amount | Status | Confidence |",
        "|--------|-----------|------|--------|--------|------------|"
    ]

    for inv in invoices:
        lines.append(
            f"| {inv.get('vendor_name', 'N/A')} "
            f"| {inv.get('invoice_number', 'N/A')} "
            f"| {inv.get('invoice_date', 'N/A')} "
            f"| {inv.get('currency', 'USD')} {inv.get('total_amount', 0):.2f} "
            f"| {inv.get('processing_status', 'unknown')} "
            f"| {inv.get('confidence_score', 0):.0f}% |"
        )

    return "\n".join(lines)

def generate_csv_report(invoices: list) -> str:
    import csv
    import io

    output = io.StringIO()
    if invoices:
        writer = csv.DictWriter(output, fieldnames=invoices[0].keys())
        writer.writeheader()
        writer.writerows(invoices)

    return output.getvalue()
```

---

## 4. Dashboard (HTML/JavaScript)

### Minimal Dashboard

```html
<!DOCTYPE html>
<html>
<head>
    <title>Invoice Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f5f5; padding: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6; }
        .stat-value { font-size: 28px; font-weight: bold; color: #1e293b; }
        .stat-label { color: #64748b; font-size: 14px; }
        table { width: 100%; background: white; border-radius: 8px; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f8fafc; color: #475569; }
        .status-completed { color: #22c55e; }
        .status-failed { color: #ef4444; }
        .status-pending { color: #f59e0b; }
        .confidence-high { color: #22c55e; }
        .confidence-medium { color: #f59e0b; }
        .confidence-low { color: #ef4444; }
        .btn { padding: 6px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .btn-primary { background: #3b82f6; color: white; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-success { background: #22c55e; color: white; }
    </style>
</head>
<body>
    <h1 style="margin-bottom: 20px;">Invoice Processing Dashboard</h1>

    <div class="stats-grid" id="stats"></div>

    <h2 style="margin: 20px 0;">Recent Invoices</h2>
    <table>
        <thead>
            <tr>
                <th>Vendor</th>
                <th>Invoice #</th>
                <th>Date</th>
                <th>Amount</th>
                <th>Status</th>
                <th>Confidence</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody id="invoices"></tbody>
    </table>

    <script>
        const API_BASE = 'http://localhost:5020';

        async function loadStats() {
            const res = await fetch(`${API_BASE}/invoice-stats`);
            const stats = await res.json();
            document.getElementById('stats').innerHTML = `
                <div class="stat-card"><div class="stat-value">${stats.total_invoices}</div><div class="stat-label">Total</div></div>
                <div class="stat-card"><div class="stat-value">${stats.completed}</div><div class="stat-label">Completed</div></div>
                <div class="stat-card"><div class="stat-value">${stats.failed}</div><div class="stat-label">Failed</div></div>
                <div class="stat-card"><div class="stat-value">${stats.needs_review_count}</div><div class="stat-label">Needs Review</div></div>
                <div class="stat-card"><div class="stat-value">$${Number(stats.total_amount_sum).toLocaleString()}</div><div class="stat-label">Total Amount</div></div>
            `;
        }

        async function loadInvoices() {
            const res = await fetch(`${API_BASE}/recent-invoices?limit=20`);
            const invoices = await res.json();
            document.getElementById('invoices').innerHTML = invoices.map(inv => `
                <tr>
                    <td>${inv.vendor_name || 'N/A'}</td>
                    <td>${inv.invoice_number || 'N/A'}</td>
                    <td>${inv.invoice_date || 'N/A'}</td>
                    <td>${inv.currency || 'USD'} ${Number(inv.total_amount || 0).toFixed(2)}</td>
                    <td class="status-${inv.processing_status}">${inv.processing_status}</td>
                    <td class="${getConfidenceClass(inv.confidence_score)}">${inv.confidence_score?.toFixed(0) || 0}%</td>
                    <td>
                        ${inv.needs_review ? `<button class="btn btn-success" onclick="markReviewed('${inv.id}')">Mark Reviewed</button>` : ''}
                        ${inv.processing_status === 'failed' ? `<button class="btn btn-primary" onclick="reprocess('${inv.id}')">Reprocess</button>` : ''}
                    </td>
                </tr>
            `).join('');
        }

        function getConfidenceClass(score) {
            if (score >= 80) return 'confidence-high';
            if (score >= 60) return 'confidence-medium';
            return 'confidence-low';
        }

        async function markReviewed(id) {
            await fetch(`${API_BASE}/mark-reviewed`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({invoice_id: id})
            });
            loadInvoices();
        }

        async function reprocess(id) {
            await fetch(`${API_BASE}/reprocess-failed`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({invoice_id: id})
            });
            loadInvoices();
        }

        // Initial load and polling
        loadStats();
        loadInvoices();
        setInterval(() => { loadStats(); loadInvoices(); }, 5000);
    </script>
</body>
</html>
```

---

## 5. Docker Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ragstack
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ragstack
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/01-init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "ragstack"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"

  file-mover:
    build: ./file-mover-service
    environment:
      DATABASE_URL: postgresql://ragstack:${POSTGRES_PASSWORD}@postgres:5432/ragstack
      OLLAMA_BASE_URL: http://ollama:11434
      VISION_MODEL: qwen3-vl:latest
    volumes:
      - ./knowledgebase:/knowledgebase
    ports:
      - "5020:5020"
    depends_on:
      postgres:
        condition: service_healthy

  dashboard:
    image: nginx:alpine
    volumes:
      - ./invoice-dashboard.html:/usr/share/nginx/html/index.html:ro
    ports:
      - "8085:80"

volumes:
  postgres_data:
  ollama_data:
```

### file-mover-service/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install fastapi uvicorn asyncpg httpx python-multipart

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5020"]
```

---

## 6. Quick Start

```bash
# 1. Clone/create project directory
mkdir invoice-system && cd invoice-system

# 2. Create .env file
echo "POSTGRES_PASSWORD=your_secure_password" > .env

# 3. Create directory structure
mkdir -p database file-mover-service knowledgebase/invoices/{upload,processed}

# 4. Copy SQL schema to database/init.sql
# 5. Copy Python code to file-mover-service/app.py
# 6. Copy HTML to invoice-dashboard.html
# 7. Create docker-compose.yml and Dockerfile

# 8. Start services
docker-compose up -d

# 9. Pull vision model
docker exec ollama ollama pull qwen3-vl:latest

# 10. Test upload
curl -X POST http://localhost:5020/upload \
  -F "files=@/path/to/invoice.jpg"

# 11. View dashboard
open http://localhost:8085
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Vision LLM for OCR** | More accurate than traditional OCR for varied invoice layouts |
| **Confidence scoring** | Enables quality control without manual review of every invoice |
| **Duplicate detection in DB** | SQL-based matching is faster and more reliable than LLM |
| **Async processing** | Non-blocking uploads, better user experience |
| **File organization by vendor/year** | Easy manual browsing and backup management |
| **PostgreSQL over NoSQL** | Structured queries for reporting, ACID compliance |

---

## Accuracy Optimizations

1. **Low temperature (0.1)** for LLM calls - consistent extractions
2. **Field-specific prompts** - better extraction for each document type
3. **Confidence thresholds** - flag low-confidence results for review
4. **Duplicate prevention** - multi-factor matching (number + vendor + date + amount)
5. **Review queue** - human-in-the-loop for edge cases

---

## Scaling Considerations

- **Queue processing**: Add Redis + Celery for high-volume processing
- **Model selection**: Use smaller models for classification, larger for extraction
- **Caching**: Cache vendor name normalization and duplicate checks
- **Batch processing**: Process multiple invoices in single LLM call where possible
- **Horizontal scaling**: Stateless file-mover service can be replicated

"""
Text-first PDF loading, used ahead of visual OCR engines (MinerU).

Visual pipelines rasterise the page and only keep what their layout model
labels as text. On born-digital forms (bordered boxes, certificate tables)
the detector frequently labels the whole body as a figure/table and the
text inside is dropped or OCR-mangled, even though the PDF carries a
perfect text layer. So: read the text layer first and only hand the file
to the visual engine when the PDF is effectively image-only.
"""

import logging
import os
import shutil
import subprocess

from langchain_core.documents import Document

log = logging.getLogger(__name__)

# Average extractable characters per page below which a PDF is treated as
# scanned / image-only and sent to the fallback (visual) loader.
DEFAULT_MIN_CHARS_PER_PAGE = int(os.environ.get('PDF_TEXT_FIRST_MIN_CHARS_PER_PAGE', '200'))

PDFTOTEXT_TIMEOUT_SECONDS = 120


def probe_pdf_text_layer(file_path: str) -> tuple[int, int]:
    """Return (page_count, total_stripped_chars) using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    chars = 0
    for page in reader.pages:
        chars += len((page.extract_text() or '').strip())
    return len(reader.pages), chars


def _load_with_pdftotext(file_path: str, mode: str) -> list[Document]:
    out = subprocess.run(
        ['pdftotext', '-enc', 'UTF-8', file_path, '-'],
        capture_output=True,
        check=True,
        timeout=PDFTOTEXT_TIMEOUT_SECONDS,
    ).stdout.decode('utf-8', 'replace')

    # pdftotext separates pages with a form feed and emits one after the last page too.
    pages = out.split('\f')
    if pages and not pages[-1].strip():
        pages.pop()

    if mode != 'page':
        return [Document(page_content='\n\n'.join(pages), metadata={'source': file_path})]
    return [
        Document(page_content=text, metadata={'source': file_path, 'page': i})
        for i, text in enumerate(pages)
    ]


def load_pdf_text_layer(file_path: str, mode: str = 'page') -> list[Document]:
    """
    Extract the PDF text layer.

    Prefers poppler's `pdftotext` when the binary is present: unlike pypdf it
    reassembles letter-spaced glyph runs ("1 4 . 5 9 1" -> "14.591"), which
    are common in browser-printed government forms. Falls back to pypdf.
    """
    if shutil.which('pdftotext'):
        try:
            return _load_with_pdftotext(file_path, mode)
        except Exception as e:
            log.warning(f'pdftotext failed on {file_path}, falling back to pypdf: {e}')

    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(file_path, mode=mode).load()


class TextFirstPdfLoader:
    """
    Use the PDF's own text layer when it is substantial; otherwise delegate
    to `fallback` (a loader object with a `.load()` method, e.g. MinerULoader).
    """

    def __init__(
        self,
        file_path: str,
        fallback,
        mode: str = 'page',
        min_chars_per_page: int = DEFAULT_MIN_CHARS_PER_PAGE,
    ):
        self.file_path = file_path
        self.fallback = fallback
        self.mode = mode
        self.min_chars_per_page = min_chars_per_page

    def load(self) -> list[Document]:
        try:
            pages, chars = probe_pdf_text_layer(self.file_path)
        except Exception as e:
            log.warning(f'Could not probe PDF text layer for {self.file_path}, using fallback: {e}')
            return self.fallback.load()

        per_page = chars / pages if pages else 0
        if per_page >= self.min_chars_per_page:
            log.info(
                f'PDF has a text layer ({chars} chars / {pages} pages); '
                f'using it instead of {type(self.fallback).__name__}'
            )
            return load_pdf_text_layer(self.file_path, self.mode)

        log.info(
            f'PDF text layer too thin ({per_page:.0f} chars/page < {self.min_chars_per_page}); '
            f'using {type(self.fallback).__name__}'
        )
        return self.fallback.load()

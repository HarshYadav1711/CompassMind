"""Backward-compatible PDF parsing API (delegates to ingestion layer)."""

from __future__ import annotations

from compassmind.ingestion.pdf_io import parse_test_pdf as parse_pdf_rows

__all__ = ["parse_pdf_rows"]

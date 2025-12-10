"""Business logic for PDF processing."""

import logging
from typing import Callable, Dict, Any

from src.app.services.pdf.file_pdf_pipeline import process_pdf_from_url


logging.basicConfig(level=logging.INFO)


class PDFService:
    """Service for PDF processing and embedding operations."""

    @staticmethod
    def process_pdf_pages(file_url: str, embed_batch: Callable) -> Dict[str, Any]:
        """
        Process PDF from URL and generate embeddings for each page.

        Args:
            file_url: URL of the PDF file
            embed_batch: Function to embed batch of texts

        Returns:
            Dictionary with page embeddings and metadata
        """
        # Process PDF using the pipeline
        result = process_pdf_from_url(file_url, embed_batch)

        # Format response
        embeddings = [
            {
                "pageNumber": page["page_number"],
                "charCount": page["char_count"],
                "embedding": page["embedding"],
            }
            for page in result["pages"]
        ]

        return {
            "embeddings": embeddings,
            "pageCount": result["page_count"],
        }

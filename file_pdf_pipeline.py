"""
PDF Pipeline Module - Implements SOLID principles for PDF processing and embedding.

This module provides interfaces and implementations for:
- Fetching PDF files from remote URLs
- Extracting text from PDF pages
- Embedding text content
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import io
import requests
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO)


# ============================================================================
# INTERFACES (Following Interface Segregation Principle)
# ============================================================================


class IFileFetcher(ABC):
    """Interface for fetching files from remote sources."""

    @abstractmethod
    def fetch(self, url: str) -> bytes:
        """
        Fetch file content from a URL.

        Args:
            url: The URL to fetch the file from

        Returns:
            bytes: The file content as bytes

        Raises:
            RuntimeError: If fetching fails
        """
        pass


class ITextExtractor(ABC):
    """Interface for extracting text from documents."""

    @abstractmethod
    def extract_pages(self, file_content: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from document pages.

        Args:
            file_content: The document content as bytes

        Returns:
            List of dictionaries containing page information

        Raises:
            RuntimeError: If extraction fails
        """
        pass


class IEmbeddingGenerator(ABC):
    """Interface for generating embeddings from text."""

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        pass


# ============================================================================
# IMPLEMENTATIONS (Following Single Responsibility Principle)
# ============================================================================


class HTTPFileFetcher(IFileFetcher):
    """Fetches files from HTTP/HTTPS URLs."""

    def __init__(self, timeout: int = 30):
        """
        Initialize the HTTP file fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    def fetch(self, url: str) -> bytes:
        """Fetch file content from HTTP/HTTPS URL."""
        try:
            logging.info(f"Fetching PDF from URL: {url}")
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            logging.info(f"Successfully fetched PDF ({len(response.content)} bytes)")
            return response.content
        except requests.RequestException as e:
            logging.error(f"Failed to fetch file from {url}: {e}")
            raise RuntimeError(f"Failed to fetch file: {str(e)}")


class PDFTextExtractor(ITextExtractor):
    """Extracts text from PDF files page by page."""

    def extract_pages(self, file_content: bytes) -> List[Dict[str, Any]]:
        """Extract text from each page of a PDF file."""
        try:
            logging.info("Starting PDF text extraction")
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)

            pages = []
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                # Clean and normalize text
                text = self._clean_text(text)

                if text.strip():  # Only include pages with actual content
                    pages.append(
                        {
                            "page_number": page_num,
                            "text": text,
                            "char_count": len(text),
                        }
                    )
                    logging.debug(f"Extracted page {page_num}: {len(text)} characters")
                else:
                    logging.warning(f"Page {page_num} has no extractable text")

            logging.info(f"Successfully extracted {len(pages)} pages from PDF")
            return pages

        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Replace multiple newlines with single newline
        import re

        text = re.sub(r"\n+", "\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        # Strip leading/trailing whitespace
        text = text.strip()

        return text


class BatchEmbeddingGenerator(IEmbeddingGenerator):
    """Generates embeddings using a batch embedding function."""

    def __init__(self, embed_batch_fn: Callable[[List[str]], Any]):
        """
        Initialize the embedding generator.

        Args:
            embed_batch_fn: Function that takes a list of texts and returns embeddings
        """
        self.embed_batch_fn = embed_batch_fn

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            logging.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embed_batch_fn(texts)

            # Convert to list of lists if needed
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()
            elif not isinstance(embeddings, list):
                embeddings = list(embeddings)

            logging.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logging.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")


# ============================================================================
# SERVICE (Following Dependency Inversion Principle)
# ============================================================================


class PDFEmbeddingService:
    """
    Service for processing PDF files and generating embeddings.

    This class orchestrates the entire pipeline while depending on abstractions
    (interfaces) rather than concrete implementations (Dependency Inversion Principle).
    """

    def __init__(
        self,
        file_fetcher: IFileFetcher,
        text_extractor: ITextExtractor,
        embedding_generator: IEmbeddingGenerator,
    ):
        """
        Initialize the PDF embedding service.

        Args:
            file_fetcher: Implementation of IFileFetcher
            text_extractor: Implementation of ITextExtractor
            embedding_generator: Implementation of IEmbeddingGenerator
        """
        self.file_fetcher = file_fetcher
        self.text_extractor = text_extractor
        self.embedding_generator = embedding_generator

    def process_pdf(self, file_url: str) -> Dict[str, Any]:
        """
        Process a PDF file from URL and generate embeddings for each page.

        Args:
            file_url: URL of the PDF file

        Returns:
            Dictionary containing embeddings and metadata for each page

        Raises:
            RuntimeError: If any step in the pipeline fails
        """
        # Step 1: Fetch the PDF file
        file_content = self.file_fetcher.fetch(file_url)

        # Step 2: Extract text from each page
        pages = self.text_extractor.extract_pages(file_content)

        if not pages:
            raise RuntimeError("No text content could be extracted from the PDF")

        # Step 3: Generate embeddings for all pages
        texts = [page["text"] for page in pages]
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # Step 4: Combine results
        result = []
        for page, embedding in zip(pages, embeddings):
            result.append(
                {
                    "page_number": page["page_number"],
                    "char_count": page["char_count"],
                    "embedding": embedding,
                }
            )

        logging.info(f"PDF processing complete: {len(result)} pages with embeddings")

        return {"page_count": len(result), "pages": result}


# ============================================================================
# FACTORY (Following Open/Closed Principle)
# ============================================================================


class PDFEmbeddingServiceFactory:
    """
    Factory for creating PDFEmbeddingService instances.

    This allows the system to be extended with new implementations
    without modifying existing code (Open/Closed Principle).
    """

    @staticmethod
    def create_service(
        embed_batch_fn: Callable[[List[str]], Any],
    ) -> PDFEmbeddingService:
        """
        Create a PDFEmbeddingService with default implementations.

        Args:
            embed_batch_fn: Function to use for batch embedding

        Returns:
            Configured PDFEmbeddingService instance
        """
        file_fetcher = HTTPFileFetcher(timeout=30)
        text_extractor = PDFTextExtractor()
        embedding_generator = BatchEmbeddingGenerator(embed_batch_fn)

        return PDFEmbeddingService(
            file_fetcher=file_fetcher,
            text_extractor=text_extractor,
            embedding_generator=embedding_generator,
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def process_pdf_from_url(
    file_url: str, embed_batch_fn: Callable[[List[str]], Any]
) -> Dict[str, Any]:
    """
    Convenience function to process a PDF from a URL.

    Args:
        file_url: URL of the PDF file
        embed_batch_fn: Function to use for batch embedding

    Returns:
        Dictionary containing embeddings and metadata for each page
    """
    service = PDFEmbeddingServiceFactory.create_service(embed_batch_fn)
    return service.process_pdf(file_url)

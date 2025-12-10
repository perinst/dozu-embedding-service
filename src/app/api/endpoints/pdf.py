"""PDF processing endpoints."""

import logging
from typing import Dict
from fastapi import APIRouter, HTTPException, Depends

from ...schemas.pdf_schema import PDFEmbeddingRequest
from ...services.pdf.pdf_service import PDFService
from ..deps import get_embed_batch

router = APIRouter(prefix="/pdf", tags=["pdf"])
logging.basicConfig(level=logging.INFO)


@router.post("/page/embedding", response_model=Dict)
def pdf_page_embedding(
    req: PDFEmbeddingRequest,
    embed_batch=Depends(get_embed_batch),
):
    """
    Process PDF from URL and generate embeddings for each page.
    """
    try:
        result = PDFService.process_pdf_pages(req.fileUrl, embed_batch)
        return result

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error in PDF embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

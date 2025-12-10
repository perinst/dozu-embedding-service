"""Central API router that includes all endpoint routers."""

from fastapi import APIRouter
from .endpoints import health, youtube, embedding, pdf

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router)
api_router.include_router(youtube.router)
api_router.include_router(embedding.router)
api_router.include_router(pdf.router)

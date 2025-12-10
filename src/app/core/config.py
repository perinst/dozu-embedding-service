"""Configuration settings for the application."""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    MODEL_NAME: str = os.getenv("MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
    HF_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_HUB_TOKEN")
    DEVICE_SETTING: str = os.getenv("DEVICE", "auto")

    # API Configuration
    API_TITLE: str = "Dozu Embedding Service"
    API_VERSION: str = "1.0.0"


settings = Settings()

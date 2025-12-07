"""Configuration and environment setup for the Plant Classifier API."""

import os
from functools import lru_cache
from openai import OpenAI
from tavily import TavilyClient
from langchain.chat_models import init_chat_model


# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"
OPENAI_MODEL_SMALL = "gpt-4o-mini"
OPENAI_MODEL_LEGACY = "gpt-3.5-turbo"


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your environment or .env file."
        )
    return api_key


def get_tavily_api_key() -> str:
    """Get Tavily API key from environment variables."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Please set it in your environment or .env file."
        )
    return api_key


@lru_cache()
def get_openai_client() -> OpenAI:
    """Get OpenAI client instance (cached)."""
    get_openai_api_key()  # Validate key exists
    return OpenAI()  # Reads API key from environment


@lru_cache()
def get_tavily_client() -> TavilyClient:
    """Get Tavily client instance (cached)."""
    get_tavily_api_key()  # Validate key exists
    return TavilyClient()  # Reads API key from environment


@lru_cache()
def get_openai_model():
    """Get configured LangChain OpenAI model instance (cached)."""
    get_openai_api_key()  # Validate key exists
    return init_chat_model(
        OPENAI_MODEL_SMALL,
        model_provider="openai",
        temperature=0.8
    )

"""Application configuration and safe defaults for local execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    app_name: str = "mcp-ticket-rag"
    app_env: str = "dev"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    owner: str = "langchain-ai"
    repo: str = "langchain"

    max_issues: int = 20
    max_ocr_images: int = 10
    retrieval_top_k: int = 5

    include_comments: bool = True
    include_repo_docs: bool = False

    github_token: str | None = None
    github_api_base: str = "https://api.github.com"

    github_mcp_url: str = "https://api.githubcopilot.com/mcp/"
    github_mcp_transport: str = "stdio"
    github_mcp_server_command: str = "docker"
    github_mcp_server_args: str = (
        "run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN -e GITHUB_TOOLSETS=issues,repos -e GITHUB_DYNAMIC_TOOLSETS=1 "
        "-e GITHUB_READ_ONLY=1 ghcr.io/github/github-mcp-server"
    )

    mistral_base_url: str = "http://localhost:11434"
    semantic_resolver_model: str = "phi3:mini"
    semantic_resolver_connect_timeout_seconds: int = 5
    semantic_resolver_first_byte_timeout_seconds: int = 20
    semantic_resolver_timeout_seconds: int = 60

    answer_model: str = "mistral"
    answer_connect_timeout_seconds: int = 5
    answer_first_byte_timeout_seconds: int = 45
    answer_timeout_seconds: int = 240

    phi3_vision_model: str = "llava-phi3"
    phi3_fallback_model: str = "phi3:latest"
    vision_connect_timeout_seconds: int = 5
    vision_first_byte_timeout_seconds: int = 30
    phi3_timeout_seconds: int = 120
    ollama_max_retries: int = 1
    ollama_enable_warmup: bool = False
    ollama_warmup_on_start: bool = False

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 16
    embedding_local_files_only: bool = True
    hf_cache_dir: str | None = None

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chroma_dir: Path = Path("data/chroma")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    """Load application settings from environment with conservative defaults."""

    load_dotenv(override=False)

    mistral_endpoint = (
        os.getenv("MISTRAL_ENDPOINT")
        or os.getenv("MISTRAL_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    )

    return Settings(
        app_name=os.getenv("APP_NAME", "mcp-ticket-rag"),
        app_env=os.getenv("APP_ENV", "dev"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        owner=os.getenv("GITHUB_OWNER", "langchain-ai"),
        repo=os.getenv("GITHUB_REPO", "langchain"),
        max_issues=int(os.getenv("MAX_ISSUES", "20")),
        max_ocr_images=int(os.getenv("MAX_OCR_IMAGES", "10")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        include_comments=_env_bool("INCLUDE_COMMENTS", True),
        include_repo_docs=_env_bool("INCLUDE_REPO_DOCS", False),
        github_token=os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
        github_api_base=os.getenv("GITHUB_API_BASE", "https://api.github.com"),
        github_mcp_url=os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/"),
        github_mcp_transport=os.getenv("GITHUB_MCP_TRANSPORT", "stdio"),
        github_mcp_server_command=os.getenv("GITHUB_MCP_SERVER_COMMAND", "docker"),
        github_mcp_server_args=os.getenv(
            "GITHUB_MCP_SERVER_ARGS",
            (
                "run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN -e GITHUB_TOOLSETS=issues,repos -e GITHUB_DYNAMIC_TOOLSETS=1 "
                "-e GITHUB_READ_ONLY=1 ghcr.io/github/github-mcp-server"
            ),
        ),
        mistral_base_url=mistral_endpoint.rstrip("/"),
        semantic_resolver_model=os.getenv("SEMANTIC_RESOLVER_MODEL", "phi3:mini"),
        semantic_resolver_connect_timeout_seconds=int(os.getenv("SEMANTIC_RESOLVER_CONNECT_TIMEOUT_SECONDS", "5")),
        semantic_resolver_first_byte_timeout_seconds=int(
            os.getenv("SEMANTIC_RESOLVER_FIRST_BYTE_TIMEOUT_SECONDS", "20")
        ),
        semantic_resolver_timeout_seconds=int(os.getenv("SEMANTIC_RESOLVER_TIMEOUT_SECONDS", "60")),
        answer_model=os.getenv("ANSWER_MODEL", os.getenv("MISTRAL_MODEL", "mistral")),
        answer_connect_timeout_seconds=int(os.getenv("ANSWER_CONNECT_TIMEOUT_SECONDS", "5")),
        answer_first_byte_timeout_seconds=int(os.getenv("ANSWER_FIRST_BYTE_TIMEOUT_SECONDS", "45")),
        answer_timeout_seconds=int(os.getenv("ANSWER_TIMEOUT_SECONDS", os.getenv("MISTRAL_TIMEOUT_SECONDS", "240"))),
        phi3_vision_model=os.getenv("PHI3_VISION_MODEL", "llava-phi3"),
        phi3_fallback_model=os.getenv("PHI3_FALLBACK_MODEL", "phi3:latest"),
        vision_connect_timeout_seconds=int(os.getenv("VISION_CONNECT_TIMEOUT_SECONDS", "5")),
        vision_first_byte_timeout_seconds=int(os.getenv("VISION_FIRST_BYTE_TIMEOUT_SECONDS", "30")),
        phi3_timeout_seconds=int(os.getenv("PHI3_TIMEOUT_SECONDS", "120")),
        ollama_max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "1")),
        ollama_enable_warmup=_env_bool("OLLAMA_ENABLE_WARMUP", False),
        ollama_warmup_on_start=_env_bool("OLLAMA_WARMUP_ON_START", False),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
        embedding_local_files_only=_env_bool("EMBEDDING_LOCAL_FILES_ONLY", True),
        hf_cache_dir=os.getenv("HF_CACHE_DIR"),
    )

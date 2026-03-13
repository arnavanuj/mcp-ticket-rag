"""Image download and Phi-3 Vision multimodal analysis utilities."""

from __future__ import annotations

import base64
from pathlib import Path

import requests

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.ollama_client import OllamaClient, OllamaGenerationError

logger = get_logger("app.ingestion.image_ocr")


def download_image(url: str, out_path: Path, timeout: int = 20) -> bool:
    """Download a remote image to disk."""

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return True
    except Exception as exc:
        logger.warning("IMAGE_DOWNLOAD_FAILED | url=%s | error=%s", url, exc)
        return False


def analyze_image_with_phi3(settings: Settings, image_path: Path, image_url: str) -> dict[str, str | bool]:
    """Run local vision analysis via Ollama (`llava-phi3` primary). 

    Returns:
        {
            "success": bool,
            "ocr_text": str,
            "analysis_text": str,
            "model_used": str,
        }
    """

    prompt = (
        "You are analyzing a GitHub issue screenshot/image.\n"
        "1) Extract any visible text (OCR-like).\n"
        "2) Provide a concise technical image description.\n"
        "Respond exactly in this format:\n"
        "OCR_TEXT: <text or NONE>\n"
        "ANALYSIS_TEXT: <technical description>"
    )

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    result = {
        "success": False,
        "ocr_text": "",
        "analysis_text": "",
        "model_used": settings.phi3_vision_model,
        "llm_diagnostics": {},
    }

    model_candidates = [settings.phi3_vision_model, settings.phi3_fallback_model]
    client = OllamaClient(base_url=settings.mistral_base_url, default_max_retries=settings.ollama_max_retries)

    for model in model_candidates:
        try:
            response = client.generate(
                component="vision_analysis",
                model=model,
                prompt=prompt,
                connect_timeout_seconds=settings.vision_connect_timeout_seconds,
                first_byte_timeout_seconds=settings.vision_first_byte_timeout_seconds,
                total_timeout_seconds=settings.phi3_timeout_seconds,
                max_retries=settings.ollama_max_retries,
                images_b64=[image_b64],
                options={"temperature": 0.0},
            )
            text = response.text
            ocr_text, analysis_text = _parse_phi3_output(text)

            result = {
                "success": True,
                "ocr_text": ocr_text,
                "analysis_text": analysis_text,
                "model_used": model,
                "llm_diagnostics": response.diagnostics,
            }
            logger.info(
                "IMAGE_ANALYSIS_DONE | url=%s | path=%s | model=%s | ocr_len=%s | analysis_len=%s",
                image_url,
                image_path,
                model,
                len(ocr_text),
                len(analysis_text),
            )
            return result
        except OllamaGenerationError as exc:
            logger.warning(
                "VISION_ANALYSIS_FAILED | url=%s | path=%s | model=%s | reason=%s",
                image_url,
                image_path,
                model,
                exc.diagnostics.get("fallback_reason", str(exc)),
            )
            result["llm_diagnostics"] = exc.diagnostics

    logger.warning(
        "IMAGE_ANALYSIS_FALLBACK | url=%s | path=%s | reason=llava_phi3_unavailable",
        image_url,
        image_path,
    )
    result["analysis_text"] = "Image analysis unavailable: Phi-3 Vision model could not process this image."
    return result


def _parse_phi3_output(text: str) -> tuple[str, str]:
    """Parse expected Phi-3 structured output."""

    ocr_text = ""
    analysis_text = ""

    for line in text.splitlines():
        clean = line.strip()
        if clean.upper().startswith("OCR_TEXT:"):
            ocr_text = clean.split(":", 1)[1].strip()
        elif clean.upper().startswith("ANALYSIS_TEXT:"):
            analysis_text = clean.split(":", 1)[1].strip()

    if not analysis_text and text:
        analysis_text = text.strip()

    if ocr_text.upper() == "NONE":
        ocr_text = ""

    return ocr_text, analysis_text

"""Shared Ollama client with streaming observability, retries, and warmup."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from app.core.logging import get_logger

logger = get_logger("app.services.ollama_client")


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama generation fails after retries."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(slots=True)
class OllamaResult:
    text: str
    diagnostics: dict[str, Any]


_COUNTER_LOCK = threading.Lock()
_ACTIVE_REQUESTS = 0
_LAST_USED_BY_MODEL: dict[str, float] = {}


def get_active_request_count() -> int:
    with _COUNTER_LOCK:
        return _ACTIVE_REQUESTS


class OllamaClient:
    """Streaming-capable Ollama client with timing diagnostics and retries."""

    def __init__(self, base_url: str, default_max_retries: int = 1) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_max_retries = max(0, int(default_max_retries))

    def generate(
        self,
        *,
        component: str,
        model: str,
        prompt: str,
        connect_timeout_seconds: int,
        first_byte_timeout_seconds: int,
        total_timeout_seconds: int,
        max_retries: int | None = None,
        images_b64: list[str] | None = None,
        options: dict[str, Any] | None = None,
        warmup_before_retry: bool = False,
    ) -> OllamaResult:
        retries = self.default_max_retries if max_retries is None else max(0, int(max_retries))
        request_id = uuid.uuid4().hex[:10]
        prompt_hash = hashlib.sha1(prompt.encode("utf-8", errors="ignore")).hexdigest()[:12]

        diagnostics: dict[str, Any] = {
            "request_id": request_id,
            "component": component,
            "model": model,
            "endpoint": f"{self.base_url}/api/generate",
            "prompt_chars": len(prompt),
            "prompt_hash": prompt_hash,
            "connect_timeout_seconds": connect_timeout_seconds,
            "first_byte_timeout_seconds": first_byte_timeout_seconds,
            "total_timeout_seconds": total_timeout_seconds,
            "retries_attempted": 0,
            "fallback_reason": "",
            "model_state": _estimate_model_state(model),
            "concurrent_request_count_submit": 0,
            "concurrent_request_count_finish": 0,
            "first_byte_latency_ms": None,
            "total_duration_ms": None,
            "response_chars": 0,
            "success": False,
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options or {"temperature": 0.0},
        }
        if images_b64:
            payload["images"] = images_b64

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            diagnostics["retries_attempted"] = attempt
            submit_time = time.perf_counter()
            concurrent_on_submit = _inc_active_requests()
            diagnostics["concurrent_request_count_submit"] = concurrent_on_submit
            logger.info(
                "OLLAMA_REQUEST_SUBMITTED | request_id=%s | component=%s | model=%s | endpoint=%s | prompt_chars=%s | prompt_hash=%s | connect_timeout=%s | first_byte_timeout=%s | total_timeout=%s | concurrent=%s | retries_attempted=%s | model_state=%s",
                request_id,
                component,
                model,
                diagnostics["endpoint"],
                len(prompt),
                prompt_hash,
                connect_timeout_seconds,
                first_byte_timeout_seconds,
                total_timeout_seconds,
                concurrent_on_submit,
                attempt,
                diagnostics["model_state"],
            )

            first_byte_time: float | None = None
            chunks: list[str] = []
            try:
                with requests.post(
                    diagnostics["endpoint"],
                    json=payload,
                    stream=True,
                    timeout=(connect_timeout_seconds, first_byte_timeout_seconds),
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"non_200_response:{response.status_code}")

                    # Keep the socket alive after first byte to tolerate generation time.
                    _set_socket_timeout(response, first_byte_timeout_seconds)

                    for raw_line in response.iter_lines(decode_unicode=True):
                        now = time.perf_counter()
                        elapsed_total = now - submit_time
                        if elapsed_total > total_timeout_seconds:
                            raise TimeoutError("total_generation_timeout")

                        if not raw_line:
                            continue
                        if first_byte_time is None:
                            first_byte_time = now
                            first_ms = (first_byte_time - submit_time) * 1000
                            diagnostics["first_byte_latency_ms"] = round(first_ms, 2)
                            diagnostics["model_state"] = _estimate_state_from_first_byte(model, first_ms)
                            logger.info(
                                "OLLAMA_FIRST_BYTE_RECEIVED | request_id=%s | component=%s | model=%s | first_byte_latency_ms=%.2f | model_state=%s",
                                request_id,
                                component,
                                model,
                                first_ms,
                                diagnostics["model_state"],
                            )
                            # After first byte, switch to total-time-based socket timeout.
                            remaining = max(5.0, float(total_timeout_seconds) - float(first_ms / 1000.0))
                            _set_socket_timeout(response, remaining)

                        parsed = _parse_json_line(raw_line)
                        if not parsed:
                            continue
                        piece = parsed.get("response")
                        if isinstance(piece, str) and piece:
                            chunks.append(piece)
                        if parsed.get("done") is True:
                            break

                final_text = "".join(chunks).strip()
                if not final_text:
                    raise RuntimeError("empty_response")

                total_ms = (time.perf_counter() - submit_time) * 1000
                diagnostics["total_duration_ms"] = round(total_ms, 2)
                diagnostics["response_chars"] = len(final_text)
                diagnostics["success"] = True
                _mark_model_used(model)
                concurrent_on_finish = _dec_active_requests()
                diagnostics["concurrent_request_count_finish"] = concurrent_on_finish
                logger.info(
                    "OLLAMA_REQUEST_FINISHED | request_id=%s | component=%s | model=%s | success=true | total_duration_ms=%.2f | response_chars=%s | concurrent=%s | retries_attempted=%s",
                    request_id,
                    component,
                    model,
                    total_ms,
                    len(final_text),
                    concurrent_on_finish,
                    attempt,
                )
                return OllamaResult(text=final_text, diagnostics=dict(diagnostics))

            except requests.Timeout as exc:
                reason = "first_byte_timeout" if first_byte_time is None else "generation_timeout"
                diagnostics["fallback_reason"] = reason
                last_error = exc
                logger.warning(
                    "OLLAMA_RETRY | request_id=%s | component=%s | model=%s | attempt=%s | reason=%s",
                    request_id,
                    component,
                    model,
                    attempt,
                    reason,
                )
            except requests.ConnectionError as exc:
                diagnostics["fallback_reason"] = "connection_error"
                last_error = exc
                logger.warning(
                    "OLLAMA_RETRY | request_id=%s | component=%s | model=%s | attempt=%s | reason=connection_error",
                    request_id,
                    component,
                    model,
                    attempt,
                )
            except Exception as exc:
                diagnostics["fallback_reason"] = str(exc)
                last_error = exc
                logger.warning(
                    "OLLAMA_RETRY | request_id=%s | component=%s | model=%s | attempt=%s | reason=%s",
                    request_id,
                    component,
                    model,
                    attempt,
                    diagnostics["fallback_reason"],
                )
            finally:
                total_ms = (time.perf_counter() - submit_time) * 1000
                diagnostics["total_duration_ms"] = round(total_ms, 2)
                concurrent_on_finish = _dec_active_requests()
                diagnostics["concurrent_request_count_finish"] = concurrent_on_finish
                logger.info(
                    "OLLAMA_REQUEST_FINISHED | request_id=%s | component=%s | model=%s | success=%s | total_duration_ms=%.2f | response_chars=%s | concurrent=%s | retries_attempted=%s | fallback_reason=%s",
                    request_id,
                    component,
                    model,
                    diagnostics["success"],
                    total_ms,
                    diagnostics["response_chars"],
                    concurrent_on_finish,
                    attempt,
                    diagnostics.get("fallback_reason", ""),
                )

            if attempt >= retries:
                break
            if warmup_before_retry:
                self._health_ping(model=model)

        raise OllamaGenerationError(str(last_error or "ollama_generation_failed"), diagnostics)

    def warmup_model(self, model: str, timeout_seconds: int = 30) -> dict[str, Any]:
        """Lightweight warmup request for model loading."""

        try:
            result = self.generate(
                component="warmup",
                model=model,
                prompt="ping",
                connect_timeout_seconds=5,
                first_byte_timeout_seconds=min(timeout_seconds, 20),
                total_timeout_seconds=timeout_seconds,
                max_retries=0,
                options={"temperature": 0.0, "num_predict": 1},
            )
            logger.info("OLLAMA_WARMUP_DONE | model=%s | request_id=%s", model, result.diagnostics.get("request_id"))
            return result.diagnostics
        except OllamaGenerationError as exc:
            logger.warning("OLLAMA_WARMUP_FAILED | model=%s | reason=%s", model, exc)
            return exc.diagnostics

    def _health_ping(self, model: str) -> None:
        """Lightweight ping before retry to reduce cold-start retry flakiness."""

        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 1},
                },
                timeout=(3, 10),
            )
            logger.info("OLLAMA_HEALTH_PING | model=%s | status=ok", model)
        except Exception as exc:
            logger.warning("OLLAMA_HEALTH_PING | model=%s | status=failed | reason=%s", model, exc)


def _parse_json_line(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _estimate_model_state(model: str) -> str:
    with _COUNTER_LOCK:
        last_used = _LAST_USED_BY_MODEL.get(model)
    if last_used is None:
        return "unknown"
    if (time.time() - last_used) <= 180:
        return "warm"
    return "cold"


def _estimate_state_from_first_byte(model: str, first_byte_ms: float) -> str:
    prior = _estimate_model_state(model)
    if prior == "warm":
        return "warm"
    if first_byte_ms > 8000:
        return "cold"
    if first_byte_ms < 2500:
        return "warm"
    return "unknown"


def _mark_model_used(model: str) -> None:
    with _COUNTER_LOCK:
        _LAST_USED_BY_MODEL[model] = time.time()


def _inc_active_requests() -> int:
    global _ACTIVE_REQUESTS
    with _COUNTER_LOCK:
        _ACTIVE_REQUESTS += 1
        return _ACTIVE_REQUESTS


def _dec_active_requests() -> int:
    global _ACTIVE_REQUESTS
    with _COUNTER_LOCK:
        _ACTIVE_REQUESTS = max(0, _ACTIVE_REQUESTS - 1)
        return _ACTIVE_REQUESTS


def _set_socket_timeout(response: requests.Response, seconds: float) -> None:
    """Best-effort socket timeout update for post-first-byte streaming reads."""

    try:
        raw = response.raw
        if hasattr(raw, "_fp") and hasattr(raw._fp, "fp") and hasattr(raw._fp.fp, "raw"):
            sock = getattr(raw._fp.fp.raw, "_sock", None)
            if sock is not None:
                sock.settimeout(seconds)
    except Exception:
        # Non-fatal: keep existing timeout behavior when socket timeout override is unavailable.
        pass

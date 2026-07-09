"""LLM client abstraction — xAI Grok via OpenAI-compatible chat/completions API."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import requests

from app.config.models import LlmConfig
from app.core.exceptions import LlmClientError
from app.llm.models import LlmRequest, LlmResponse

logger = logging.getLogger(__name__)


ALERTADOR_PROMPT = """Eres un alertador inteligente de inventario para estanterias.
Tu tarea es interpretar eventos generados por vision artificial.
No inventes productos, zonas ni cantidades.
Usa unicamente el JSON recibido.
Prioriza alertas criticas: ausencia, inventario bajo, objeto mal ubicado y exceso.
Responde en espanol, claro, breve y accionable.
Incluye: estado general, alertas principales y recomendacion operativa."""


@dataclass
class LlmClient:
    """Client responsible only for consuming JSON summaries via xAI Grok API."""

    config: LlmConfig

    def analyze(self, request: LlmRequest) -> LlmResponse:
        """Analyze a JSON-only system summary using the configured provider."""

        if not self.config.enabled:
            return self._placeholder_response(request, "disabled")
        if self.config.provider.lower() not in ("grok", "groq"):
            return self._placeholder_response(request, "unsupported-provider")

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            logger.warning("LLM API key environment variable is not set: %s", self.config.api_key_env)
            return self._placeholder_response(request, "missing-api-key")

        try:
            return self._call_grok(request, api_key)
        except LlmClientError as exc:
            logger.exception("LLM diagnosis failed")
            return LlmResponse(
                analysis=f"No se pudo obtener diagnostico de Grok: {exc}",
                raw={"provider": self.config.provider, "error": str(exc)},
            )

    def _call_grok(self, request: LlmRequest, api_key: str) -> LlmResponse:
        """Call xAI via the OpenAI-compatible /v1/chat/completions endpoint."""

        prompt = request.prompt or ALERTADOR_PROMPT

        # xAI uses the standard OpenAI chat/completions format:
        # - endpoint: /v1/chat/completions
        # - field:    "messages"  (NOT "input")
        # - response: choices[0].message.content
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(request.summary, ensure_ascii=False, indent=2),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.config.base_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            # Log the response body on error to aid debugging
            if not response.ok:
                logger.error(
                    "Grok API error %s: %s",
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            raw = response.json()
        except requests.RequestException as exc:
            raise LlmClientError(f"Unable to call Grok provider: {exc}") from exc

        analysis = self._extract_chat_text(raw)
        return LlmResponse(
            analysis=analysis,
            raw={"provider": self.config.provider, "model": self.config.model},
        )

    def _placeholder_response(self, request: LlmRequest, reason: str) -> LlmResponse:
        """Return a deterministic local analysis when no real provider is enabled."""

        alert_count = len(request.summary.get("alerts", []))
        detection_count = len(request.summary.get("detections", []))
        event_count = len(request.summary.get("events", []))
        analysis = (
            f"Analisis local ({reason}): se recibieron {detection_count} detecciones, "
            f"{event_count} eventos y {alert_count} alertas. Active Grok en settings.json "
            f"y configure {self.config.api_key_env} para obtener diagnostico inteligente."
        )
        return LlmResponse(analysis=analysis, raw={"provider": self.config.provider, "reason": reason})

    @staticmethod
    def _extract_chat_text(raw: dict[str, object]) -> str:
        """Extract the assistant message from an OpenAI-compatible chat/completions response.

        Response structure:
            {
              "choices": [
                { "message": { "role": "assistant", "content": "<text>" } }
              ]
            }
        """
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        logger.warning("Unexpected Grok response structure: %s", str(raw)[:300])
        return "Grok respondio pero no se pudo extraer el texto del diagnostico."

"""LLM client abstraction for Groq via OpenAI-compatible chat/completions API."""

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


VIDEO_ALERTADOR_PROMPT = """Eres un auditor inteligente de inventario para videos de estanterias.
Tu tarea es interpretar una linea de tiempo generada por vision artificial.
No inventes productos, zonas, cantidades ni momentos.
Usa unicamente el JSON recibido.
Explica la evolucion del video: estado inicial, cambios relevantes por zona, alertas persistentes o nuevas, y estado final.
Si los productos desaparecen al final, menciona que hubo existencias previas y describe como se redujeron segun la linea de tiempo.
Prioriza ausencia, inventario bajo, objetos mal ubicados y exceso.
Responde en espanol, claro, breve y accionable."""


AUDIT_ALERTADOR_PROMPT = """Eres un auditor inteligente de inventario para Retail Monitor.
Recibiras un JSON historico guardado en una sesion del dashboard.
No inventes productos, zonas, cantidades, fechas ni eventos.
Usa unicamente el JSON recibido.
Resume el resultado del analisis, identifica alertas relevantes, explica el estado de inventario y entrega recomendaciones operativas.
Si el JSON corresponde a un video con linea de tiempo, explica la evolucion cronologica.
Responde en espanol, claro, breve y accionable."""


@dataclass
class LlmClient:
    """Client responsible only for consuming JSON summaries via Groq API."""

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
                analysis=f"No se pudo obtener diagnostico de Groq: {exc}",
                raw={"provider": self.config.provider, "error": str(exc)},
            )

    def _call_grok(self, request: LlmRequest, api_key: str) -> LlmResponse:
        """Call Groq via the OpenAI-compatible /v1/chat/completions endpoint."""

        prompt = request.prompt or ALERTADOR_PROMPT

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
                    "Groq API error %s: %s",
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            raw = response.json()
        except requests.RequestException as exc:
            raise LlmClientError(f"Unable to call Groq provider: {exc}") from exc

        analysis = self._extract_chat_text(raw)
        return LlmResponse(
            analysis=analysis,
            raw={"provider": self.config.provider, "model": self.config.model},
        )

    def _placeholder_response(self, request: LlmRequest, reason: str) -> LlmResponse:
        """Return a deterministic local analysis when no real provider is enabled."""

        if request.summary.get("summary_type") == "video_history":
            keyframe_count = len(request.summary.get("timeline_keyframes", []))
            change_count = len(request.summary.get("changes", []))
            event_counts = request.summary.get("event_counts", {})
            alert_counts = request.summary.get("alert_counts", {})
            analysis = (
                f"Analisis local de video ({reason}): se analizaron "
                f"{request.summary.get('frames_analyzed', 0)} frames, se detectaron "
                f"{keyframe_count} estados relevantes y {change_count} cambios de inventario. "
                f"Eventos por tipo: {event_counts}. Alertas por tipo: {alert_counts}. "
                f"Active Groq en settings.json y configure {self.config.api_key_env} "
                f"para obtener una narracion historica del video."
            )
            return LlmResponse(analysis=analysis, raw={"provider": self.config.provider, "reason": reason})

        alert_count = len(request.summary.get("alerts", []))
        detection_count = len(request.summary.get("detections", []))
        event_count = len(request.summary.get("events", []))
        analysis = (
            f"Analisis local ({reason}): se recibieron {detection_count} detecciones, "
            f"{event_count} eventos y {alert_count} alertas. Active Groq en settings.json "
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

        logger.warning("Unexpected Groq response structure: %s", str(raw)[:300])
        return "Groq respondio pero no se pudo extraer el texto del diagnostico."

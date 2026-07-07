"""LLM client abstraction."""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.models import LlmRequest, LlmResponse


@dataclass
class LlmClient:
    """Client responsible only for consuming JSON summaries."""

    provider_name: str

    def analyze(self, request: LlmRequest) -> LlmResponse:
        """Return a placeholder analysis for the current system summary."""

        alert_count = len(request.summary.get("alerts", []))
        detection_count = len(request.summary.get("detections", []))
        analysis = (
            f"Provider {self.provider_name} received {detection_count} detections and "
            f"{alert_count} alerts. Further diagnosis can be attached by the real provider."
        )
        return LlmResponse(analysis=analysis, raw={"provider": self.provider_name})


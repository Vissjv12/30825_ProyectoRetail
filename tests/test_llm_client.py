from __future__ import annotations

from app.config.models import LlmConfig
from app.llm.client import LlmClient
from app.llm.models import LlmRequest


def test_llm_client_uses_json_only_summary() -> None:
    response = LlmClient(LlmConfig(provider="grok", enabled=False)).analyze(
        LlmRequest(summary={"detections": [1], "events": [1], "alerts": [1, 2]})
    )
    assert "2 alerts" in response.analysis
    assert "1 detections" in response.analysis


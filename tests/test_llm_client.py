from __future__ import annotations

from app.llm.client import LlmClient
from app.llm.models import LlmRequest


def test_llm_client_uses_json_only_summary() -> None:
    response = LlmClient("openai").analyze(LlmRequest(summary={"detections": [1], "alerts": [1, 2]}))
    assert "2 alerts" in response.analysis
    assert "1 detections" in response.analysis


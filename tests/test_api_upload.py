from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import create_app


def test_analyze_image_endpoint_rejects_non_image() -> None:
    client = TestClient(create_app())
    response = client.post("/analyze-image", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_analyze_image_endpoint_accepts_image(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())

    import app.api as api_module

    monkeypatch.setattr(api_module.MonitoringPipeline, "run", lambda self, frame: {"frame_id": frame.frame_id, "source": frame.source})

    image = Image.new("RGB", (4, 4), color="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/analyze-image",
        files={"file": ("sample.jpeg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["frame_id"] == "sample.jpeg"


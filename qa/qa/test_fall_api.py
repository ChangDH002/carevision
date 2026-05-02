"""
낙상감지 API 최소 QA 테스트
- 핵심 앱 코드는 수정하지 않고, 테스트에서 무거운 AI 의존성을 mock 처리한다.
- 목표: /health, /detect/reset, /detect/fall 기본 응답 구조를 실제로 검증.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
sys.path.insert(0, str(AI_DIR))


def _install_test_doubles() -> None:
    """ai.api.routes import 시 필요한 무거운 모듈/파이프라인을 가짜 모듈로 주입."""

    # pipelines.fall_detector
    fall_mod = types.ModuleType("pipelines.fall_detector")

    class _FakeFallDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, image: str, camera_id: str = "default") -> dict:
            self.calls += 1
            return {
                "detected": False,
                "confidence": 0.0,
                "type": "FALL",
                "status": "normal",
                "method": "mock",
                "cameraId": camera_id,
            }

        def reset(self, camera_id: str = "default"):
            return None

    fall_mod.fall_detector = _FakeFallDetector()

    # pipelines.medication_detector
    med_mod = types.ModuleType("pipelines.medication_detector")

    class _FakeMedicationDetector:
        model_type = "mock"

        def detect_stream(self, image: str, camera_id: str = "default") -> dict:
            return {"status": "idle", "confidence": 0.0, "detected": False}

    med_mod.medication_detector = _FakeMedicationDetector()

    # pipelines.hand_to_mouth_detector
    h2m_mod = types.ModuleType("pipelines.hand_to_mouth_detector")
    h2m_mod.hand_to_mouth_detector = object()

    # pipelines.medication_scorer
    scorer_mod = types.ModuleType("pipelines.medication_scorer")

    class _FakeScorer:
        MOTION_POS_THRESHOLD = 0.5

        def update(self, image: str, camera_id: str):
            return {
                "medication_objects": [],
                "signals": {"hand_to_mouth_prob": 0.0, "hand_status": "idle"},
                "taken": False,
                "score": 0.0,
                "status": "idle",
                "window": 0,
            }

        def reset(self, camera_id: str):
            return None

    scorer_mod.init_scorer = lambda *_args, **_kwargs: _FakeScorer()

    # detector.detect_from_base64
    detector_mod = types.ModuleType("detector")

    class _FakeDetector:
        def detect_from_base64(self, image: str):
            return []

    detector_mod.detector = _FakeDetector()

    # streaming.mjpeg (cv2 의존 제거)
    mjpeg_mod = types.ModuleType("streaming.mjpeg")
    from fastapi import APIRouter as _APIRouter
    mjpeg_mod.stream_router = _APIRouter()
    # services.backend_client (httpx 제거 목적)
    backend_client_mod = types.ModuleType("services.backend_client")

    async def _noop_notify_detection(**_kwargs):
        return None

    async def _noop_notify_medication_log(**_kwargs):
        return None

    backend_client_mod.notify_detection = _noop_notify_detection
    backend_client_mod.notify_medication_log = _noop_notify_medication_log

    sys.modules["pipelines.fall_detector"] = fall_mod
    sys.modules["pipelines.medication_detector"] = med_mod
    sys.modules["pipelines.hand_to_mouth_detector"] = h2m_mod
    sys.modules["pipelines.medication_scorer"] = scorer_mod
    sys.modules["detector"] = detector_mod
    sys.modules["services.backend_client"] = backend_client_mod
    sys.modules["streaming.mjpeg"] = mjpeg_mod


@pytest.fixture(scope="session")
def client():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("starlette")

    _install_test_doubles()

    main = importlib.import_module("main")
    TestClient = importlib.import_module("fastapi.testclient").TestClient
    return TestClient(main.app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_detect_reset_endpoint(client):
    r = client.post("/detect/reset", json={"cameraId": "qa-cam"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("success") is True
    assert body.get("cameraId") == "qa-cam"


def test_detect_fall_response_shape(client):
    payload = {"image": "ZmFrZV9iYXNlNjQ=", "cameraId": "qa-cam"}
    r = client.post("/detect/fall", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("type") == "FALL"
    assert "status" in body
    assert "confidence" in body
    assert "method" in body


def test_detect_fall_invalid_payload_422(client):
    r = client.post("/detect/fall", json={})
    assert r.status_code == 422

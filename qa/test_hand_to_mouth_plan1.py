"""
1안 손→입 행동 감지 QA 테스트.

대상: pipelines.hand_to_mouth_fusion.fuse_hand_to_mouth (순수 함수).
이 모듈은 cv2/mediapipe/torch 를 import 하지 않으므로 무거운 의존성
없이 fusion 규칙을 단위 테스트할 수 있다.

핵심 불변식
- 어떤 입력도 taken=True 를 만들지 않는다 (손→입은 행동 후보 신호일 뿐).
- 손→입 fusion 은 medication_score / 복약 객체 감지 로직을 import 하거나
  변경하지 않는다 (독립).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
sys.path.insert(0, str(AI_DIR))

from pipelines.hand_to_mouth_fusion import (  # noqa: E402
    fuse_hand_to_mouth,
    LSTM_POSITIVE_THRESHOLD,
    LSTM_MIN_CONSECUTIVE,
    RULE_DISTANCE_THRESHOLD,
    RULE_MIN_CONSECUTIVE,
)


def _base(**overrides):
    """기본은 '손/입 있음, motion 인정, 아무 신호 없음'."""
    kw = dict(
        lstm_confidence=0.0,
        lstm_consecutive=0,
        rule_distance=None,
        rule_consecutive=0,
        has_hand=True,
        has_mouth=True,
        motion_accepted=True,
    )
    kw.update(overrides)
    return fuse_hand_to_mouth(**kw)


# ── 1. LSTM confidence 높음 → lstm_detected, final_detected ──
def test_case1_lstm_high_detected():
    r = _base(
        lstm_confidence=LSTM_POSITIVE_THRESHOLD + 0.2,
        lstm_consecutive=LSTM_MIN_CONSECUTIVE,
    )
    assert r["lstm_detected"] is True
    assert r["final_detected"] is True
    assert r["detection_source"] == "lstm"
    assert r["status"] == "hand_to_mouth_detected"
    assert r["taken"] is False


# ── 2. LSTM 낮지만 거리 rule 충족 → rule_detected, final_detected ──
def test_case2_rule_only_detected():
    r = _base(
        lstm_confidence=0.1,
        lstm_consecutive=0,
        rule_distance=RULE_DISTANCE_THRESHOLD - 0.05,
        rule_consecutive=RULE_MIN_CONSECUTIVE,
    )
    assert r["lstm_detected"] is False
    assert r["rule_detected"] is True
    assert r["final_detected"] is True
    assert r["detection_source"] == "rule"
    assert r["taken"] is False


# ── 3. LSTM + rule 둘 다 → both ──
def test_case3_both_sources():
    r = _base(
        lstm_confidence=LSTM_POSITIVE_THRESHOLD + 0.1,
        lstm_consecutive=LSTM_MIN_CONSECUTIVE,
        rule_distance=RULE_DISTANCE_THRESHOLD - 0.1,
        rule_consecutive=RULE_MIN_CONSECUTIVE + 2,
    )
    assert r["lstm_detected"] is True
    assert r["rule_detected"] is True
    assert r["detection_source"] == "both"
    assert r["final_detected"] is True
    assert r["taken"] is False


# ── 4. 둘 다 미충족 → final_detected=false, none ──
def test_case4_none():
    r = _base(lstm_confidence=0.1, rule_distance=0.9, rule_consecutive=1)
    assert r["final_detected"] is False
    assert r["detection_source"] == "none"
    assert r["status"] == "no_hand_to_mouth"
    assert r["taken"] is False


# ── 5. landmark 부족 → insufficient_landmarks ──
def test_case5_insufficient_landmarks():
    r = _base(
        has_hand=False,
        lstm_confidence=0.99,
        lstm_consecutive=10,  # LSTM 이 강해도 손 없으면 무효
    )
    assert r["status"] == "insufficient_landmarks"
    assert r["final_detected"] is False
    assert r["taken"] is False


# ── 6. motion 부족 → motion_not_accepted ──
def test_case6_motion_not_accepted():
    r = _base(
        lstm_confidence=LSTM_POSITIVE_THRESHOLD + 0.3,
        lstm_consecutive=LSTM_MIN_CONSECUTIVE,
        motion_accepted=False,
    )
    assert r["status"] == "motion_not_accepted"
    assert r["final_detected"] is False
    # 신호 자체는 떴지만 motion 게이팅으로 최종 억제
    assert r["lstm_detected"] is True
    assert r["taken"] is False


# ── 7. 어떤 경우에도 taken=true 를 만들지 않음 ──
@pytest.mark.parametrize("lstm_conf", [0.0, 0.49, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("rule_dist", [None, 0.0, 0.45, 0.9])
@pytest.mark.parametrize("has_hand", [True, False])
@pytest.mark.parametrize("motion", [True, False])
def test_case7_never_taken(lstm_conf, rule_dist, has_hand, motion):
    r = _base(
        lstm_confidence=lstm_conf,
        lstm_consecutive=5,
        rule_distance=rule_dist,
        rule_consecutive=5,
        has_hand=has_hand,
        motion_accepted=motion,
    )
    assert r["taken"] is False
    assert "score" not in r          # 복약 점수/완료 개념을 만들지 않음


# ── 8. 복약 객체/스코어 로직과 독립 (import 격리) ──
def test_case8_no_medication_coupling():
    import pipelines.hand_to_mouth_fusion as fusion_mod

    src = Path(fusion_mod.__file__).read_text(encoding="utf-8")
    # 산문(docstring)이 아니라 실제 import 문 기준으로 검사한다.
    import_lines = [
        ln.strip()
        for ln in src.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    joined = " ".join(import_lines)
    # fusion 모듈은 복약 객체/스코어 로직과 무거운 의존성을 import 하지 않는다.
    for forbidden in (
        "medication_scorer",
        "medication_detector",
        "registered",
        "cv2",
        "torch",
        "mediapipe",
        "ultralytics",
    ):
        assert forbidden not in joined, f"fusion 모듈이 {forbidden} 를 import 함"


# ── 경계: consecutive 부족 시 미감지 ──
def test_lstm_consecutive_guard():
    r = _base(
        lstm_confidence=LSTM_POSITIVE_THRESHOLD + 0.4,
        lstm_consecutive=LSTM_MIN_CONSECUTIVE - 1,
    )
    assert r["lstm_detected"] is False


def test_rule_consecutive_guard():
    r = _base(
        rule_distance=RULE_DISTANCE_THRESHOLD - 0.1,
        rule_consecutive=RULE_MIN_CONSECUTIVE - 1,
    )
    assert r["rule_detected"] is False


# ════════════════════════════════════════════════
# 보수적 수정: 손이 입에 매우 가까울 때만 motion gate 우회
# ════════════════════════════════════════════════
from pipelines.hand_to_mouth_fusion import RULE_NEAR_DISTANCE  # noqa: E402


def test_very_close_bypasses_motion_gate():
    """손끝-입 매우 가까움 + rule 충족 + motion 부족 → 우회로 final=true."""
    r = _base(
        lstm_confidence=0.1,
        rule_distance=RULE_NEAR_DISTANCE - 0.02,   # 매우 가까움
        rule_consecutive=RULE_MIN_CONSECUTIVE,
        motion_accepted=False,                      # 정적(입에 머무름)
    )
    assert r["rule_detected"] is True
    assert r["rule_very_close"] is True
    assert r["motion_bypassed"] is True
    assert r["final_detected"] is True
    assert r["status"] == "hand_to_mouth_detected"
    assert r["taken"] is False


def test_moderate_distance_still_motion_gated():
    """가깝지만(<=thr) '매우 가깝'지는 않음 + motion 부족 → 여전히 차단.
    (우회가 무분별하게 늘지 않음을 보장 — 오탐 방지)"""
    mid = (RULE_NEAR_DISTANCE + RULE_DISTANCE_THRESHOLD) / 2
    r = _base(
        lstm_confidence=0.1,
        rule_distance=mid,
        rule_consecutive=RULE_MIN_CONSECUTIVE,
        motion_accepted=False,
    )
    assert r["rule_detected"] is True
    assert r["rule_very_close"] is False
    assert r["final_detected"] is False
    assert r["status"] == "motion_not_accepted"
    assert r["block_reason"] == "motion_gate"


def test_pending_near_mouth_when_consecutive_insufficient():
    """입에 매우 근접하지만 rule 연속 미충족 → pending_near_mouth, final=false."""
    r = _base(
        lstm_confidence=0.1,
        rule_distance=RULE_NEAR_DISTANCE - 0.05,
        rule_consecutive=RULE_MIN_CONSECUTIVE - 1,   # 연속 부족
        motion_accepted=True,
    )
    assert r["rule_detected"] is False
    assert r["final_detected"] is False
    assert r["status"] == "pending_near_mouth"
    assert r["block_reason"] in ("rule_consecutive_pending", "near_but_no_signal")
    assert r["taken"] is False


def test_far_distance_no_bypass_no_fp():
    """손이 입에서 멈(거리 큼) + motion 부족 → 우회 없음, final=false (오탐 X)."""
    r = _base(
        lstm_confidence=0.1,
        rule_distance=0.9,
        rule_consecutive=10,
        motion_accepted=False,
    )
    assert r["rule_very_close"] is False
    assert r["motion_bypassed"] is False
    assert r["final_detected"] is False


def test_still_no_person_insufficient_landmarks():
    """가만히/사람 없음(손 landmark 없음) → insufficient_landmarks, false."""
    r = _base(has_hand=False, has_mouth=False, rule_distance=None)
    assert r["status"] == "insufficient_landmarks"
    assert r["final_detected"] is False
    assert r["block_reason"] == "insufficient_landmarks"


def test_hand_moves_but_not_near_mouth_false():
    """손은 움직이지만(motion O) 입 근처 아님 + LSTM 약함 → false."""
    r = _base(
        lstm_confidence=0.2,
        rule_distance=0.8,
        rule_consecutive=0,
        motion_accepted=True,
    )
    assert r["final_detected"] is False
    assert r["status"] == "no_hand_to_mouth"


def test_holding_object_away_from_mouth_false():
    """물체 들고 있으나 입에서 멈 → false (rule far, lstm 약)."""
    r = _base(lstm_confidence=0.3, rule_distance=0.6, motion_accepted=True)
    assert r["final_detected"] is False
    assert r["taken"] is False


@pytest.mark.parametrize("rd", [None, 0.05, 0.18, 0.45, 0.9])
@pytest.mark.parametrize("mot", [True, False])
def test_bypass_never_makes_taken(rd, mot):
    r = _base(
        lstm_confidence=0.9, lstm_consecutive=5,
        rule_distance=rd, rule_consecutive=5, motion_accepted=mot,
    )
    assert r["taken"] is False
    assert "score" not in r

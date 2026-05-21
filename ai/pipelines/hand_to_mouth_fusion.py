"""
손→입 행동 감지 fusion 로직 (1안) — 의존성 없는 순수 모듈

cv2 / mediapipe / torch 를 import 하지 않는다. 따라서 QA 가 무거운
의존성 없이 fusion 규칙만 단위 테스트할 수 있다.

목적: 복약으로 볼 수 있는 핵심 행동 "손이 입 근처로 이동하는 동작" 의
후보 신호를 LSTM + 거리 rule 로 결합해 판정한다.
※ 복약 "완료" 를 확정하지 않는다. taken 을 만들지 않는다.
"""

from __future__ import annotations

import os
from typing import Optional

# ── 튜닝 상수 (env override 가능) ────────────────
LSTM_POSITIVE_THRESHOLD = float(os.getenv("H2M_LSTM_THRESH", "0.5"))
LSTM_MIN_CONSECUTIVE = int(os.getenv("H2M_LSTM_CONSEC", "2"))
RULE_DISTANCE_THRESHOLD = float(os.getenv("H2M_RULE_DIST", "0.45"))
RULE_MIN_CONSECUTIVE = int(os.getenv("H2M_RULE_CONSEC", "3"))
MOTION_WINDOW = int(os.getenv("H2M_MOTION_WINDOW", "8"))
MOTION_MIN = float(os.getenv("H2M_MOTION_MIN", "0.004"))
# 손끝이 입에 "매우 가까운" 거리 — 이 경우는 손이 사실상 입에 닿아 있는
# 강한 근거이므로(복약/음수 시 손은 입에서 거의 정지), motion gate 를
# 우회한다. 거리가 멀면(이 값 초과) gate 그대로 → 오탐 증가 없음.
RULE_NEAR_DISTANCE = float(os.getenv("H2M_RULE_NEAR_DIST", "0.18"))


def fuse_hand_to_mouth(
    *,
    lstm_confidence: float,
    lstm_consecutive: int,
    rule_distance: Optional[float],
    rule_consecutive: int,
    has_hand: bool,
    has_mouth: bool,
    motion_accepted: bool,
    lstm_threshold: float = LSTM_POSITIVE_THRESHOLD,
    lstm_min_consecutive: int = LSTM_MIN_CONSECUTIVE,
    rule_threshold: float = RULE_DISTANCE_THRESHOLD,
    rule_min_consecutive: int = RULE_MIN_CONSECUTIVE,
    rule_near_distance: float = RULE_NEAR_DISTANCE,
) -> dict:
    """
    LSTM 신호 + 거리 rule 신호 결합 → 손→입 행동 후보 판정.

    판정 규칙
    1. LSTM confidence ≥ threshold 이고 연속 양성이면 lstm_detected
    2. 손끝-입 거리 ≤ threshold 가 연속 N프레임 유지되면 rule_detected
    3. final_detected = lstm_detected OR rule_detected
    4. detection_source: 둘 다→both / LSTM만→lstm / rule만→rule / 없음→none
    5. 손 landmark 부족 → status=insufficient_landmarks, final_detected=false
    6. motion 부족 → status=motion_not_accepted, final_detected=false
       단, 손끝이 입에 매우 가까우면(rule_distance ≤ rule_near_distance)
       motion gate 를 우회한다. 손이 입에 닿아 정지하는 순간은 복약/음수의
       핵심 장면이므로 정적이라고 억제하면 false negative 가 된다.
       거리가 멀면 gate 그대로이므로 오탐은 늘지 않는다.
    7. 거리는 매우 가깝지만 rule 연속/LSTM 미충족이면 pending_near_mouth
       (final_detected=false, 정보용 — 확정 아님)

    block_reason: final_detected=false 일 때 막힌 사유 디버그 표시.

    ※ 어떤 경우에도 taken / 복약 완료를 만들지 않는다 (후보 신호 전용).
    """
    lstm_detected = (
        lstm_confidence >= lstm_threshold
        and lstm_consecutive >= lstm_min_consecutive
    )
    rule_detected = (
        rule_distance is not None
        and rule_distance <= rule_threshold
        and rule_consecutive >= rule_min_consecutive
    )

    if lstm_detected and rule_detected:
        source = "both"
    elif lstm_detected:
        source = "lstm"
    elif rule_detected:
        source = "rule"
    else:
        source = "none"

    raw_detected = lstm_detected or rule_detected

    # 손끝이 입에 "매우 가까운" 강한 근거 → motion gate 우회 조건
    rule_very_close = (
        rule_distance is not None and rule_distance <= rule_near_distance
    )
    motion_ok = motion_accepted or rule_very_close

    block_reason = None

    # 규칙 5,6,7 우선 (landmark/motion 게이팅이 raw 판정보다 우선)
    if not has_hand:
        status = "insufficient_landmarks"
        final_detected = False
        block_reason = "insufficient_landmarks"
        reason = "손 랜드마크가 부족해 손→입 행동을 판정할 수 없습니다."
    elif raw_detected and not motion_ok:
        status = "motion_not_accepted"
        final_detected = False
        block_reason = "motion_gate"
        reason = "손이 거의 움직이지 않아 손→입 동작으로 인정하지 않습니다."
    elif raw_detected:
        status = "hand_to_mouth_detected"
        final_detected = True
        reason = "손이 입 근처로 이동하는 동작이 감지되었습니다. (복약 완료 확정 아님)"
    elif rule_very_close:
        # 손은 입에 매우 가깝지만 rule 연속/LSTM 이 아직 미충족 — 대기 표시
        status = "pending_near_mouth"
        final_detected = False
        block_reason = (
            "rule_consecutive_pending" if rule_distance is not None
            and rule_distance <= rule_threshold else "near_but_no_signal"
        )
        reason = "손이 입에 근접했으나 확정 조건(연속 프레임)이 아직 부족합니다. (확정 아님)"
    else:
        status = "no_hand_to_mouth"
        final_detected = False
        block_reason = "no_signal"
        reason = "손→입 행동이 감지되지 않았습니다."

    return {
        "final_detected": final_detected,
        "detection_source": source if raw_detected else "none",
        "lstm_detected": lstm_detected,
        "rule_detected": rule_detected,
        "rule_very_close": rule_very_close,
        "motion_bypassed": bool(rule_very_close and not motion_accepted),
        "block_reason": block_reason,
        "status": status,
        "reason": reason,
        # AI 는 후보 신호만 — 절대 복약 완료 확정값을 만들지 않는다.
        "taken": False,
    }

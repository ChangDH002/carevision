"""
손→입 동작 감지 파이프라인 (1안: 손→입 행동 감지)

이 모듈의 목적은 **복약으로 볼 수 있는 핵심 행동 = "손이 입 근처로
이동하는 동작"** 을 안정적으로 감지해 *후보 신호* 를 제공하는 것이다.
복약 "완료" 를 확정하지 않는다. taken=true 를 만들지 않는다.

구성
────
1) LSTM 경로 (기존, 구조 변경 없음)
   - MediaPipe Hands 21랜드마크 × (x, y) = 42차원
   - 2-layer LSTM(input=42, hidden=128) → FC(128→64)→ReLU→Dropout→FC(64→1)
   - 체크포인트: ai/models/hand_to_mouth_lstm.pt
2) 거리 rule 경로 (보조, 신규 추가)
   - MediaPipe FaceDetection 으로 입(mouth) 키포인트 추출
   - 손끝(엄지/검지/중지/약지/소지)–입 정규화 거리 계산
   - threshold 이하가 연속 N프레임 유지되면 rule_detected
3) motion 게이팅 (신규)
   - 손끝이 실제로 움직였는지(정적 오탐 억제)
4) fusion (신규, 순수 함수 fuse_hand_to_mouth — QA 단위테스트 대상)
   - final_detected = lstm_detected OR rule_detected
   - detection_source ∈ {lstm, rule, both, none}

⚠️ 기존 LSTM 모델 구조/체크포인트/42차원 feature 는 변경하지 않았다.
⚠️ detect() 의 기존 키(detected, confidence, status, type)는 유지하고
   새 필드는 additive 로만 추가한다 (medication_scorer 호환 보호:
   scorer 는 confidence=LSTM sigmoid 값을 그대로 사용한다).
"""

from __future__ import annotations

import base64
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# fusion 로직과 튜닝 상수는 의존성 없는 별도 모듈에 둔다 (QA 단위테스트용).
from pipelines.hand_to_mouth_fusion import (
    fuse_hand_to_mouth,
    LSTM_POSITIVE_THRESHOLD as POSITIVE_THRESHOLD,
    LSTM_MIN_CONSECUTIVE as CONSECUTIVE_POSITIVE,
    RULE_DISTANCE_THRESHOLD,
    RULE_MIN_CONSECUTIVE,
    MOTION_WINDOW,
    MOTION_MIN,
)

# ─────────────────────────────────────────────
# ASSUMPTIONS (LSTM — 기존값 유지, 변경 금지)
# ─────────────────────────────────────────────
WINDOW_SIZE = 15            # LSTM 시퀀스 길이
INPUT_DIM = 42              # 21 landmarks × (x, y)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3

# 손끝 랜드마크 인덱스 (엄지, 검지, 중지, 약지, 소지 tip)
FINGERTIPS = (4, 8, 12, 16, 20)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_to_mouth_lstm.pt"

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection


# ─────────────────────────────────────────────
# Model (구조 변경 금지)
# ─────────────────────────────────────────────
class HandToMouthLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),   # fc.0
            nn.ReLU(),                    # fc.1
            nn.Dropout(dropout),          # fc.2
            nn.Linear(64, 1),             # fc.3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])    # 마지막 타임스텝


# ─────────────────────────────────────────────
# Detector  (fusion 로직은 pipelines.hand_to_mouth_fusion 에 분리)
# ─────────────────────────────────────────────
class HandToMouthDetector:
    """
    실시간 손→입 동작 감지기 (LSTM + 거리 rule + motion 게이팅).
    복약 행동 후보 신호만 제공한다. 복약 완료를 확정하지 않는다.
    """

    def __init__(self, model_path: Path = MODEL_PATH, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = HandToMouthLSTM().to(self.device)

        if not model_path.exists():
            raise FileNotFoundError(f"hand_to_mouth_lstm.pt not found at {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # 카메라별 상태
        self.buffers: dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=WINDOW_SIZE)
        )
        self.positive_counters: dict[str, int] = defaultdict(int)      # LSTM 연속 양성
        self.rule_counters: dict[str, int] = defaultdict(int)          # rule 연속 충족
        self.fingertip_hist: dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=MOTION_WINDOW)
        )
        # routes.py 가 (이중 추론 없이) 풍부한 결과를 읽어가기 위한 캐시
        self.last_result: dict[str, dict] = {}

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # 입 키포인트용 얼굴 감지 (short-range), 거리 rule 전용
        self.face = mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    # ── image utils ────────────────────────────
    @staticmethod
    def _decode_image(image_base64: str) -> np.ndarray:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # ── feature extraction ────────────────────
    def _process_frame(self, image_bgr: np.ndarray):
        """
        한 프레임에서 LSTM 42-vec / 손끝 좌표 / 입 좌표를 함께 추출한다.

        Returns: (hand_vec(42,)|None, fingertips list[(x,y)], mouth(x,y)|None)
        - hand_vec 구성 순서는 기존과 동일 (i*2=x, i*2+1=y) — LSTM 영향 없음.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        hand_res = self.hands.process(image_rgb)
        hand_vec = None
        fingertips: list[tuple[float, float]] = []
        if hand_res.multi_hand_landmarks:
            lms = hand_res.multi_hand_landmarks[0].landmark  # 21개
            hand_vec = np.empty(42, dtype=np.float32)
            for i, lm in enumerate(lms):
                hand_vec[i * 2] = lm.x
                hand_vec[i * 2 + 1] = lm.y
            for idx in FINGERTIPS:
                fingertips.append((lms[idx].x, lms[idx].y))

        face_res = self.face.process(image_rgb)
        mouth = None
        if face_res.detections:
            kps = face_res.detections[0].location_data.relative_keypoints
            # MediaPipe FaceDetection 키포인트 index 3 = MOUTH_CENTER
            if len(kps) > 3:
                mouth = (kps[3].x, kps[3].y)

        return hand_vec, fingertips, mouth

    @staticmethod
    def _min_fingertip_mouth_distance(fingertips, mouth) -> Optional[float]:
        if not fingertips or mouth is None:
            return None
        mx, my = mouth
        dists = [((fx - mx) ** 2 + (fy - my) ** 2) ** 0.5 for fx, fy in fingertips]
        return float(min(dists))

    # ── public API ────────────────────────────
    def detect(self, image_base64: str, camera_id: str = "default") -> dict:
        image = self._decode_image(image_base64)
        if image is None:
            res = self._empty_response("decode_failed")
            self.last_result[camera_id] = res
            return res

        hand_vec, fingertips, mouth = self._process_frame(image)
        has_hand = hand_vec is not None
        has_mouth = mouth is not None

        # ── LSTM 버퍼 (기존 동작 유지: 손 없으면 0-padding) ──
        buf = self.buffers[camera_id]
        if has_hand:
            buf.append(hand_vec)
        else:
            buf.append(np.zeros(42, dtype=np.float32))

        # ── 거리 rule ──
        rule_distance = self._min_fingertip_mouth_distance(fingertips, mouth)
        if rule_distance is not None and rule_distance <= RULE_DISTANCE_THRESHOLD:
            self.rule_counters[camera_id] += 1
        else:
            self.rule_counters[camera_id] = 0
        rule_consecutive = self.rule_counters[camera_id]

        # ── motion 게이팅 (검지 tip 이동량) ──
        motion_score = 0.0
        motion_accepted = True
        if has_hand and fingertips:
            idx_tip = np.array(fingertips[1], dtype=np.float32)  # 검지 tip
            hist = self.fingertip_hist[camera_id]
            hist.append(idx_tip)
            if len(hist) >= 2:
                diffs = [
                    float(np.linalg.norm(hist[i] - hist[i - 1]))
                    for i in range(1, len(hist))
                ]
                motion_score = float(np.mean(diffs))
                motion_accepted = motion_score >= MOTION_MIN
            else:
                motion_accepted = True  # 표본 부족 — 조기 억제하지 않음

        # ── LSTM 추론 (버퍼 가득 찰 때만) ──
        if len(buf) < WINDOW_SIZE:
            res = {
                "detected": False,
                "final_detected": False,
                "confidence": 0.0,
                "lstm_confidence": 0.0,
                "rule_detected": False,
                "rule_distance": round(rule_distance, 4) if rule_distance is not None else None,
                "rule_threshold": RULE_DISTANCE_THRESHOLD,
                "consecutive_frames": rule_consecutive,
                "lstm_consecutive": self.positive_counters[camera_id],
                "motion_accepted": motion_accepted,
                "motion_score": round(motion_score, 5),
                "detection_source": "none",
                "rule_very_close": False,
                "motion_bypassed": False,
                "block_reason": "warming_up",
                "type": "HAND_TO_MOUTH",
                "status": f"warming_up ({len(buf)}/{WINDOW_SIZE})",
                "window_size": WINDOW_SIZE,
                "reason": "버퍼를 채우는 중입니다.",
            }
            self.last_result[camera_id] = res
            return res

        seq = np.stack(list(buf), axis=0)                       # (W, 42)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, W, 42)
        with torch.no_grad():
            logit = self.model(x)
            prob = float(torch.sigmoid(logit).item())

        if prob >= POSITIVE_THRESHOLD:
            self.positive_counters[camera_id] += 1
        else:
            self.positive_counters[camera_id] = 0
        lstm_consecutive = self.positive_counters[camera_id]

        # ── fusion (순수 함수) ──
        fused = fuse_hand_to_mouth(
            lstm_confidence=prob,
            lstm_consecutive=lstm_consecutive,
            rule_distance=rule_distance,
            rule_consecutive=rule_consecutive,
            has_hand=has_hand,
            has_mouth=has_mouth,
            motion_accepted=motion_accepted,
        )

        res = {
            # 기존 키 (호환 유지) — scorer 는 confidence=LSTM prob 사용
            "detected": fused["final_detected"],
            "confidence": round(prob, 4),
            "type": "HAND_TO_MOUTH",
            "status": fused["status"],
            # 신규 풍부 필드 (additive)
            "final_detected": fused["final_detected"],
            "detection_source": fused["detection_source"],
            "lstm_detected": fused["lstm_detected"],
            "lstm_confidence": round(prob, 4),
            "rule_detected": fused["rule_detected"],
            "rule_distance": round(rule_distance, 4) if rule_distance is not None else None,
            "rule_threshold": RULE_DISTANCE_THRESHOLD,
            "rule_very_close": fused["rule_very_close"],
            "motion_bypassed": fused["motion_bypassed"],
            "block_reason": fused["block_reason"],
            "consecutive_frames": rule_consecutive,
            "lstm_consecutive": lstm_consecutive,
            "motion_accepted": motion_accepted,
            "motion_score": round(motion_score, 5),
            "has_hand": has_hand,
            "has_mouth": has_mouth,
            "window_size": WINDOW_SIZE,
            "threshold": POSITIVE_THRESHOLD,
            "reason": fused["reason"],
        }
        self.last_result[camera_id] = res
        return res

    def reset(self, camera_id: str = "default"):
        self.buffers.pop(camera_id, None)
        self.positive_counters[camera_id] = 0
        self.rule_counters[camera_id] = 0
        self.fingertip_hist.pop(camera_id, None)
        self.last_result.pop(camera_id, None)

    @staticmethod
    def _empty_response(reason: str) -> dict:
        return {
            "detected": False,
            "final_detected": False,
            "confidence": 0.0,
            "lstm_confidence": 0.0,
            "rule_detected": False,
            "rule_distance": None,
            "rule_threshold": RULE_DISTANCE_THRESHOLD,
            "consecutive_frames": 0,
            "motion_accepted": False,
            "detection_source": "none",
            "rule_very_close": False,
            "motion_bypassed": False,
            "block_reason": reason,
            "type": "HAND_TO_MOUTH",
            "status": reason,
            "reason": reason,
        }


# 모듈 전역 인스턴스 (lazy init: 이 파일이 import 될 때만 생성)
try:
    hand_to_mouth_detector = HandToMouthDetector()
except FileNotFoundError as e:
    hand_to_mouth_detector = None
    print(f"[hand_to_mouth_detector] WARNING: {e}")

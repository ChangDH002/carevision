import os
import json
import time
import cv2
import mediapipe as mp
import base64
import numpy as np
from collections import defaultdict, deque

mp_pose = mp.solutions.pose

# 감지 민감도를 낮춰 엎드린/누운 자세도 최대한 잡게 설정
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24
SEQUENCE_LEN = 30
NUM_FEATURES = 132  # 33 keypoints × 4 (x, y, z, visibility)
FALL_CLASS = 0
FALL_THRESHOLD = 0.5  # legacy 호환 — 새 코드는 self.FALL_PROB_THRESHOLD 사용

# ── 환경변수 기본값 (lying suppression / fall threshold) ─────
# - 운영자가 환경변수로 조정할 수 있다. fall_detector.py 코드 변경 없이도 튜닝 가능.
# - default 는 외부 영상 평가 결과를 토대로 보수적으로 설정.
DEFAULT_FALL_PROB_THRESHOLD = 0.65       # softmax[Fall] 이 이 값 이상이어야 후보
DEFAULT_FALL_CONSECUTIVE_THRESHOLD = 3   # 연속 N 프레임 → emergency 후보
DEFAULT_FALL_CONFIRM_WINDOWS = 3         # 이 회수 미만은 movement_pending (관찰 단계)
DEFAULT_FALL_SLOW_MOTION_MAX = 0.020     # 이 미만 motion = 'slow' 로 간주 (천천히 눕기)
DEFAULT_FALL_DYNAMIC_MOTION_MIN = 0.020  # dynamic event gate 의 motion 최소치
DEFAULT_FALL_TORSO_CHANGE_MIN = 30.0     # dynamic event gate 의 torso angle 변화량(°) 최소치
# NEW: 낙상 사건 latch — 한 번 fall_emergency 가 확정되면 명시적 reset 전까지 상태 유지.
# Default 0 = 최초 1회만 알림 (재알림 없음). 양수 N 으로 설정하면 N 초마다 reminder 알림.
DEFAULT_FALL_ALERT_COOLDOWN_SECONDS = 0.0
DEFAULT_VERTICAL_DROP_MIN = 0.10         # hip_y normalized 변화량 (0~1) — 이 미만이면 lying
DEFAULT_MOTION_SCORE_MIN = 0.005         # frame-to-frame |Δhip_y| 평균 — 이 미만이면 정지
DEFAULT_KINEMATICS_HISTORY = 30          # 최근 frame 보관 수 (≈ 1초 @ 30fps)
KIN_MIN_SAMPLES_FOR_SUPPRESSION = 10     # 워밍업 — 이 이상 쌓여야 suppression 적용

# ── 휴리스틱 튜닝 상수 ─────────────────────────
# hip_drop(속도) 조건은 앉기 동작도 그대로 잡아서 제거했다.
# torso_horizontal 은 워밍업 동안만 사용하고, LSTM 활성 후엔
# head_below_hip(강한 신호)만 휴리스틱으로 OR 결합한다.
HORIZONTAL_TORSO_THRESHOLD = 0.10   # |shoulder_y - hip_y| 이하면 몸통 수평


def _read_env_float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = float(raw)
        if lo <= v <= hi:
            print(f"[FallDetector] env override {name} = {v} (default {default})")
            return v
        print(f"[FallDetector] env {name}={raw} 범위 외([{lo},{hi}]) — default {default}")
    except (TypeError, ValueError):
        print(f"[FallDetector] env {name}={raw!r} float 변환 실패 — default {default}")
    return default


def _read_env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        v = int(raw)
        if lo <= v <= hi:
            print(f"[FallDetector] env override {name} = {v} (default {default})")
            return v
        print(f"[FallDetector] env {name}={raw} 범위 외([{lo},{hi}]) — default {default}")
    except (TypeError, ValueError):
        print(f"[FallDetector] env {name}={raw!r} int 변환 실패 — default {default}")
    return default


# ── backend 별 클래스 라벨 매핑 ─────────────────────────
# get_model_info() 의 labels 와 detect() 의 model_prediction 둘 다 참조.
_VB_LABELS = {0: "Fall", 1: "Walking", 2: "Lying_Down", 3: "Sitting", 4: "Standing_Transition"}
_LEGACY_LABELS = {0: "Normal", 1: "Fall"}

# YOLO 폴백: bbox 가로/세로 비율이 이 값 이상이면 수평 자세(낙상 후보)로 판단
# 2.0 이상 = 완전히 누운 자세만 해당 (앉거나 기댄 자세 오탐 방지)
YOLO_HORIZONTAL_RATIO = 2.0

# 모델 파일 경로
# - 우선순위:
#   1) PyTorch vB (.pt, 5-class, fall=0)        — 권장: .venv311 단일 런타임 호환
#   2) Keras vB (.keras, 5-class, fall=0)       — QA 검증 모델 (TF 환경에서만 로드)
#   3) Legacy PyTorch (.pt, 2-class, fall=1)    — fallback (정확도 낮음)
#   4) Heuristic                                 — 모델 파일 모두 부재 시
# - 어떤 모델 파일도 없으면 휴리스틱만 사용하며 서버는 정상 동작한다.
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_PT_VB_MODEL_PATH = os.path.join(_MODELS_DIR, "fall_lstm_vB.pt")
_PT_VB_META_PATH = os.path.join(_MODELS_DIR, "fall_lstm_vB_meta.json")
_KERAS_MODEL_PATH = os.path.join(_MODELS_DIR, "fall_lstm_vB_best.keras")
_VB_LABELS_PATH = os.path.join(_MODELS_DIR, "labels_vB.json")
_PT_MODEL_PATH = os.path.join(_MODELS_DIR, "fall_lstm.pt")
_PT_META_PATH = os.path.join(_MODELS_DIR, "fall_lstm_meta.json")

# 호환을 위한 별칭 (외부에서 _MODEL_PATH를 참조하던 코드 보호)
_MODEL_PATH = _KERAS_MODEL_PATH

_lstm_model = None
_lstm_backend = None          # "keras" | "torch" | None
_lstm_meta: dict | None = None
_model_load_attempted = False  # 한 번 실패하면 재시도하지 않음 (로그 폭주 방지)

# 백엔드 진단용 — 마지막 로드 시도의 실패 사유 / 활성 경로 추적
_pt_vb_fail_reason: str | None = None
_keras_fail_reason: str | None = None
_pt_fail_reason: str | None = None
_active_model_path: str | None = None

# YOLOv8 기본 모델 (person 클래스 감지용 폴백)
_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8n.pt")
        _yolo_model = YOLO(yolo_path)
    except Exception as e:
        print(f"[FallDetector] YOLO 로드 실패: {e}")
    return _yolo_model


def _build_and_load_model():
    """모델 구조 직접 구성 후 weights만 로드 (Keras 버전 호환 우회)."""
    import zipfile, tempfile
    import keras

    inputs = keras.layers.Input(shape=(SEQUENCE_LEN, NUM_FEATURES), name="pose_seq")
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(5, activation="softmax", name="logits")(x)
    model = keras.Model(inputs, out)

    with zipfile.ZipFile(_MODEL_PATH) as z:
        with tempfile.TemporaryDirectory() as tmpdir:
            z.extract("model.weights.h5", tmpdir)
            model.load_weights(os.path.join(tmpdir, "model.weights.h5"))

    return model


def _build_pt_model(meta: dict):
    """train_fall.py 의 FallLSTM 과 동일한 PyTorch 모델 (legacy 2-class).

    meta JSON 에서 input_dim/hidden_dim/num_layers/num_classes 를 읽고,
    누락된 항목은 학습 스크립트의 기본값으로 보강한다.
    """
    import torch.nn as nn

    input_dim = int(meta.get("input_dim", NUM_FEATURES))
    hidden_dim = int(meta.get("hidden_dim", 64))
    num_layers = int(meta.get("num_layers", 2))
    num_classes = int(meta.get("num_classes", 2))
    dropout = float(meta.get("dropout", 0.3))

    class FallLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):  # x: (B, T, input_dim)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return FallLSTM()


def _build_pt_vb_model(meta: dict):
    """vB Keras 와 동일 구조의 PyTorch 모델 (BiLSTM × 2 + Dense × 2, 5-class).

    qa/convert_keras_to_pt.py 또는 qa/train_vB_pt.py 가 저장한 .pt 와 호환.
    """
    import torch
    import torch.nn as nn

    input_dim = int(meta.get("input_dim", NUM_FEATURES))
    lstm_units = meta.get("lstm_units", [128, 64])
    dense_units = meta.get("dense_units", [64, int(meta.get("num_classes", 5))])
    L1, L2 = int(lstm_units[0]), int(lstm_units[1])
    D1, D2 = int(dense_units[0]), int(dense_units[1])

    class FallVBLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.bilstm1 = nn.LSTM(input_dim, L1, batch_first=True, bidirectional=True)
            self.bilstm2 = nn.LSTM(2 * L1, L2, batch_first=True, bidirectional=True)
            self.dense1 = nn.Linear(2 * L2, D1)
            self.dense2 = nn.Linear(D1, D2)

        def forward(self, x):
            x, _ = self.bilstm1(x)
            _, (h_n, _) = self.bilstm2(x)
            x = torch.cat([h_n[0], h_n[1]], dim=-1)
            x = torch.relu(self.dense1(x))
            return self.dense2(x)

    return FallVBLSTM()


def _load_pt_model():
    """fall_lstm.pt + fall_lstm_meta.json (legacy 2-class) 로드."""
    import torch

    meta: dict = {}
    if os.path.exists(_PT_META_PATH):
        try:
            with open(_PT_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[FallDetector] meta JSON 파싱 실패 — 기본값 사용: {e}")

    model = _build_pt_model(meta)
    state = torch.load(_PT_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def _load_pt_vb_model():
    """fall_lstm_vB.pt + fall_lstm_vB_meta.json (5-class vB) 로드."""
    import torch

    if not os.path.exists(_PT_VB_META_PATH):
        raise FileNotFoundError(f"vB meta 파일 부재: {_PT_VB_META_PATH}")
    with open(_PT_VB_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if int(meta.get("window_size", 0)) != SEQUENCE_LEN:
        raise ValueError(
            f"vB meta window_size 불일치 — 기대 {SEQUENCE_LEN}, "
            f"실제 {meta.get('window_size')}"
        )
    if int(meta.get("num_classes", 0)) != 5:
        raise ValueError(
            f"vB meta num_classes 불일치 — 기대 5, 실제 {meta.get('num_classes')}"
        )

    model = _build_pt_vb_model(meta)
    state = torch.load(_PT_VB_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def _read_vb_labels() -> dict:
    """labels_vB.json 을 읽어 메타 정보(version/sequence_len/num_features 등) 반환."""
    if not os.path.exists(_VB_LABELS_PATH):
        return {}
    try:
        with open(_VB_LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[FallDetector] labels_vB.json 파싱 실패: {e}")
        return {}


def _get_model():
    """LSTM 모델 로드 — PyTorch vB > Keras vB > Legacy PyTorch > 휴리스틱.

    - 모델 로드 성공/실패는 stdout 에 명확히 한 번만 기록한다.
    - 실패 사유는 모듈 변수에 보존되어 `get_model_info()` 가 진단용으로 노출한다.
    - 휴리스틱 폴백은 detect() 가 자체적으로 처리한다.
    """
    global _lstm_model, _lstm_backend, _lstm_meta, _model_load_attempted
    global _keras_fail_reason, _pt_fail_reason, _active_model_path
    global _pt_vb_fail_reason
    if _lstm_model is not None:
        return _lstm_model
    if _model_load_attempted:
        return None
    _model_load_attempted = True

    print("[FallDetector] LSTM 모델 로드 시작 — 우선순위: PyTorch vB > Keras vB > Legacy PT > 휴리스틱")

    # 1) PyTorch vB (.pt, 5-class, fall=0) — 최우선
    if os.path.exists(_PT_VB_MODEL_PATH):
        try:
            _lstm_model, _lstm_meta = _load_pt_vb_model()
            _lstm_backend = "torch_vB"
            _active_model_path = _PT_VB_MODEL_PATH
            win = (_lstm_meta or {}).get("window_size", SEQUENCE_LEN)
            n_classes = (_lstm_meta or {}).get("num_classes", 5)
            ver = (_lstm_meta or {}).get("model_version", "vB")
            print(
                f"[FallDetector] [OK] PyTorch vB 모델 로드 완료 ({ver}) — "
                f"path={_PT_VB_MODEL_PATH}, window_size={win}, "
                f"num_classes={n_classes}, fall_class=0"
            )
            return _lstm_model
        except Exception as e:
            _pt_vb_fail_reason = f"{type(e).__name__}: {e}"
            print(f"[FallDetector] [FAIL] PyTorch vB 로드 실패 — Keras 폴백 시도: {e}")
    else:
        _pt_vb_fail_reason = f"file_not_found: {_PT_VB_MODEL_PATH}"
        print(f"[FallDetector] [SKIP] PT vB 미발견: {_PT_VB_MODEL_PATH}")

    # 2) Keras vB (.keras) — TF 환경에서만 로드 가능
    if os.path.exists(_KERAS_MODEL_PATH):
        try:
            _lstm_model = _build_and_load_model()
            _lstm_backend = "keras"
            _active_model_path = _KERAS_MODEL_PATH
            vb_meta = _read_vb_labels()
            _lstm_meta = vb_meta or None
            ver = vb_meta.get("version", "vB")
            seq = vb_meta.get("sequence_len", SEQUENCE_LEN)
            feat = vb_meta.get("num_features", NUM_FEATURES)
            print(
                f"[FallDetector] [OK] Keras 모델 로드 완료 ({ver}) — "
                f"path={_KERAS_MODEL_PATH}, sequence_len={seq}, "
                f"num_features={feat}, fall_class=0"
            )
            return _lstm_model
        except Exception as e:
            _keras_fail_reason = f"{type(e).__name__}: {e}"
            print(f"[FallDetector] [FAIL] Keras (vB) 로드 실패 — Legacy PT 시도: {e}")
    else:
        _keras_fail_reason = f"file_not_found: {_KERAS_MODEL_PATH}"
        print(f"[FallDetector] [SKIP] vB Keras 미발견: {_KERAS_MODEL_PATH}")

    # 3) Legacy PyTorch (.pt, 2-class, fall=1) — fallback
    if os.path.exists(_PT_MODEL_PATH):
        try:
            _lstm_model, _lstm_meta = _load_pt_model()
            _lstm_backend = "torch_legacy"
            _active_model_path = _PT_MODEL_PATH
            win = (_lstm_meta or {}).get("window_size", "?")
            n_classes = (_lstm_meta or {}).get("num_classes", "?")
            print(
                f"[FallDetector] [OK] Legacy PyTorch 모델 로드 완료 — "
                f"path={_PT_MODEL_PATH}, window_size={win}, "
                f"num_classes={n_classes}, fall_class=1"
            )
            return _lstm_model
        except Exception as e:
            _pt_fail_reason = f"{type(e).__name__}: {e}"
            print(f"[FallDetector] [FAIL] Legacy PT 로드 실패 — 휴리스틱만 사용: {e}")
    else:
        _pt_fail_reason = f"file_not_found: {_PT_MODEL_PATH}"
        print(f"[FallDetector] [SKIP] Legacy PT 미발견: {_PT_MODEL_PATH}")

    # 4) 모든 모델 부재 → 휴리스틱만
    print(
        "[FallDetector] [WARN] LSTM 모델 파일이 없어 휴리스틱(MediaPipe Pose + YOLO bbox)만 사용합니다.\n"
        f"  탐색 경로: {_PT_VB_MODEL_PATH}\n"
        f"             {_KERAS_MODEL_PATH}\n"
        f"             {_PT_MODEL_PATH}"
    )
    return None


def get_model_info() -> dict:
    """현재 활성화된 LSTM backend 의 메타 정보 (진단/UI 노출용).

    web 런타임이 어떤 모델을 실제로 사용하는지 명시적으로 보여준다.
    - `model_backend`: 'keras_vB' | 'pytorch_fallback' | 'heuristic'
    - `model_path`, `sequence_len`, `num_features`, `num_classes`, `fall_class`
    - `fallback_reason`: keras_vB 가 활성이 아닐 때 그 이유
    - `keras_fail_reason`, `pt_fail_reason`: 디버그용 raw exception 메시지
    """
    # 모델이 아직 로드 시도되지 않았으면 lazy load 트리거 (서버 첫 호출 보호)
    _get_model()

    if _lstm_backend == "keras":
        meta = _lstm_meta or {}
        return {
            "model_backend": "keras_vB",
            "model_path": _active_model_path,
            "version": meta.get("version"),
            "sequence_len": int(meta.get("sequence_len", SEQUENCE_LEN)),
            "num_features": int(meta.get("num_features", NUM_FEATURES)),
            "num_classes": int(meta.get("num_classes", 5)),
            "fall_class": 0,
            "fallback_reason": None,
            "keras_fail_reason": None,
            "pt_vb_fail_reason": _pt_vb_fail_reason,  # PT vB 시도 후 fallback 한 경우 보존
            "pt_fail_reason": None,
            "qa_validated": True,
        }
    if _lstm_backend == "torch_vB":
        meta = _lstm_meta or {}
        return {
            "model_backend": "pytorch_vB",
            "model_path": _active_model_path,
            "version": meta.get("model_version", "V2.1-B-5class-coarse"),
            "sequence_len": int(meta.get("window_size", SEQUENCE_LEN)),
            "num_features": int(meta.get("input_dim", NUM_FEATURES)),
            "num_classes": int(meta.get("num_classes", 5)),
            "fall_class": int(meta.get("fall_class", 0)),
            "fallback_reason": None,
            "keras_fail_reason": None,
            "pt_vb_fail_reason": None,
            "pt_fail_reason": None,
            # Keras vB 와 동일 구조이므로 QA baseline 재현 가능 (기본 True).
            # 운영자는 qa/evaluate_vB_pt_npz.py 로 baseline 재현을 검증한 뒤 사용 권장.
            "qa_validated": True,
        }
    if _lstm_backend == "torch_legacy":
        meta = _lstm_meta or {}
        return {
            "model_backend": "pytorch_fallback",
            "model_path": _active_model_path,
            "version": "pt-2class-legacy",
            "sequence_len": int(meta.get("window_size", 15)),
            "num_features": int(meta.get("input_dim", NUM_FEATURES)),
            "num_classes": int(meta.get("num_classes", 2)),
            "fall_class": int(meta.get("fall_class", 1)),
            "fallback_reason": (
                "PyTorch vB(.pt) / Keras vB(.keras) 모두 로드 실패 — Legacy PT(2-class) fallback 사용 중. "
                "QA 검증 결과(0.74 acc / 0.71 Fall recall)는 vB 기준이며, 현재 런타임 성능은 다름."
            ),
            "keras_fail_reason": _keras_fail_reason,
            "pt_vb_fail_reason": _pt_vb_fail_reason,
            "pt_fail_reason": None,
            "qa_validated": False,
        }
    return {
        "model_backend": "heuristic",
        "model_path": None,
        "version": None,
        "sequence_len": None,
        "num_features": None,
        "num_classes": None,
        "fall_class": None,
        "fallback_reason": (
            "LSTM 모델 모두 로드 실패 — MediaPipe Pose 휴리스틱 + YOLO bbox fallback 만 사용."
        ),
        "keras_fail_reason": _keras_fail_reason,
        "pt_vb_fail_reason": _pt_vb_fail_reason,
        "pt_fail_reason": _pt_fail_reason,
        "qa_validated": False,
    }


def _normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Hip-centering + 95th-percentile scale (훈련 전처리와 동일)."""
    T = seq.shape[0]
    pts = seq.astype(np.float32).reshape(T, 33, 4).copy()
    hip = (pts[:, LEFT_HIP_IDX, :3] + pts[:, RIGHT_HIP_IDX, :3]) / 2.0
    pts[:, :, :3] -= hip[:, None, :]
    vis = pts[:, :, 3].reshape(-1)
    xy = pts[:, :, :2].reshape(-1, 2)
    mask = vis > 0.1
    if mask.sum() > 10:
        xyv = xy[mask]
        scale = max(
            float(np.percentile(np.abs(xyv[:, 0]), 95)),
            float(np.percentile(np.abs(xyv[:, 1]), 95)),
            1e-3,
        )
    else:
        scale = 1.0
    pts[:, :, :3] /= scale
    return pts.reshape(T, NUM_FEATURES)


def _heuristic_fall(landmarks, strong_only: bool = False) -> tuple[bool, str]:
    """휴리스틱 낙상 판정.

    조건:
    (a) head_below_hip  : 머리가 골반보다 아래 (강한 신호 — 누움/엎드림)
    (b) torso_horizontal: 어깨-골반 y 차이가 매우 작음 (몸통 수평)
                          — strong_only=True 이면 사용하지 않음

    앉기 동작은 hip이 떨어지더라도 머리/어깨가 계속 hip 위에 있으므로 두 조건
    모두 false → 낙상 아님. strong_only 는 LSTM 활성 이후 휴리스틱 범위를 좁힐 때 사용.

    반환: (is_fall, reason)
    """
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
    hip_y = (
        landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
    ) / 2
    shoulder_y = (
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    ) / 2

    # (a) 머리가 골반 아래
    if nose_y > hip_y:
        return True, "head_below_hip"

    # (b) 몸통 수평 — 워밍업 동안만
    if not strong_only and abs(shoulder_y - hip_y) < HORIZONTAL_TORSO_THRESHOLD:
        return True, "torso_horizontal"

    return False, "upright"


# ═══════════════════════════════════════════════════════════════════════
# Kinematics — pose keypoint 후처리 (lying suppression 용)
# ═══════════════════════════════════════════════════════════════════════

def _extract_kinematics(landmarks) -> dict:
    """현재 프레임의 단순 kinematic 디스크립터.

    - nose_y / shoulder_y / hip_y : MediaPipe 정규화 좌표 (0=top, 1=bottom)
    - torso_angle: 어깨↔골반 벡터의 기울기. 0° = 직립, 90° = 수평
    """
    import math

    nose_y = float(landmarks[mp_pose.PoseLandmark.NOSE].y)
    l_sh_y = float(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
    r_sh_y = float(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    l_hp_y = float(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
    r_hp_y = float(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
    l_sh_x = float(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
    r_sh_x = float(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
    l_hp_x = float(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x)
    r_hp_x = float(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)

    shoulder_y = (l_sh_y + r_sh_y) / 2.0
    hip_y = (l_hp_y + r_hp_y) / 2.0
    shoulder_x = (l_sh_x + r_sh_x) / 2.0
    hip_x = (l_hp_x + r_hp_x) / 2.0

    dy = abs(shoulder_y - hip_y)
    dx = abs(shoulder_x - hip_x)
    # 각도: torso 가 수직에서 얼마나 기울었는지 (0=직립, 90=수평)
    angle = math.degrees(math.atan2(dx, dy + 1e-6))

    return {
        "nose_y": nose_y,
        "shoulder_y": shoulder_y,
        "hip_y": hip_y,
        "torso_angle": angle,
    }


def _compute_kinematic_signals(buffer) -> dict:
    """최근 kinematics 버퍼 → suppression 용 features."""
    n = len(buffer)
    if n < 2:
        return {
            "vertical_drop": 0.0, "motion_score": 0.0,
            "torso_angle_change": 0.0, "samples": n,
            "current_torso_angle": (buffer[-1]["torso_angle"] if n else 0.0),
            "current_hip_y": (buffer[-1]["hip_y"] if n else 0.0),
        }
    hip_ys = [b["hip_y"] for b in buffer]
    angles = [b["torso_angle"] for b in buffer]

    # vertical_drop: 버퍼 내 hip_y 의 최대 변화량 (0~1 normalized)
    vertical_drop = float(max(hip_ys) - min(hip_ys))

    # motion_score: frame-to-frame |Δhip_y| 의 평균
    diffs = [abs(hip_ys[i] - hip_ys[i - 1]) for i in range(1, n)]
    motion_score = float(sum(diffs) / max(1, len(diffs)))

    # torso_angle_change: 첫/마지막 각도 차 (degrees) — 자세 변화 강도
    torso_angle_change = float(abs(angles[-1] - angles[0]))

    return {
        "vertical_drop": round(vertical_drop, 4),
        "motion_score": round(motion_score, 6),
        "torso_angle_change": round(torso_angle_change, 2),
        "samples": n,
        "current_torso_angle": round(angles[-1], 2),
        "current_hip_y": round(hip_ys[-1], 4),
    }


def _model_label(backend: str | None, idx: int, meta: dict | None) -> str:
    """top class 인덱스 → 라벨 이름."""
    if idx < 0:
        return "unknown"
    if backend == "torch_vB":
        labels = (meta or {}).get("labels") or {}
        # meta 의 labels 는 {"0": "Fall", ...} 형태 (str key)
        return labels.get(str(idx)) or _VB_LABELS.get(idx, f"class_{idx}")
    if backend == "keras":
        return _VB_LABELS.get(idx, f"class_{idx}")
    if backend == "torch_legacy":
        return _LEGACY_LABELS.get(idx, f"class_{idx}")
    return f"class_{idx}"


def _yolo_fall_fallback(image: np.ndarray) -> tuple[bool, float]:
    """
    MediaPipe가 사람을 못 잡을 때 YOLO bbox로 수평 자세 판단.
    사람 bbox의 가로/세로 비율이 YOLO_HORIZONTAL_RATIO 이상 → 낙상 후보.
    """
    yolo = _get_yolo()
    if yolo is None:
        return False, 0.0
    try:
        results = yolo(image, verbose=False, classes=[0])  # class 0 = person
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                if h < 1:
                    continue
                ratio = w / h
                conf = float(box.conf[0])
                if ratio >= YOLO_HORIZONTAL_RATIO and conf >= 0.3:
                    # 수평 bbox → 낙상 가능성. 비율이 클수록 신뢰도 높임
                    fall_conf = min(0.75, 0.4 + (ratio - YOLO_HORIZONTAL_RATIO) * 0.1)
                    return True, round(fall_conf, 2)
    except Exception:
        pass
    return False, 0.0


class FallDetector:
    """실시간 낙상 감지기.

    판정 흐름:
    1) MediaPipe Pose 로 keypoint 추출 (실패 시 YOLO bbox fallback).
    2) 30 frame 슬라이딩 버퍼가 차면 LSTM (vB / legacy) 추론.
    3) `argmax == Fall` AND `softmax[Fall] >= FALL_PROB_THRESHOLD` 일 때만 모델 fall.
    4) 휴리스틱(head_below_hip, torso_horizontal) 과 OR 결합.
    5) **lying suppression** — 최근 kinematics 버퍼에 의미 있는 vertical drop 또는
       motion 이 없으면 fall 후보를 'lying_suppressed' 로 강등.
    6) `CONSECUTIVE_THRESHOLD` 회 연속 발생 → emergency.

    임계값은 모두 환경변수로 조정 가능 (default 는 보수적):
      FALL_PROB_THRESHOLD          (0.65)  softmax[Fall] 통과 임계
      FALL_CONSECUTIVE_THRESHOLD   (3)     연속 fall 프레임 → emergency
      FALL_VERTICAL_DROP_MIN       (0.10)  recent hip_y range — 이 미만 = 정적
      FALL_MOTION_SCORE_MIN        (0.005) recent |Δhip_y| 평균 — 이 미만 = 정지
    """

    def __init__(self):
        # 환경변수 → 인스턴스 상수 (테스트/데모에서 다른 값 주입 용이)
        self.CONSECUTIVE_THRESHOLD = _read_env_int(
            "FALL_CONSECUTIVE_THRESHOLD", DEFAULT_FALL_CONSECUTIVE_THRESHOLD, 1, 100,
        )
        self.CONFIRM_WINDOWS = _read_env_int(
            "FALL_CONFIRM_WINDOWS", DEFAULT_FALL_CONFIRM_WINDOWS, 1, 100,
        )
        self.FALL_PROB_THRESHOLD = _read_env_float(
            "FALL_PROB_THRESHOLD", DEFAULT_FALL_PROB_THRESHOLD, 0.0, 1.0,
        )
        self.VERTICAL_DROP_MIN = _read_env_float(
            "FALL_VERTICAL_DROP_MIN", DEFAULT_VERTICAL_DROP_MIN, 0.0, 1.0,
        )
        self.MOTION_SCORE_MIN = _read_env_float(
            "FALL_MOTION_SCORE_MIN", DEFAULT_MOTION_SCORE_MIN, 0.0, 1.0,
        )
        self.SLOW_MOTION_MAX = _read_env_float(
            "FALL_SLOW_MOTION_MAX", DEFAULT_FALL_SLOW_MOTION_MAX, 0.0, 1.0,
        )
        # NEW: dynamic event gate — 정적 누움 상태에서 모델이 Fall 로 튀는 것을 차단
        self.DYNAMIC_MOTION_MIN = _read_env_float(
            "FALL_DYNAMIC_MOTION_MIN", DEFAULT_FALL_DYNAMIC_MOTION_MIN, 0.0, 1.0,
        )
        self.TORSO_CHANGE_MIN = _read_env_float(
            "FALL_TORSO_CHANGE_MIN", DEFAULT_FALL_TORSO_CHANGE_MIN, 0.0, 180.0,
        )
        # NEW: 낙상 사건 latch — 한 번 fall_emergency 가 확정되면 명시적 reset 전까지
        # final_decision 을 "fall_emergency" 로 강제 유지. 알람 재발화 주기는 cooldown 으로 제어.
        self.ALERT_COOLDOWN_SECONDS = _read_env_float(
            "FALL_ALERT_COOLDOWN_SECONDS", DEFAULT_FALL_ALERT_COOLDOWN_SECONDS, 0.0, 86400.0,
        )
        self.KINEMATICS_HISTORY = DEFAULT_KINEMATICS_HISTORY

        # Effective emergency threshold:
        #   alert_triggered=True 가 되려면 counter >= EMERGENCY_THRESHOLD.
        #   CONFIRM_WINDOWS+1 보장 → fall_suspected 가 최소 1 프레임은 항상 fire.
        #   (CONSECUTIVE_THRESHOLD 가 더 크면 그 값을 사용)
        self.EMERGENCY_THRESHOLD = max(
            self.CONSECUTIVE_THRESHOLD, self.CONFIRM_WINDOWS + 1
        )

        self.fall_counters = defaultdict(int)
        self.frame_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=SEQUENCE_LEN)
        )
        self.lstm_started: dict[str, bool] = defaultdict(bool)
        # NEW: per-camera 최근 kinematics 히스토리 (suppression 판정용)
        self.kin_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.KINEMATICS_HISTORY)
        )
        # NEW: 낙상 사건 latch — 한 번 fall_emergency 가 확정되면 명시적 reset 전까지
        # final_decision 을 강제로 "fall_emergency" 로 유지. lying_suppressed/normal/
        # movement_pending 으로 자동 전이하지 못하게 막는다.
        self.fall_incident_active: dict[str, bool] = defaultdict(bool)
        self.fall_incident_started_at: dict[str, float | None] = defaultdict(lambda: None)
        self.fall_incident_id: dict[str, str | None] = defaultdict(lambda: None)
        self.last_alert_at: dict[str, float | None] = defaultdict(lambda: None)

    def _apply_incident_latch(
        self,
        camera_id: str,
        tentative_decision: str,
        tentative_status: str,
        count: int,
        *,
        model_backend: str | None = None,
        model_prediction: str | None = None,
        fall_probability: float = 0.0,
        dynamic_event: bool = False,
        warmup: bool = False,
    ) -> tuple[str, str, bool, str | None, dict]:
        """낙상 사건 latch — fall_emergency 가 한 번 확정되면 명시적 reset 전까지 유지.

        latch 활성화는 다음 모든 조건을 만족할 때만 허용:
          - model_backend == "pytorch_vB"
          - model_prediction == "Fall"
          - fall_probability >= FALL_PROB_THRESHOLD
          - dynamic_event == True
          - warmup == False
          - tentative_decision == "fall_emergency" (count >= EMERGENCY_THRESHOLD)

        반환: (final_decision, status, alert_triggered, alert_reason, incident_meta)
            incident_meta 에 latch_allowed / latch_block_reason 포함.
        """
        import time
        now = time.time()
        was_active = self.fall_incident_active[camera_id]

        # ── latch 활성화 게이트 계산 ─────────────────────────────────────
        block_reasons = []
        if model_backend != "pytorch_vB":
            block_reasons.append(f"backend={model_backend}!=pytorch_vB")
        if model_prediction != "Fall":
            block_reasons.append(f"model_prediction={model_prediction}!=Fall")
        if fall_probability < self.FALL_PROB_THRESHOLD:
            block_reasons.append(
                f"fall_prob={fall_probability:.3f}<{self.FALL_PROB_THRESHOLD:.3f}"
            )
        if not dynamic_event:
            block_reasons.append("dynamic_fall_event=false")
        if warmup:
            block_reasons.append("warmup=true")
        latch_allowed = (len(block_reasons) == 0)
        latch_block_reason = " & ".join(block_reasons) if block_reasons else None

        if tentative_decision == "fall_emergency" and not was_active and latch_allowed:
            # 새 사건 시작 — 모든 게이트 통과
            self.fall_incident_active[camera_id] = True
            self.fall_incident_started_at[camera_id] = now
            self.fall_incident_id[camera_id] = f"FALL_{int(now * 1000)}_{camera_id}"
            self.last_alert_at[camera_id] = now
            alert_triggered = True
            alert_reason = (
                f"new_incident: id={self.fall_incident_id[camera_id]}, "
                f"count={count}>=EMERGENCY_THRESHOLD={self.EMERGENCY_THRESHOLD}, "
                f"all_gates_passed"
            )
            final_decision = "fall_emergency"
            status = "emergency"
        elif tentative_decision == "fall_emergency" and not was_active and not latch_allowed:
            # tentative 가 emergency 이지만 게이트 차단 — 알림 발화 금지, 상태 강등
            final_decision = "fall_suspected"
            status = "suspected"
            alert_triggered = False
            alert_reason = f"latch_blocked: {latch_block_reason}"
        elif was_active:
            # 사건이 이미 진행 중 — 상태 강제 유지
            last = self.last_alert_at[camera_id] or 0.0
            elapsed = now - last
            if self.ALERT_COOLDOWN_SECONDS > 0 and elapsed >= self.ALERT_COOLDOWN_SECONDS:
                self.last_alert_at[camera_id] = now
                alert_triggered = True
                alert_reason = (
                    f"incident_reminder: id={self.fall_incident_id[camera_id]}, "
                    f"elapsed={elapsed:.1f}s>=cooldown={self.ALERT_COOLDOWN_SECONDS:.1f}s"
                )
            else:
                alert_triggered = False
                alert_reason = None
            final_decision = "fall_emergency"
            status = "emergency"
        else:
            # 사건 비활성 — tentative 그대로 사용
            final_decision = tentative_decision
            status = tentative_status
            alert_triggered = (final_decision == "fall_emergency")
            alert_reason = (
                f"confirmed_dynamic_fall: count={count}>=EMERGENCY_THRESHOLD={self.EMERGENCY_THRESHOLD}"
                if alert_triggered else None
            )

        meta = {
            "fall_incident_active": self.fall_incident_active[camera_id],
            "incident_state": "active" if self.fall_incident_active[camera_id] else "inactive",
            "fall_incident_started_at": self.fall_incident_started_at[camera_id],
            "fall_incident_id": self.fall_incident_id[camera_id],
            "latch_allowed": latch_allowed,
            "latch_block_reason": latch_block_reason,
        }
        return final_decision, status, alert_triggered, alert_reason, meta

    def _is_slow_transition(self, kin_signals: dict) -> bool:
        """천천히 눕기/일어나기 같은 점진적 자세 변화 감지.

        조건:
          - vertical_drop 은 의미 있게 발생 (>= VERTICAL_DROP_MIN)
            → 그냥 정자세가 아니라 자세가 변하는 중
          - motion_score 는 'slow' 범위 (< SLOW_MOTION_MAX)
            → 실제 낙상의 급격한 motion spike 가 없음

        TRUE 면 fall 후보를 movement_pending 로 강등하고 counter 를 hold 한다
        (증가도 reset 도 하지 않음). 시간이 지나 자세가 안정되면 lying suppression
        이 받아쳐 lying_suppressed 로 자연 전이된다.
        """
        if kin_signals.get("samples", 0) < KIN_MIN_SAMPLES_FOR_SUPPRESSION:
            return False
        drop = float(kin_signals.get("vertical_drop", 0.0))
        motion = float(kin_signals.get("motion_score", 0.0))
        return (drop >= self.VERTICAL_DROP_MIN) and (motion < self.SLOW_MOTION_MAX)

    def _compute_dynamic_event(self, kin_signals: dict) -> tuple[bool, str]:
        """급격한 동적 사건 감지 게이트 — 진짜 낙상에만 emergency 를 허용.

        세 가지 신호가 모두 fall-like 임계 이상이어야 dynamic event 로 인정:
          - vertical_drop          >= VERTICAL_DROP_MIN
          - motion_score           >= DYNAMIC_MOTION_MIN
          - torso_angle_change(°)  >= TORSO_CHANGE_MIN

        하나라도 미달이면 dynamic event 가 아니다. 즉 모델이 Fall 로 잘못 예측하더라도
        kinematics 가 받쳐주지 않으면 emergency 로 승격되지 않는다.

        Returns:
            (active: bool, reason: str)
            - active = True  : 진짜 낙상 같은 동적 변화가 관찰됨
            - active = False : 정적 상태 또는 천천히 변화 — emergency 차단해야 함
            - reason : 사람이 읽을 수 있는 진단 문자열 (CameraPage 디버그 스트립용)
        """
        if kin_signals.get("samples", 0) < KIN_MIN_SAMPLES_FOR_SUPPRESSION:
            return False, f"insufficient_samples={kin_signals.get('samples', 0)}<{KIN_MIN_SAMPLES_FOR_SUPPRESSION}"
        drop = float(kin_signals.get("vertical_drop", 0.0))
        motion = float(kin_signals.get("motion_score", 0.0))
        angle = float(kin_signals.get("torso_angle_change", 0.0))

        drop_ok = drop >= self.VERTICAL_DROP_MIN
        motion_ok = motion >= self.DYNAMIC_MOTION_MIN
        angle_ok = angle >= self.TORSO_CHANGE_MIN

        if drop_ok and motion_ok and angle_ok:
            return True, (
                f"all_satisfied: drop={drop:.4f}>={self.VERTICAL_DROP_MIN:.4f} "
                f"& motion={motion:.5f}>={self.DYNAMIC_MOTION_MIN:.5f} "
                f"& torso_Δ={angle:.1f}°>={self.TORSO_CHANGE_MIN:.1f}°"
            )

        unmet = []
        if not drop_ok:
            unmet.append(f"drop={drop:.4f}<{self.VERTICAL_DROP_MIN:.4f}")
        if not motion_ok:
            unmet.append(f"motion={motion:.5f}<{self.DYNAMIC_MOTION_MIN:.5f}")
        if not angle_ok:
            unmet.append(f"torso_Δ={angle:.1f}°<{self.TORSO_CHANGE_MIN:.1f}°")
        return False, "static_state: " + " & ".join(unmet)

    def _decode_image(self, image_base64: str) -> np.ndarray:
        img_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _extract_features(self, image: np.ndarray):
        """MediaPipe Pose → (132,) 특징 벡터 + landmarks 반환."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None
        lm = results.pose_landmarks.landmark
        feat = np.array(
            [[l.x, l.y, l.z, l.visibility] for l in lm], dtype=np.float32
        ).reshape(NUM_FEATURES)
        return feat, lm

    def _predict_lstm(self, buffer: deque):
        """LSTM 추론 — backend 별로 분기.

        Returns:
            (fall_prob: float, all_probs: np.ndarray | None, top_idx: int)
            - fall_prob   : softmax[Fall] 확률
            - all_probs   : 전체 클래스별 softmax (None = 추론 불가)
            - top_idx     : argmax 클래스 인덱스 (-1 = 추론 불가)
        """
        model = _get_model()
        if model is None:
            return 0.0, None, -1

        if _lstm_backend == "keras":
            if len(buffer) < SEQUENCE_LEN:
                return 0.0, None, -1
            seq = np.stack(list(buffer), axis=0)
            seq_norm = _normalize_sequence(seq)
            x = seq_norm[np.newaxis, ...].astype(np.float32)
            probs = np.asarray(model.predict(x, verbose=0)[0], dtype=np.float32)
            top_idx = int(np.argmax(probs))
            return float(probs[FALL_CLASS]), probs, top_idx

        if _lstm_backend == "torch_vB":
            import torch
            meta = _lstm_meta or {}
            win = int(meta.get("window_size", SEQUENCE_LEN))
            if len(buffer) < win:
                return 0.0, None, -1
            tail = list(buffer)[-win:]
            seq = np.stack(tail, axis=0)
            seq_norm = _normalize_sequence(seq)
            x = torch.from_numpy(seq_norm[np.newaxis, ...].astype(np.float32))
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
            fall_idx = int(meta.get("fall_class", 0))
            top_idx = int(np.argmax(probs))
            return float(probs[fall_idx]), probs, top_idx

        if _lstm_backend == "torch_legacy":
            import torch
            meta = _lstm_meta or {}
            win = int(meta.get("window_size", 15))
            if len(buffer) < win:
                return 0.0, None, -1
            tail = list(buffer)[-win:]
            seq = np.stack(tail, axis=0).astype(np.float32)
            x = torch.from_numpy(seq[np.newaxis, ...])
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
            num_classes = int(meta.get("num_classes", 2))
            fall_idx = int(meta.get("fall_class", 1 if num_classes >= 2 else 0))
            top_idx = int(np.argmax(probs))
            return float(probs[fall_idx]), probs, top_idx

        return 0.0, None, -1

    def _update_counter(self, camera_id: str, is_fall: bool) -> int:
        if is_fall:
            self.fall_counters[camera_id] += 1
        else:
            self.fall_counters[camera_id] = 0
        return self.fall_counters[camera_id]

    def _status_from_count(self, count: int, fall_active: bool) -> str:
        """consecutive 카운트 → status 문자열."""
        if not fall_active:
            return "normal"
        if count >= self.CONSECUTIVE_THRESHOLD:
            return "emergency"
        if count >= 2:
            return "suspected"
        if count >= 1:
            return "caution"
        return "normal"

    def detect(self, image_base64: str, camera_id: str = "default") -> dict:
        image = self._decode_image(image_base64)
        features, landmarks = self._extract_features(image)

        # ─────────────────────────────────────────────────────────
        # A) MediaPipe 가 사람을 못 잡은 경우 → YOLO bbox fallback
        # ─────────────────────────────────────────────────────────
        if features is None:
            is_fall, yolo_conf = _yolo_fall_fallback(image)
            # YOLO bbox fallback 도 같은 state machine 을 따른다 (kinematics 부재이므로
            # transition / suppression 미적용 — 단순 카운트 기반).
            if is_fall:
                self.fall_counters[camera_id] += 1
            else:
                self.fall_counters[camera_id] = 0
            count = self.fall_counters[camera_id]

            if not is_fall:
                final_decision, status = "no_person", "no_person"
            elif count == 0:
                final_decision, status = "normal", "normal"
            elif count < self.CONFIRM_WINDOWS:
                final_decision, status = "movement_pending", "checking"
            elif count < self.EMERGENCY_THRESHOLD:
                final_decision, status = "fall_suspected", "suspected"
            else:
                final_decision, status = "fall_emergency", "emergency"

            # 사건 latch 적용 — YOLO bbox 만으로는 dynamic_event 가 false 이므로
            # 사실상 latch 활성화 불가 (의도된 동작 — pose 기반 신호가 있어야 알람).
            final_decision, status, alert_triggered, alert_reason, incident_meta = (
                self._apply_incident_latch(
                    camera_id, final_decision, status, count,
                    model_backend="yolo_bbox",
                    model_prediction="yolo_bbox" if is_fall else "no_pose",
                    fall_probability=0.0,
                    dynamic_event=False,
                    warmup=False,
                )
            )
            return {
                "detected": alert_triggered,
                "alert_triggered": alert_triggered,
                "alert_reason": alert_reason,
                "confidence": round(float(yolo_conf), 3),
                "fall_probability": 0.0,
                "model_prediction": "no_pose" if not is_fall else "yolo_bbox",
                "final_decision": final_decision,
                "suppression_reason": None,
                "transition_state": None,
                # YOLO fallback 은 kinematics 가 없으므로 dynamic event 정보 없음
                "dynamic_fall_event": False,
                "dynamic_gate_reason": "no_pose_kinematics",
                "latch_allowed": incident_meta["latch_allowed"],
                "latch_block_reason": incident_meta["latch_block_reason"],
                "vertical_drop": 0.0,
                "motion_score": 0.0,
                "torso_angle_change": 0.0,
                "recent_vertical_drop": 0.0,
                "recent_motion_score": 0.0,
                "recent_torso_angle_change": 0.0,
                "consecutive_fall_windows": count,
                "pending_windows": max(0, self.EMERGENCY_THRESHOLD - count) if count > 0 else 0,
                "fall_incident_active": incident_meta["fall_incident_active"],
                "incident_state": incident_meta["incident_state"],
                "fall_incident_started_at": incident_meta["fall_incident_started_at"],
                "fall_incident_id": incident_meta["fall_incident_id"],
                "type": "FALL",
                "status": status,
                "consecutive_frames": count,  # legacy alias
                "method": "yolo_bbox" if is_fall else "no_pose",
            }

        # ─────────────────────────────────────────────────────────
        # B) MediaPipe 성공 — keypoint 추출 + kinematics 버퍼링
        # ─────────────────────────────────────────────────────────
        self.frame_buffers[camera_id].append(features)
        buffer_full = len(self.frame_buffers[camera_id]) >= SEQUENCE_LEN

        kin = _extract_kinematics(landmarks)
        self.kin_buffers[camera_id].append(kin)
        kin_signals = _compute_kinematic_signals(self.kin_buffers[camera_id])

        # 휴리스틱 (워밍업 동안 더 관대, LSTM 활성 후엔 강한 신호만)
        heuristic_is_fall, reason = _heuristic_fall(landmarks, strong_only=buffer_full)

        # ── B-1) 워밍업: 버퍼 부족 → 휴리스틱 + Phase 1/2/3 gate 적용 ──
        # 워밍업 단계에서는 fall_emergency 까지 도달 가능하지만 latch 는 warmup=True 로 차단.
        if not buffer_full:
            suppressed = False
            suppression_reason = None
            is_transitioning = False
            transition_state = None

            # Phase 3 dynamic_event 는 항상 계산 (gate + UI 노출 둘 다 사용)
            warmup_dynamic_event, warmup_dynamic_reason = self._compute_dynamic_event(kin_signals)

            # 휴리스틱이 fall 을 외칠 때만 후처리 적용
            if heuristic_is_fall and kin_signals["samples"] >= KIN_MIN_SAMPLES_FOR_SUPPRESSION:
                # Phase 1: 정적 누움
                if (kin_signals["vertical_drop"] < self.VERTICAL_DROP_MIN
                        and kin_signals["motion_score"] < self.MOTION_SCORE_MIN):
                    suppressed = True
                    suppression_reason = (
                        f"lying_no_dynamic_event: drop={kin_signals['vertical_drop']:.4f}<"
                        f"{self.VERTICAL_DROP_MIN:.4f} & motion="
                        f"{kin_signals['motion_score']:.5f}<{self.MOTION_SCORE_MIN:.5f}"
                    )
                # Phase 2: 천천히 눕기
                elif self._is_slow_transition(kin_signals):
                    is_transitioning = True
                    transition_state = (
                        f"slow_lying_or_sitting: drop={kin_signals['vertical_drop']:.4f}>="
                        f"{self.VERTICAL_DROP_MIN:.4f} & motion="
                        f"{kin_signals['motion_score']:.5f}<{self.SLOW_MOTION_MAX:.5f}"
                    )
                # Phase 3: 정적-while-Fall (dynamic event 없음) → suppress
                elif not warmup_dynamic_event:
                    suppressed = True
                    suppression_reason = f"static_no_dynamic_event: {warmup_dynamic_reason}"

            confirmed_fall = heuristic_is_fall and not suppressed and not is_transitioning
            # RESET (HOLD 제거) — 천천히 눕는 중간 spike 누적 차단
            if confirmed_fall:
                self.fall_counters[camera_id] += 1
            else:
                self.fall_counters[camera_id] = 0
            count = self.fall_counters[camera_id]

            # 워밍업 단계 상태 — fall_* 라벨은 dynamic_event=True 일 때만
            if suppressed:
                status, final_decision = "normal", "lying_suppressed"
            elif is_transitioning:
                status, final_decision = "checking", "movement_pending"
            elif not warmup_dynamic_event:
                if count > 0 or kin_signals.get("vertical_drop", 0.0) >= self.VERTICAL_DROP_MIN:
                    status, final_decision = "checking", "movement_pending"
                else:
                    status, final_decision = "buffering", "warmup"
            elif count == 0:
                status, final_decision = "buffering", "warmup"
            elif count < self.CONFIRM_WINDOWS:
                status, final_decision = "checking", "movement_pending"
            elif count < self.EMERGENCY_THRESHOLD:
                status, final_decision = "suspected", "fall_suspected"
            else:
                status, final_decision = "emergency", "fall_emergency"

            # 사건 latch 적용 — warmup=True 로 latch 활성화 차단
            final_decision, status, alert_triggered, alert_reason, incident_meta = (
                self._apply_incident_latch(
                    camera_id, final_decision, status, count,
                    model_backend="warmup",
                    model_prediction="warmup",
                    fall_probability=0.0,
                    dynamic_event=warmup_dynamic_event,
                    warmup=True,
                )
            )
            if incident_meta["fall_incident_active"]:
                suppression_reason = None
                transition_state = None

            return {
                "detected": alert_triggered,
                "alert_triggered": alert_triggered,
                "alert_reason": alert_reason,
                "confidence": 0.5 if confirmed_fall else 0.0,
                "fall_probability": 0.0,  # LSTM 미활성
                "model_prediction": "warmup",
                "final_decision": final_decision,
                "suppression_reason": suppression_reason,
                "transition_state": transition_state,
                "dynamic_fall_event": warmup_dynamic_event,
                "dynamic_gate_reason": warmup_dynamic_reason,
                "latch_allowed": incident_meta["latch_allowed"],
                "latch_block_reason": incident_meta["latch_block_reason"],
                "vertical_drop": kin_signals["vertical_drop"],
                "motion_score": kin_signals["motion_score"],
                "torso_angle_change": kin_signals["torso_angle_change"],
                "recent_vertical_drop": kin_signals["vertical_drop"],
                "recent_motion_score": kin_signals["motion_score"],
                "recent_torso_angle_change": kin_signals["torso_angle_change"],
                "consecutive_fall_windows": count,
                "pending_windows": max(0, self.EMERGENCY_THRESHOLD - count) if count > 0 else 0,
                "fall_incident_active": incident_meta["fall_incident_active"],
                "incident_state": incident_meta["incident_state"],
                "fall_incident_started_at": incident_meta["fall_incident_started_at"],
                "fall_incident_id": incident_meta["fall_incident_id"],
                "type": "FALL",
                "status": status,
                "buffered": len(self.frame_buffers[camera_id]),
                "consecutive_frames": count,
                "method": f"heuristic:{reason}",
                "heuristic": heuristic_is_fall,
            }

        # ── B-2) 버퍼 풀 — LSTM 활성 ──
        if not self.lstm_started[camera_id]:
            self.lstm_started[camera_id] = True

        fall_prob, all_probs, top_idx = self._predict_lstm(self.frame_buffers[camera_id])

        # backend 별 fall class index
        if _lstm_backend == "torch_legacy":
            fall_idx = int((_lstm_meta or {}).get("fall_class", 1))
        else:
            fall_idx = int((_lstm_meta or {}).get("fall_class", 0)) if _lstm_backend == "torch_vB" else 0

        # 모델의 top class / 확신도 (fall_probability 와 별개)
        if all_probs is not None and top_idx >= 0:
            model_pred_name = _model_label(_lstm_backend, top_idx, _lstm_meta)
            model_confidence = float(all_probs[top_idx])
        else:
            model_pred_name = "unknown"
            model_confidence = 0.0

        # LSTM fall 후보: argmax 가 Fall 이고 + threshold 통과
        lstm_fall = (top_idx == fall_idx) and (fall_prob >= self.FALL_PROB_THRESHOLD)
        combined_fall = lstm_fall or heuristic_is_fall

        # ── Phase 1: lying suppression (정적인 누움 상태) ────────────────────
        suppressed = False
        suppression_reason = None
        if combined_fall and kin_signals["samples"] >= KIN_MIN_SAMPLES_FOR_SUPPRESSION:
            if (kin_signals["vertical_drop"] < self.VERTICAL_DROP_MIN
                    and kin_signals["motion_score"] < self.MOTION_SCORE_MIN):
                suppressed = True
                suppression_reason = (
                    f"lying_no_dynamic_event: drop={kin_signals['vertical_drop']:.4f}<"
                    f"{self.VERTICAL_DROP_MIN:.4f} & motion="
                    f"{kin_signals['motion_score']:.5f}<{self.MOTION_SCORE_MIN:.5f}"
                )

        # ── Phase 2: slow transition (천천히 눕는 중간 과정) ────────────────
        # vertical_drop 이 의미 있게 발생하지만 motion_score 는 'slow' — 누우려는 중.
        # counter 를 hold 해서 fall_emergency 까지 가지 않도록 막는다.
        is_transitioning = False
        transition_state = None
        if combined_fall and not suppressed:
            if self._is_slow_transition(kin_signals):
                is_transitioning = True
                transition_state = (
                    f"slow_lying_or_sitting: drop={kin_signals['vertical_drop']:.4f}>="
                    f"{self.VERTICAL_DROP_MIN:.4f} & motion="
                    f"{kin_signals['motion_score']:.5f}<{self.SLOW_MOTION_MAX:.5f}"
                )

        # ── Phase 3: Dynamic event gate ────────────────────────────────────
        # 모델이 Fall 로 예측해도 최근 window 에 진짜 급격한 변화 (drop + motion + 회전)
        # 가 없으면 dynamic_fall_event=False → emergency 로 승격 불가.
        # 항상 계산해서 API 로 노출 (운영자/UI 진단).
        dynamic_event, dynamic_gate_reason = self._compute_dynamic_event(kin_signals)

        # 정적-while-Fall: 모델이 Fall 인데 dynamic_event=False 이고 Phase 1/2 도 미적용.
        # → 이 경우는 누운 뒤 정지 상태에서 모델이 헛소리 하는 케이스. suppress.
        if combined_fall and not suppressed and not is_transitioning and not dynamic_event:
            suppressed = True
            suppression_reason = f"static_no_dynamic_event: {dynamic_gate_reason}"

        # ── Counter 업데이트 — 엄격 모드 ─────────────────────────────────
        # confirmed_fall: combined_fall AND NOT suppressed AND NOT transitioning
        #   (suppressed 가 이미 dynamic_event=False 케이스를 잡으므로 confirmed_fall 자체가
        #    "dynamic_event=True 한 frame" 을 의미)
        # 그 외 (transitioning 포함) — RESET. 천천히 눕는 중에 brief motion spike 가
        # 있어도 counter 가 누적되지 않도록 한다.
        confirmed_fall = combined_fall and not suppressed and not is_transitioning
        if confirmed_fall:
            self.fall_counters[camera_id] += 1
        else:
            self.fall_counters[camera_id] = 0
        count = self.fall_counters[camera_id]

        # ── 상태 머신 — fall_* 라벨은 dynamic_event=True 일 때만 노출 ────
        if suppressed:
            final_decision = "lying_suppressed"
            status = "normal"
        elif is_transitioning:
            final_decision = "movement_pending"
            status = "checking"
        elif not dynamic_event:
            # Defense-in-depth — counter > 0 이라도 현재 dynamic_event=False 면
            # fall_* 라벨 절대 X. 천천히 눕거나 자세 전환 중인 모호한 상태로 처리.
            if count > 0 or kin_signals.get("vertical_drop", 0.0) >= self.VERTICAL_DROP_MIN:
                final_decision = "movement_pending"
                status = "checking"
            else:
                final_decision = "normal"
                status = "normal"
        elif count == 0:
            final_decision = "normal"
            status = "normal"
        elif count < self.CONFIRM_WINDOWS:
            final_decision = "movement_pending"
            status = "checking"
        elif count < self.EMERGENCY_THRESHOLD:
            # 여기 도달 = dynamic_event=True (위 분기에서 이미 거름)
            final_decision = "fall_suspected"
            status = "suspected"
        else:  # count >= EMERGENCY_THRESHOLD AND dynamic_event=True
            if lstm_fall:
                final_decision = "fall_emergency"
                status = "emergency"
            else:
                # 모델이 더 이상 Fall 을 직접 예측하지 않음 → suspended
                final_decision = "fall_suspected"
                status = "suspected"

        # ── 사건 latch 적용 — 모든 게이트 통과 시에만 활성화 ────────────────
        # backend 식별자 (get_model_info 와 일관)
        if _lstm_backend == "torch_vB":
            backend_name = "pytorch_vB"
        elif _lstm_backend == "keras":
            backend_name = "keras_vB"
        elif _lstm_backend == "torch_legacy":
            backend_name = "pytorch_fallback"
        else:
            backend_name = "heuristic"
        final_decision, status, alert_triggered, alert_reason, incident_meta = (
            self._apply_incident_latch(
                camera_id, final_decision, status, count,
                model_backend=backend_name,
                model_prediction=model_pred_name,
                fall_probability=float(fall_prob),
                dynamic_event=dynamic_event,
                warmup=False,  # LSTM-active branch — not warmup
            )
        )
        # latch 가 활성이면 lying suppression / transition 표시도 가리지 않음
        if incident_meta["fall_incident_active"]:
            suppression_reason = None
            transition_state = None
        # pending 단계에서 EMERGENCY 까지 남은 프레임 수
        pending_windows = max(0, self.EMERGENCY_THRESHOLD - count) if count > 0 else 0

        # method 표기 (legacy 호환)
        if suppressed:
            method = "lstm+kinematic_suppress"
        elif is_transitioning:
            method = "lstm+slow_transition_hold"
        elif lstm_fall:
            method = "lstm"
        elif heuristic_is_fall:
            method = f"heuristic:{reason}"
        else:
            method = "lstm"

        return {
            # 알림 게이팅 — frontend / backend 둘 다 이 필드만 보면 됨
            "detected": alert_triggered,
            "alert_triggered": alert_triggered,
            "alert_reason": alert_reason,
            # 의미 분리:
            #   confidence       = 모델이 자기 top prediction 에 대해 얼마나 확신하는지
            #   fall_probability = 모델의 Fall 클래스 raw softmax 확률
            "confidence": round(model_confidence, 3),
            "fall_probability": round(float(fall_prob), 3),
            "model_prediction": model_pred_name,
            "final_decision": final_decision,
            "suppression_reason": suppression_reason,
            "transition_state": transition_state,
            # NEW: dynamic event gate 신호
            "dynamic_fall_event": dynamic_event,
            "dynamic_gate_reason": dynamic_gate_reason,
            # latch 게이트 가시화 (warmup/backend/prob/dynamic 가드 결과)
            "latch_allowed": incident_meta["latch_allowed"],
            "latch_block_reason": incident_meta["latch_block_reason"],
            # kinematic signals — 'recent_*' 는 동일 값의 명시적 이름 (API 일관성)
            "vertical_drop": kin_signals["vertical_drop"],
            "motion_score": kin_signals["motion_score"],
            "torso_angle_change": kin_signals["torso_angle_change"],
            "recent_vertical_drop": kin_signals["vertical_drop"],
            "recent_motion_score": kin_signals["motion_score"],
            "recent_torso_angle_change": kin_signals["torso_angle_change"],
            "consecutive_fall_windows": count,
            "pending_windows": pending_windows,
            # NEW: 사건 latch 정보
            "fall_incident_active": incident_meta["fall_incident_active"],
            "incident_state": incident_meta["incident_state"],
            "fall_incident_started_at": incident_meta["fall_incident_started_at"],
            "fall_incident_id": incident_meta["fall_incident_id"],
            # legacy/호환 필드 유지 (test_fall_api 의 'method' 검증용)
            "type": "FALL",
            "status": status,
            "consecutive_frames": count,
            "method": method,
            "lstm_prob": round(float(fall_prob), 3),
            "heuristic": heuristic_is_fall,
        }

    def reset(self, camera_id: str = "default"):
        """카운터 / 버퍼 / 사건 latch 모두 초기화."""
        self.fall_counters[camera_id] = 0
        self.frame_buffers[camera_id].clear()
        self.kin_buffers[camera_id].clear()
        self.lstm_started[camera_id] = False
        # 낙상 사건 해제 — 명시적 reset 에서만 가능
        self.fall_incident_active[camera_id] = False
        self.fall_incident_started_at[camera_id] = None
        self.fall_incident_id[camera_id] = None
        self.last_alert_at[camera_id] = None


fall_detector = FallDetector()

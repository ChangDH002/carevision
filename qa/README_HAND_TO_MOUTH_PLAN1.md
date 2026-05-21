# CareVision 1안 — 손→입 행동 감지 QA

## 1. 1안의 목적

1안은 카메라 영상에서 **복약으로 볼 수 있는 핵심 행동 = "손이 입 근처로
이동하는 동작"** 을 안정적으로 감지하는 방식이다.

이 작업의 범위는 **손→입 행동 후보 신호를 안정화**하는 것이며, 복약
완료/성공을 확정하는 것이 아니다. 1안은 2안(등록 객체)과 비교·병합하지
않고, 2안의 보조/ fallback 도 아니다. 이번 작업은 손→입 LSTM 감지
고도화만 다룬다.

## 2. 감지 방식

### 2.1 LSTM 기반 감지 (기존 — 구조 변경 없음)
- MediaPipe Hands 21랜드마크 × (x, y) = 42차원 feature
- 2-layer LSTM(input=42, hidden=128) → FC(128→64)→ReLU→Dropout→FC(64→1)
- 체크포인트 `ai/models/hand_to_mouth_lstm.pt`, 시퀀스 길이 15
- `lstm_confidence` = sigmoid 출력. `lstm_confidence ≥ threshold` 이고
  연속 양성 프레임이 `LSTM_MIN_CONSECUTIVE` 이상이면 `lstm_detected=true`
- ⚠️ 모델 구조/체크포인트/42차원 feature 는 변경하지 않았다.

### 2.2 거리 rule 기반 보조 감지 (신규 추가)
- MediaPipe FaceDetection 으로 입(mouth) 키포인트 추출
- 손끝(엄지·검지·중지·약지·소지 tip)–입 정규화 거리 중 최솟값 계산
- `rule_distance ≤ rule_threshold` 가 연속 `RULE_MIN_CONSECUTIVE`
  프레임 유지되면 `rule_detected=true`

### 2.3 motion 게이팅 (신규)
- 검지 tip 의 최근 프레임 이동량 평균(`motion_score`)이
  `MOTION_MIN` 미만이면 `motion_accepted=false` → 정적 오탐 억제

## 3. final_detected 계산

```
lstm_detected = lstm_confidence ≥ threshold AND lstm_consecutive ≥ N
rule_detected = rule_distance ≤ dist_thr AND rule_consecutive ≥ M
raw_detected  = lstm_detected OR rule_detected

if not has_hand            → status=insufficient_landmarks, final_detected=false
elif raw and not motion    → status=motion_not_accepted,    final_detected=false
elif raw                   → status=hand_to_mouth_detected, final_detected=true
else                       → status=no_hand_to_mouth,       final_detected=false
```

`final_detected = (lstm_detected OR rule_detected)` 이되, **landmark 부족 /
motion 부족** 게이팅이 우선한다.

## 4. detection_source 의미

| 값 | 의미 |
|---|---|
| `both` | LSTM 과 거리 rule 둘 다 감지 |
| `lstm` | LSTM 만 감지 |
| `rule` | 거리 rule 만 감지 |
| `none` | 둘 다 미감지 |

## 5. 응답 필드 (`/detect/live` → `hand_to_mouth`)

기존 키(`detected`, `confidence`, `status`)는 호환 유지하고 아래 필드를
**additive** 로 추가했다.

```jsonc
{
  "hand_to_mouth": {
    "detected": true,                 // 기존 키 (scorer EMA 기반, 호환)
    "confidence": 0.71,               // 기존 키 (scorer EMA prob, 호환)
    "status": "hand_to_mouth_detected",
    "final_detected": true,           // 1안 최종 손→입 행동 감지
    "detection_source": "both",       // lstm | rule | both | none
    "lstm_detected": true,
    "lstm_confidence": 0.74,
    "rule_detected": true,
    "rule_distance": 0.34,
    "rule_threshold": 0.45,
    "consecutive_frames": 18,         // rule 연속 충족 프레임
    "lstm_consecutive": 4,            // LSTM 연속 양성 프레임
    "motion_accepted": true,
    "motion_score": 0.012,
    "has_hand": true,
    "has_mouth": true,
    "reason": "손이 입 근처로 이동하는 동작이 감지되었습니다. (복약 완료 확정 아님)"
  }
}
```

> 구현 노트: `routes.py` 는 스코어러가 이미 호출한 detector 의
> `last_result` 캐시를 읽어 위 필드를 채운다. **이중 추론이 없고**,
> `medication_scorer` / `medication_detector` 코드는 수정하지 않았다.

## 6. 한계

손→입 행동은 **복약 가능성을 나타내는 행동 신호**일 뿐, 복약 완료
확정이 아니다. 물 마시기·간식 등에서도 손→입 동작이 발생할 수 있다.
따라서 1안 단독으로 "복약 완료" 를 단정하지 않는다.

## 7. 담당 범위

### AI / 데이터 담당 (이번 작업 범위)
- 손→입 행동 감지 신호(LSTM + 거리 rule + motion 게이팅)
- `final_detected`, `detection_source`, `confidence`, debug 필드
- fusion 규칙 단위 QA

### 백엔드 / 프론트 담당 (이번 작업 범위 밖)
- 최종 복약 상태(taken/missed/suspected) 판단
- 복약 스케줄 결합, 사용자/보호자 확인
- 복약 이력 저장, 알림 발송

## 8. 프론트 표시 규칙

`CameraPage.jsx` 의 손→입 카드에 표시. 허용 문구만 사용한다.
- 허용: "손→입 행동 감지", "복약 유사 동작 후보", "LSTM 감지",
  "거리 규칙 감지", "복약 완료 확정 아님"
- 금지: "복약 완료", "약을 복용했습니다", "복약 성공", "taken=true"
- `final_detected=false` 면 후보 표현조차 띄우지 않는다.

## 9. QA 테스트

`qa/test_hand_to_mouth_plan1.py` (pytest, 순수 함수 — cv2/mediapipe/torch
의존 없음).

```cmd
cd C:\carevision
ai\venv\Scripts\python.exe -m pytest -q qa\test_hand_to_mouth_plan1.py
```

커버 케이스
1. LSTM confidence 높음 → `lstm_detected`, `final_detected`
2. LSTM 낮지만 거리 rule 충족 → `rule_detected`, `final_detected`
3. 둘 다 충족 → `detection_source="both"`
4. 둘 다 미충족 → `final_detected=false`, `none`
5. landmark 부족 → `status="insufficient_landmarks"`
6. motion 부족 → `status="motion_not_accepted"`
7. 파라미터라이즈: 어떤 입력도 `taken=true` 없음 (`score` 키도 없음)
8. fusion 모듈이 `medication_scorer`/`medication_detector`/등록객체/
   무거운 의존성을 import 하지 않음 (구조적 독립)
9. LSTM/rule 연속 프레임 경계 가드

결과: **89 passed**.

> 참고: `qa/test_fall_api.py::test_detect_fall_response_shape` 1건은 이번
> 작업과 무관한 **사전 존재 실패**다 (테스트 더블의 `fall_detector` mock
> 에 `get_model_info` 미정의 — 본 작업 변경 전에도 동일 실패).

## 9.1 실영상 검증 & motion gate 보수적 보완 (2차)

실영상 3종(`hand_to_mouth_test.mp4`, `0326.mp4`,
`hand_to_mouth_test3.mp4`)을 `ai/debug_hand_to_mouth.py` 로 프레임별
분석한 결과:

| 영상 | frames | final_detected(수정전→후) | 비고 |
|---|---:|---|---|
| hand_to_mouth_test.mp4 | 205 | 182 → **182** | 손-입 최소거리 0.457 (>near) → 우회 미발동, 불변 |
| 0326.mp4 | 486 | 38 → **38** | 손-입 최소거리 0.443 (>near) → 우회 미발동, 불변 (오탐 0) |
| hand_to_mouth_test3.mp4 | 192 | 139 → **152** | 손-입 최근접 0.013 → 머무는 순간 회복 |

**원인 분석 (test3):** `insufficient_landmarks` 아님(손+입 189/192
검출), warmup/LSTM 지연 아님. 진짜 원인은 **손이 입에 머무는 순간
motion_score 가 낮아 motion gate 에 막힘** (거리<=thr 150프레임 중 35가
motion 차단), + `rule_distance` 노이즈로 rule 연속 카운터 리셋.

**보수적 수정:** 손끝-입 거리가 `RULE_NEAR_DISTANCE`(기본 0.18, env
`H2M_RULE_NEAR_DIST`) **이하로 매우 가까울 때만** motion gate 우회.
거리가 멀면 gate 그대로 → `0326.mp4`/`test.mp4` 처럼 손이 입에서 먼
영상은 결과 완전 불변(오탐 0 증가). 추가로:
- `pending_near_mouth` 상태: 입에 매우 근접하나 연속/LSTM 미충족 시
  (final_detected=false, 정보용 — 확정 아님)
- `block_reason` 디버그 필드: `motion_gate` / `rule_consecutive_pending`
  / `insufficient_landmarks` / `warming_up` / `no_signal`
- `rule_very_close`, `motion_bypassed` 디버그 플래그

이 수정은 **참양성 회복 전용**이며, 어떤 경우에도 `taken` 을 만들지
않는다(QA 불변식 유지).

## 10. 변경 / 비변경 파일

### 변경
| 파일 | 변경 |
|---|---|
| `ai/pipelines/hand_to_mouth_fusion.py` | **신규** — 의존성 없는 fusion 순수 모듈 + 튜닝 상수 |
| `ai/pipelines/hand_to_mouth_detector.py` | 거리 rule(FaceDetection)·motion 게이팅·`last_result` 캐시 추가, fusion 분리 import. **LSTM 모델 구조/체크포인트/42차원 feature 미변경** |
| `ai/api/routes.py` | `/detect/live` 의 `hand_to_mouth` 에 풍부 필드 additive (이중 추론 없음) |
| `frontend/src/camera/CameraPage.jsx` | 손→입 카드 강화(허용 문구만, `final_detected` 기준) |
| `qa/test_hand_to_mouth_plan1.py` | **신규** — 1안 QA (motion 우회/ pending_near_mouth 케이스 포함) |
| `qa/README_HAND_TO_MOUTH_PLAN1.md` | **신규** — 본 문서 |
| `ai/validate_hand_to_mouth.py` | **신규** — 영상/웹캠 집계 검증 하니스 |
| `ai/debug_hand_to_mouth.py` | **신규** — 프레임별 debug 로거 + 손-입 최근접 분석 |

### 비변경 (명시적으로 건드리지 않음)
- `ai/pipelines/medication_scorer.py`
- `ai/pipelines/medication_detector.py`
- 등록 객체 / registered bottle / 2안 관련 일체 (이번 세션 미진행)
- `medication_score.taken` 등 최종 복약 완료 판정
- 낙상 파이프라인, `qa/test_fall_api.py`

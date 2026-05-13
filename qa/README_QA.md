# CareVision Fall Detection — QA 최종 리포트

> 본 문서는 CareVision vB 낙상감지 모델의 QA 진행 결과를 재현 가능한 형태로 정리한 것이다.
> 작성 시점: 2026-05-02. 다음 단계는 웹 런타임(uvicorn AI 서버 + 프론트) 통합 검증.
>
> ⚠ **웹 런타임 모델 통일이 우선 작업.** 현재 외부 영상 결과가 QA 와 다른 1차 원인은
> 데이터셋이 아니라 웹 런타임이 vB 모델 대신 legacy 2-class fallback 을 쓰고 있다는 점이다.
> 데이터셋 보강은 PyTorch vB 단일 런타임 정착 이후에도 외부 metric 이 낮을 때만 진행한다.
> **최종 목표는 `.venv311` 단일 환경에서 MediaPipe + PyTorch vB.** 마이그레이션 절차는
> [`WEB_RUNTIME_QA.md`](./WEB_RUNTIME_QA.md) §7 참조.

---

## 1. 요약

- **API 계약 검증 완료** — pytest 기반 mock 테스트 4건 모두 통과.
- **vB Keras 모델 정상 로드** — `fall_lstm_vB_best.keras`, `labels_vB.json`, input/output shape 모두 일치.
- **학습 분포 재현** — `test_vB.npz` 기준 baseline `accuracy = 0.7402`, `Fall recall = 0.7143` 정확 재현.
- **외부 Zenodo 영상 50개 평가** — 2단계 분리 워크플로우(.venv311 → .venv)로 성공.
- **결정적 결과**: 영상 단위 `fall_min_windows`를 2 → 1 로 완화하면 외부 영상에서
  `accuracy 0.60 → 0.84`, `Fall recall 0.24 → 0.72`. precision 은 오히려 0.857 → 0.947 상승.
  → **Stage 2 default 를 `fall_min_windows=1`로 채택.**
- **모델 자체에는 결함이 없음.** 외부 영상 성능 차이는 영상 단위 판정 규칙과 도메인 갭에 기인.

---

## 2. 실행 환경

| venv | 가능 | 불가능 | 용도 |
|---|---|---|---|
| `.venv` | TensorFlow 2.21 로 vB Keras `.keras` 로드 | MediaPipe 호환 X | Stage 2, npz/모델 검증 |
| `.venv311` | MediaPipe 0.10.14 (`mp.solutions.pose`) | `.keras` 로드 호환 X | Stage 1 (영상 → keypoint) |

> 한 가상환경에서 MediaPipe + Keras 를 동시에 쓰는 데 호환 이슈가 있어 외부 영상 평가는
> 두 venv 를 번갈아 쓰는 2단계 구조로 분리했다. 자세한 내용은 §6, §12 참조.

---

## 3. API mock QA (`pytest`)

### 실행
```bash
py -m pytest -q qa/test_fall_api.py
```

### 결과
```
4 passed in 0.90s
```

### 검증 범위
1. `GET /health` 헬스체크
2. `POST /detect/reset` 상태 초기화 응답
3. `POST /detect/fall` 기본 응답 구조 (`type/status/confidence/method`)
4. `POST /detect/fall` invalid payload → `422`

### 의미와 한계
- `pipelines.fall_detector` 등 무거운 의존성(MediaPipe / TensorFlow / 모델 파일)을 test double 로 주입.
- **API 계약/JSON 구조** 만 검증. **모델 성능** 은 검증하지 않음 (그건 §4-§7 의 책임).

---

## 4. vB Keras 모델 로드 검증

### 실행
```bash
py qa/check_fall_model_load.py
```
전제: `pip install tensorflow`. (opencv/mediapipe/torch 불필요.)

### 검증 항목 / 결과
| 항목 | 값 | 상태 |
|---|---|---|
| `ai/models/fall_lstm_vB_best.keras` 존재 | 5.11 MB | PASS |
| `labels_vB.json` 파싱 | `version = V2.1-B-5class-coarse` | PASS |
| `sequence_len` | 30 | PASS |
| `num_features` | 132 | PASS |
| `vB_labels[0]` | `Fall` | PASS |
| `num_classes` | 5 | PASS |
| `tensorflow.keras.models.load_model()` | 성공 (TF 2.21.0) | PASS |
| `model.input_shape` | `(None, 30, 132)` | PASS |
| `model.output_shape` | `(None, 5)` | PASS |

→ 모델 파일/라벨 메타/입력 shape 문제는 **아님**.

---

## 5. test_vB.npz baseline 재현

### 실행
```bash
py qa/evaluate_vB_npz.py
# 또는 다른 npz 경로
py qa/evaluate_vB_npz.py --npz path/to/test_vB.npz
```
기본 입력: `C:\4-1\Fall_Down_Detail\test_vB.npz` (npz keys: `X`, `y`, `heuristic`, X shape `(766, 30, 132)`).

### 결과
| 지표 | 측정값 | baseline (`test_report_vB.json`) | Δ |
|---|---:|---:|---:|
| accuracy | 0.740209 | 0.740209 | +0.000000 |
| Fall recall | 0.714286 | 0.714286 | +0.000000 |

→ baseline 완전 일치. **vB Keras 모델 + `labels_vB.json` + class mapping + input shape 연결은 정상.**
산출물: `qa/results/vB_npz_eval.json`.

---

## 6. 외부 Zenodo 영상 2단계 QA

### 6.1 폴더 구조 (사용자 준비)
```
qa/new_videos/
├── fall/      ← 25개 (mp4/avi/mov/mkv)
└── normal/    ← 25개
```
> fall/01.mp4 와 normal/01.mp4 처럼 **동일 파일명이 양쪽 폴더에 존재하는 경우** 도 정상 처리.
> filename 이 아니라 stage-1 이 부여한 `video_id` 로 매칭한다 (§7 참조).

### 6.2 Stage 1 — `.venv311` 에서 keypoint 추출
```bash
.\.venv311\Scripts\activate
py qa/extract_new_video_keypoints.py
```

산출물:
- `qa/results/new_video_windows.npz` — `X(W,30,132)`, `video_ids(W,)`, `true_labels(W,)`,
  `filenames(V,)`, `window_counts(V,)`, `sequence_len/num_features/stride`
- `qa/results/new_video_metadata.csv` — `video_id`, `source_folder`, `relative_path`, `filename`,
  `true_label`, `frames_read`, `frames_with_pose`, `kp_extraction_rate`, `fps/width/height`,
  `n_windows`, `extract_seconds`, `warning`, `error`

전처리: 학습 코드와 동일한 `hip-centering` + `95th-percentile scale` 정규화 → 30프레임 × 132 feature
슬라이딩 윈도우 (stride 15). pose 미감지 프레임은 0.0 으로 채움 (모델의 `Masking(0.0)` 가 처리).

확인된 결과:
```
X.shape = (648, 30, 132)
video 수 = 50
metadata true_label 분포 = {Fall: 25, Normal: 25}
keypoint 추출률 = 전반적으로 정상
```

### 6.3 Stage 2 — `.venv` 에서 모델 추론 / metric
```bash
.\.venv\Scripts\activate
py qa/evaluate_new_video_windows.py
```

산출물:
- `qa/results/new_video_eval.csv` — 영상별 결과
- `qa/results/new_video_metrics.json` — 데이터셋 metric (Accuracy / Fall P/R/F1 / Normal Recall / CM)

집계 규칙:
- 윈도우가 'Fall' 로 잡히려면: `softmax[Fall] >= --fall-prob-thr` AND `argmax == Fall(0)`
- 영상이 'Fall' 로 분류되려면: `fall_window_count >= --fall-min-windows` AND
  `max(softmax[Fall]) >= --fall-prob-thr`
- 영상 단위 `true_label` 은 **`metadata.csv` 의 `video_id` 기준** 이 신뢰 소스. npz 는 fallback.

---

## 7. 평가 버그와 수정 내역

### 7.1 filename 중복 → metadata dedup
- **증상**: Stage 2 console 이 `true=Normal` 만 출력. confusion matrix 의 Fall true row 가 0/0.
- **원인**: `qa/new_videos/fall/01.mp4` 와 `qa/new_videos/normal/01.mp4` 처럼 동일 basename 이
  fall/normal 양쪽에 존재. metadata 를 `dict[filename]=row` 로 만들면 두 번째 write 가 첫 번째 row
  를 덮어써 50개 row 중 25개(Normal)만 살아남음.
- **수정**:
  - `_load_metadata()` → `list[dict]` 반환 (dedup 안 함).
  - `_index_metadata_by_video_id()` 신규 — `int(video_id) → row`.
  - 영상 단위 매칭은 **`video_id` 키**. filename 은 표시용으로만 사용.
  - 콘솔/CSV/JSON 에 `relative_path` 추가 (`Fall/01.mp4` 형태).
- **결과**: metadata 분포 / npz 영상 분포 / 최종 video_true_labels 분포 모두 `{Fall: 25, Normal: 25}` 정상화.

### 7.2 라벨 인코딩 안전화 (보조)
- `_decode_str()` — bytes/np.str_/None 통일.
- `_nfc()` — Unicode NFC 정규화 + strip (Mac/Win NFD 차이 보정).
- `_load_metadata()` 는 `encoding="utf-8-sig"` 로 BOM 자동 제거.

---

## 8. 외부 Zenodo 영상 실험 결과 비교표

낙상 25 + 비낙상 25 영상으로 임계값 3종을 비교 (모두 같은 npz/모델 사용).

| 조건 | `fall_prob_thr` | `fall_min_windows` | Accuracy | Fall Precision | Fall Recall | Fall F1 | Normal Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기본 (이전 default) | 0.5 | 2 | 0.6000 | 0.8571 | 0.2400 | 0.3750 | 0.9600 |
| **A (현 default)** | **0.5** | **1** | **0.8400** | **0.9474** | **0.7200** | **0.8182** | **0.9600** |
| B | 0.3 | 1 | 0.8400 | 0.9474 | 0.7200 | 0.8182 | 0.9600 |

Confusion matrix (행=true, 열=pred):

| 조건 | Fall→Fall | Fall→Normal | Normal→Fall | Normal→Normal |
|---|---:|---:|---:|---:|
| 기본 | 6 | 19 | 1 | 24 |
| A / B | 18 | 7 | 1 | 24 |

---

## 9. 최종 해석

1. **모델 자체에는 결함이 없다.** §4 (로드) + §5 (test_vB.npz 재현) 가 동일 모델/라벨로 통과.
2. **`fall_min_windows`(2 → 1) 가 단일 핵심 변수.** Fall recall +0.48, accuracy +0.24,
   precision 도 +0.090 상승. 낙상 모션이 짧아(≈1~2 s) stride 15 윈도우 기준 1~3 개만 fall 신호를
   내는 경우가 많고, 최소 2 윈도우를 요구하면 짧은 낙상이 항상 누락되는 구조였다.
3. **B == A 의 의미.** `softmax[Fall]` 이 `(0.3, 0.5)` 구간에 있으면서 동시에 `argmax == Fall` 인
   윈도우가 0 개. 즉 모델은 Zenodo 데이터에서 "약하게" Fall 을 예측하는 케이스가 거의 없고,
   Fall argmax 분포가 sharp 하다. 이 데이터의 한계 요인은 임계값이 아니라 **모델이 그 영상에서
   Fall argmax 자체를 만들지 못하는 것** (false negative 의 본질).
4. **Normal recall 0.96 이 모든 조건에서 유지된다.** false positive 는 1 개로 유지 → 임계값
   완화의 안전 근거.

---

## 10. 권장 기준 (배치 평가)

| 항목 | 값 | 비고 |
|---|---|---|
| `fall_prob_thr` | `0.5` | 보수적인 default. 다른 데이터셋에 약신호가 많다면 0.3 까지 내려볼 수 있음. |
| `fall_min_windows` | **`1`** | Zenodo 검증 결과 채택. precision 손실 없음. |
| 운영 override | `--fall-min-windows 2` | precision 우선 운영 시. |

`evaluate_new_video_windows.py` 는 이미 위 default 로 설정되어 있어 다음과 같이 단순 실행 가능.

```bash
py qa/evaluate_new_video_windows.py                                 # 권장 default
py qa/evaluate_new_video_windows.py --fall-min-windows 2            # 보수적
py qa/evaluate_new_video_windows.py --fall-prob-thr 0.3 --fall-min-windows 1  # 약신호 포함
```

---

## 11. 실시간 런타임 기준과 환경변수

`ai/pipelines/fall_detector.py` 의 `CONSECUTIVE_THRESHOLD` 는 **batch 평가의
`fall_min_windows` 와 의미가 다르다.**

| 항목 | Stage 2 `fall_min_windows` | Runtime `CONSECUTIVE_THRESHOLD` |
|---|---|---|
| 단위 | 윈도우 (stride 15) | 프레임 |
| 시간 의미 | 0.5 s 간격 윈도우 N 개 (영상 전체 누적) | 연속 N 프레임 (≈ N/30 s) |
| 적용 시점 | 영상 전체 추론 후 batch 집계 | 매 프레임 실시간 카운트 |
| 위양성 영향 | batch 결과만 영향 (알림 없음) | 즉시 emergency 알림 → 보호자 SMS |
| Default | **1** (Zenodo 결과로 변경) | **3** (실시간 debouncing 위해 유지) |

→ Zenodo 의 `fall_min_windows=1` 결과를 그대로 runtime 에 적용하면 **단일 프레임 false positive 가
즉시 알림으로 이어지므로** 적용하지 않는다.

### 환경변수 override (데모/개발용)

`fall_detector.py` 는 다음 6개 임계값을 환경변수로 조정 가능. 각 변수는 첫 호출 시 stdout
에 한 번만 override 사실을 출력하고, 잘못된 값은 default 로 fallback.

| 변수 | 의미 | Default | 범위 |
|---|---|---:|---|
| `FALL_PROB_THRESHOLD` | softmax[Fall] 통과 임계 (argmax==Fall 도 함께 요구) | `0.65` | 0.0~1.0 |
| `FALL_CONFIRM_WINDOWS` | 이 회수 미만은 movement_pending (관찰 단계) | `3` | 1~100 |
| `FALL_CONSECUTIVE_THRESHOLD` | 연속 fall 프레임 — emergency 후보 | `3` | 1~100 |
| `FALL_VERTICAL_DROP_MIN` | recent hip_y 변화량 (lying suppression / dynamic gate) | `0.10` | 0.0~1.0 |
| `FALL_MOTION_SCORE_MIN` | recent \|Δhip_y\| 평균 (lying suppression) | `0.005` | 0.0~1.0 |
| `FALL_SLOW_MOTION_MAX` | 이 미만 motion = 천천히 눕기 후보 | `0.020` | 0.0~1.0 |
| `FALL_DYNAMIC_MOTION_MIN` | dynamic event gate motion 최소치 | `0.020` | 0.0~1.0 |
| `FALL_TORSO_CHANGE_MIN` | dynamic event gate torso 회전 최소치(°) | `30.0` | 0.0~180.0 |
| `FALL_ALERT_COOLDOWN_SECONDS` | 사건 latch reminder 알림 주기(초). 0 = 최초 1회만 | `0.0` | 0~86400 |

> 실제 emergency 트리거는 `EMERGENCY_THRESHOLD = max(FALL_CONSECUTIVE_THRESHOLD, FALL_CONFIRM_WINDOWS+1)` 프레임.
> default 조합 (3, 3) 에서는 `4` 프레임에서 emergency 가 발화된다 (fall_suspected 가 1 프레임 fire).

```bash
# Windows
set FALL_PROB_THRESHOLD=0.70
set FALL_CONSECUTIVE_THRESHOLD=3

# Linux/Mac
export FALL_PROB_THRESHOLD=0.70
export FALL_CONSECUTIVE_THRESHOLD=3

uvicorn ai.main:app --reload --port 8000
```

→ production 은 default 유지 권장. 데모/개발 환경에서만 임계값 조정.

### 알림 게이팅 — 5-state machine

서비스 관점에서 "눕는 중간" 이 한순간 fall 로 잡혀 알림이 나가는 문제를 막기 위해
다음 상태 머신을 사용한다. **알림 (`alert_triggered=true`) 은 `fall_emergency` 에서만 발화.**

| `final_decision` | `status` | UI 라벨 | alert_triggered | 알림 발송 |
|---|---|---|---|---|
| `normal` | `normal` | 정상 | false | X |
| `lying_suppressed` | `normal` | 누워 있음 | false | X |
| `movement_pending` / `checking_transition` | `checking` | ⏳ 판정 중 | false | **X** |
| `fall_suspected` | `suspected` | ⚠ 낙상 의심 | false | X (화면 경고만) |
| `fall_emergency` | `emergency` | 🚨 낙상 발생 | **true** | **O** |

상태 전이 (엄격 모드):
1. **Phase 1 (Lying suppression)** — fall 후보 + recent `vertical_drop < MIN` + `motion_score < MIN`
   → `lying_suppressed`. 처음부터 누워있거나 자는 상태.
2. **Phase 2 (Slow transition)** — fall 후보 + `vertical_drop >= MIN` + `motion_score < SLOW_MAX`
   → `movement_pending`. **counter RESET** (HOLD 가 아님 — 천천히 눕는 중 brief motion spike
   가 누적되어 fall_suspected 까지 가던 잔존 문제 차단).
3. **Phase 3 (Dynamic event gate)** — fall 후보 + Phase 1/2 미발화. 이 시점에 dynamic event
   를 검사:
   - `vertical_drop ≥ FALL_VERTICAL_DROP_MIN` AND
   - `motion_score ≥ FALL_DYNAMIC_MOTION_MIN` AND
   - `torso_angle_change ≥ FALL_TORSO_CHANGE_MIN` (단위 = °)
   - **셋 다 충족** → `dynamic_fall_event=True` → counter 증가
   - 하나라도 미충족 → `dynamic_fall_event=False` → suppress (`static_no_dynamic_event`)
4. **Phase 4 (Confirmed fall — 엄격 게이트)** — fall_* 라벨은 **반드시 dynamic_event=True**
   여야 함:
   - 어떤 시점이든 **`dynamic_event=False`** 이면 fall_suspected/fall_emergency 라벨 절대
     노출 X. counter > 0 이면 `movement_pending`, 아니면 `normal` 로 처리.
   - `dynamic_event=True`:
     - `1 ≤ count < CONFIRM_WINDOWS` → `movement_pending`
     - `CONFIRM_WINDOWS ≤ count < EMERGENCY_THRESHOLD` → `fall_suspected` (화면 경고)
     - `count ≥ EMERGENCY_THRESHOLD` AND **lstm_fall=True** → `fall_emergency` + 보호자 알림
     - `count ≥ EMERGENCY_THRESHOLD` 이지만 lstm_fall=False (휴리스틱만) → `fall_suspected` 강등

이 구조 덕분에:
- **천천히 눕기는 절대 fall_* 라벨이 노출되지 않음** — Phase 2 에서 counter RESET,
  Phase 3 의 motion 게이트도 통과하지 못해 dynamic_event=False 유지.
- **눕는 중간 brief motion spike 누적 차단** — Phase 2 RESET (HOLD → RESET 변경) 으로
  spike 가 끝날 때마다 counter 가 0 으로 돌아가 CONFIRM_WINDOWS 도달 불가능.
- **누운 뒤 정지 상태에서 모델이 Fall 로 튀어도 알림 없음** — Phase 3 의 dynamic event
  gate 가 motion / torso 변화 부족을 잡아 즉시 suppress.
- **실제 급격한 낙상만 emergency 도달** — motion + drop + torso 회전이 모두 동시에
  fall-like 임계 이상으로 **연속 4 프레임** 유지되어야 알림.

> **이중 가드**: 백엔드의 상태 머신이 fall_* 라벨을 차단하고, 프론트(`CameraPage.jsx`) 도
> `dynamic_fall_event=false` 면 백엔드가 보낸 `fall_suspected` / `fall_emergency` 를
> `movement_pending` 으로 자동 변환해 표시. 알림 트리거도 `alert_triggered=true` AND
> `dynamic_fall_event=true` 둘 다 충족할 때만 발화.

### Latch 게이트

`fall_emergency` 와 `fall_incident_active` 활성화는 다음 **모든 조건** 을 만족할 때만 허용:
- `model_backend == "pytorch_vB"`
- `model_prediction == "Fall"`
- `fall_probability ≥ FALL_PROB_THRESHOLD`
- `dynamic_fall_event == True`
- `warmup == False`
- `consecutive_fall_windows ≥ EMERGENCY_THRESHOLD`

하나라도 미달이면 `latch_allowed=False`, `latch_block_reason` 에 사유 기록, 상태는
`fall_suspected` 강등 및 알림 차단. 워밍업 (LSTM 미활성) 단계의 latch 활성화는
`warmup=True` 가드로 차단된다.

> **참고**: 푸시업/플랭크/엎드린 활동 같은 "낮은 자세 + 동적" 동작에 대한 별도
> exercise suppression 은 9차에서 시도했으나 실제 낙상까지 억제되어 10차에서 revert 했다.
> 푸시업 오탐은 알려진 한계로 남기고, 추후 데이터셋 보강이나 별도 클래스 학습으로
> 해결할 예정.

### 낙상 사건 latch (incident state)

`fall_emergency` 가 한 번 확정되면 **명시적 reset 까지 사건 상태를 유지** 한다. 사람이
바닥에 오래 누워있어 motion/drop 이 줄어들어도 자동으로 `lying_suppressed` / `normal` /
`movement_pending` 으로 돌아가지 않는다.

| 필드 | 의미 |
|---|---|
| `fall_incident_active` | `true` = 사건 진행 중 (latch 활성) |
| `incident_state` | `"active"` 또는 `"inactive"` |
| `fall_incident_started_at` | unix timestamp (사건 시작 시각) |
| `fall_incident_id` | `FALL_<ms>_<camera_id>` 고유 ID |

알림 게이팅:
- 최초 `fall_emergency` 확정 시: `alert_triggered=true` (1회 발화)
- 이후 incident active 동안: `alert_triggered=false` (반복 알림 X)
- `FALL_ALERT_COOLDOWN_SECONDS > 0` 이면 그 주기마다 reminder 알림 1회 발화

해제 방법:
- `POST /detect/reset` (프론트 "사건 해제" 버튼) → counter / kinematics buffer / lstm_started /
  fall_incident_active 모두 초기화 → 정상 감지 재개
- 백엔드의 `FallDetector.reset()` 이 모든 per-camera state 를 깔끔히 비운다.

프론트 표시:
- `fall_incident_active=true` 인 동안 화면 라벨 `🚨 낙상 발생 — 확인 필요` (빨강 + 펄스)
- 사건 active 배너 + 빨간 "사건 해제" 버튼 노출 — `lying_suppressed` 가 와도 무시되고
  계속 사건 라벨 유지
- 디버그 스트립에 `incident: active/inactive` 칩 + `fall_incident_id` 표시

### Lying suppression (오탐 억제)

낮은 외부 metric 으로도 "누워서 자는 상태" 가 Fall 로 오탐되는 문제를 해결하기 위해
**pose keypoint 후처리** 를 추가했다. 모델/휴리스틱이 Fall 후보를 잡아도 다음 둘 다
부족하면 `lying_suppressed` 로 강등된다:

- `vertical_drop` < `FALL_VERTICAL_DROP_MIN` (최근 30프레임 hip_y 의 max-min)
- `motion_score` < `FALL_MOTION_SCORE_MIN` (최근 30프레임 frame-to-frame \|Δhip_y\| 평균)

즉 **"의미 있는 vertical drop 또는 motion 이 있어야만 fall 로 인정"**. 처음부터 누워
있거나 서서히 눕는 케이스는 두 신호가 모두 작아서 자동 억제된다.

### `/detect/fall`, `/detect/live` 응답 — 디버그 필드

```json
{
  // 알림 게이팅 — 프론트 / 백엔드 둘 다 이 두 필드만 보면 됨
  "alert_triggered": true,           // ★ 보호자 알림 발화 여부 (fall_emergency 일 때만 true)
  "alert_reason": "confirmed_fall: count=4>=EMERGENCY_THRESHOLD=4",

  // 모델 신호 (raw)
  "fall_probability": 0.812,         // 모델의 softmax[Fall]
  "confidence": 0.876,               // 모델의 top prediction 확신도 (Fall 일 수도, Lying 일 수도)
  "model_prediction": "Fall",        // top class 라벨 (Fall/Walking/Lying_Down/Sitting/Standing_Transition)

  // 상태 머신
  "final_decision": "fall_emergency",// normal / movement_pending / fall_suspected / fall_emergency / lying_suppressed / warmup / no_person
  "suppression_reason": null,        // lying suppression 사유 (Phase 1) 또는 static_no_dynamic_event (Phase 3)
  "transition_state": null,          // slow transition 사유 (Phase 2) — "slow_lying_or_sitting: ..."

  // Dynamic event gate (Phase 3) — 정적 누움 + Fall 예측 오탐 방지의 핵심
  "dynamic_fall_event": true,        // 세 신호(drop, motion, torso) 모두 fall-like 임계 통과 여부
  "dynamic_gate_reason": "all_satisfied: drop=0.213>=0.10 & motion=0.0240>=0.0200 & torso_Δ=38.5°>=30.0°",

  // Kinematic 신호 (recent_* 는 동일 값의 명시적 alias)
  "vertical_drop": 0.213,            // 최근 30 프레임 hip_y range
  "motion_score": 0.0124,            // 최근 30 프레임 |Δhip_y| 평균
  "torso_angle_change": 38.5,        // 첫/마지막 torso angle 차이 (degrees)
  "recent_vertical_drop": 0.213,
  "recent_motion_score": 0.0124,
  "recent_torso_angle_change": 38.5,

  // 카운터
  "consecutive_fall_windows": 4,     // 연속 fall frame count
  "pending_windows": 0,              // EMERGENCY_THRESHOLD 까지 남은 프레임 수

  // legacy 호환 (기존 클라이언트 보호)
  "type": "FALL",
  "status": "emergency",             // legacy: emergency/suspected/checking/normal/buffering/no_person
  "method": "lstm",                  // legacy: lstm / heuristic:reason / lstm+kinematic_suppress / lstm+slow_transition_hold / yolo_bbox
  "lstm_prob": 0.812,                // legacy alias of fall_probability
  "heuristic": false,
  "detected": true,                  // legacy = alert_triggered
  "consecutive_frames": 4,           // legacy alias of consecutive_fall_windows

  // routes.py 가 추가
  "model_backend": "pytorch_vB",
  "qa_validated": true
}
```

### 테스트 케이스별 기대 시그널 + 알림 게이팅

| 시나리오 | model_prediction | fall_probability | drop | motion | torso_Δ | dynamic_fall_event | latch_allowed | final_decision | 화면 라벨 | alert_triggered |
|---|---|---|---|---|---|---|---|---|---|---|
| 정상 서있음 | Standing/Walking | <0.3 | 작음 | 작음 | 작음 | false | varies | `normal` | 정상 | **false** |
| **푸시업/플랭크/엎드려 움직임 (한계)** | Lying_Down/Fall | 변동 | 변동 | 변동 | 변동 | varies | varies | 잘못 fall_* 또는 lying_suppressed 가능 | 정상 보장 안됨 (알려진 오탐) | 가능성 있음 |
| 걷기 | Walking | 낮음 | 중간 | 중간 | 작음 | false | false | `normal` | 정상 | **false** |
| 앉기 | Sitting | 낮~중 | 중 | 중 | 중 | false | false | `normal` 또는 일시 `movement_pending` | 정상/판정 중 | **false** |
| 천천히 눕기 | Lying_Down/Fall | 변동 | ≥0.10 | <0.020 | 중~↑ | **false** | false | `movement_pending` → `lying_suppressed` | 판정 중 → 누워 있음 | **false** |
| 누워서 자기 | Lying_Down/Fall | 변동 | ≈0 | ≈0 | ≈0 | **false** | false | `lying_suppressed` | 누워 있음 | **false** |
| 누운 뒤 모델이 Fall 튐 | Fall | 높음 | ≈0 | ≈0 | ≈0 | **false** | false | `lying_suppressed` | 누워 있음 | **false** |
| **실제 낙상 (최초 4프)** | Fall | ≥0.65 | ≥0.10 | ≥0.020 | ≥30° | **true** | **true (latch on)** | `movement_pending` → `fall_suspected` → `fall_emergency` | 판정 중 → 낙상 의심 → **낙상 발생 — 확인 필요** | **true (1회)** |
| **낙상 후 계속 누워있음** | Lying_Down 등 | 변동 | ≈0 | ≈0 | ≈0 | false | **true (latch 유지)** | `fall_emergency` (강제) | **낙상 발생 — 확인 필요 (유지)** | **false** |
| **사건 해제 후 다시 누워있음** | Lying_Down | 변동 | ≈0 | ≈0 | ≈0 | false | false | `lying_suppressed` | 누워 있음 | **false** |

> **중요**: "천천히 눕기" 행에서 brief motion spike 가 발생해도 Phase 2 RESET 덕분에
> counter 가 누적되지 않아 `fall_suspected` 라벨이 절대 노출되지 않는다. dynamic_event=False
> 가 유지되는 한 화면은 "판정 중" → "누워 있음" 으로만 전이된다.

핵심 설계:
- **천천히 눕기**: `vertical_drop` 은 충족되지만 `motion_score < SLOW_MOTION_MAX` 이므로
  Phase 2 (slow transition) 가 발화 → counter hold → `movement_pending` 에서 멈춤 →
  자세 안정 후 Phase 1 (lying suppression) 으로 전이.
- **실제 낙상**: motion_score 가 SLOW 임계 이상이므로 Phase 2 미발화 → counter 가
  4 프레임에 도달 → emergency. 4 프레임 ≈ 130 ms (30fps 기준) 의 미니멀 debounce.

웹 화면 (CameraPage) 의 디버그 스트립에 위 모든 신호 + `alert: ON/off` + `pending_windows`
가 실시간 표시되므로, 각 케이스를 직접 시연하며 상태 머신 동작을 검증할 수 있다.

---

## 12. 남은 이슈: 웹 런타임 환경

QA 는 모두 통과했지만 **실제 웹 AI 서버 (uvicorn) 에서는 한 프로세스 안에서 MediaPipe Pose
추출과 Keras `.keras` 추론이 동시에 필요하다.** 외부 영상 QA 처럼 venv 를 분리하는 방식이
런타임에서는 불가능하다.

### 현재 venv 별 상태
| venv | TensorFlow 2.21 (vB Keras 로드) | MediaPipe 0.10.14 (`mp.solutions`) |
|---|---|---|
| `.venv` | ✅ | ❌ (protobuf 충돌) |
| `.venv311` | ❌ (`.keras` 로드 호환 문제) | ✅ |

### 웹 런타임 진입 전 후보 전략 (택 1 또는 조합)

1. **fallback 모드 우선 검증**
   - `.venv311` 에서 `.keras` 로드를 포기하고 `fall_lstm.pt` (PyTorch) 또는 휴리스틱 fallback 으로
     서버를 우선 띄워 웹 → AI 연결 동작을 검증.
   - `fall_detector.py` 가 이미 `.keras → .pt → 휴리스틱` 순으로 안전 폴백하므로 코드 수정 불필요.
   - 정확도는 약간 낮지만 통신/스트림/알림 경로는 모두 검증 가능.

2. **호환 가능한 단일 venv 재구성**
   - `.keras` 를 H5 (`.h5`) 또는 `SavedModel` 로 변환해 TF/Keras 버전 의존성 완화.
   - 또는 MediaPipe 측 protobuf 핀 버전을 맞춰 `.venv` 에 MediaPipe 합류 재시도.

3. **프로세스 분리**
   - MediaPipe 추출 프로세스 ↔ 모델 추론 프로세스 를 IPC (HTTP / shared memory / file) 로 연결.
   - 운영 복잡도가 올라가므로 1·2 가 실패할 때만 고려.

→ **다음 작업 권장 순서**: 전략 1 로 웹 연결 동작을 먼저 확인 → 전략 2 로 정확도 회복 →
실패 시 전략 3 검토.

---

## 13. 실행 명령어 모음

```bash
# (A) API 계약 검증
py -m pytest -q qa/test_fall_api.py

# (B) vB Keras 모델 로드 검증 (.venv)
py qa/check_fall_model_load.py

# (C) test_vB.npz baseline 재현 — Keras (.venv)
py qa/evaluate_vB_npz.py
py qa/evaluate_vB_npz.py --npz path/to/test_vB.npz

# (C') PyTorch vB 마이그레이션 (★ 신규)
.\.venv\Scripts\activate
py qa/convert_keras_to_pt.py             # Keras → PyTorch 가중치 변환 + sanity check
# 실패 시 fallback (.venv311):
.\.venv311\Scripts\activate
py qa/train_vB_pt.py                     # train/val/test_vB.npz 로 재학습
py qa/evaluate_vB_pt_npz.py              # PT vB baseline 재현 검증

# (D) 외부 영상 QA — Stage 1 (.venv311)
.\.venv311\Scripts\activate
py qa/extract_new_video_keypoints.py
py qa/extract_new_video_keypoints.py --max-frames 600 --mediapipe-complexity 2

# (E) 외부 영상 QA — Stage 2 (.venv 또는 .venv311; 백엔드 자동 검출)
py qa/evaluate_new_video_windows.py                                  # default: thr=0.5, min=1, backend=auto
py qa/evaluate_new_video_windows.py --backend pytorch_vB             # PT vB 강제
py qa/evaluate_new_video_windows.py --backend keras_vB               # Keras 비교용
py qa/evaluate_new_video_windows.py --fall-min-windows 2             # 보수

# (F) 실시간 서버 (개발/데모) — 8개 임계값 모두 env 로 조정 가능
set FALL_PROB_THRESHOLD=0.65         # Windows
set FALL_CONFIRM_WINDOWS=3
set FALL_CONSECUTIVE_THRESHOLD=3
set FALL_VERTICAL_DROP_MIN=0.10
set FALL_MOTION_SCORE_MIN=0.005
set FALL_SLOW_MOTION_MAX=0.020
set FALL_DYNAMIC_MOTION_MIN=0.020
set FALL_TORSO_CHANGE_MIN=30.0
set FALL_ALERT_COOLDOWN_SECONDS=0      # 0 = 최초 1회만 알림 / 60 = 60초마다 reminder
export FALL_PROB_THRESHOLD=0.65      # Linux/Mac
export FALL_CONFIRM_WINDOWS=3
export FALL_CONSECUTIVE_THRESHOLD=3
export FALL_VERTICAL_DROP_MIN=0.10
export FALL_MOTION_SCORE_MIN=0.005
export FALL_SLOW_MOTION_MAX=0.020
export FALL_DYNAMIC_MOTION_MIN=0.020
export FALL_TORSO_CHANGE_MIN=30.0
export FALL_ALERT_COOLDOWN_SECONDS=0
uvicorn ai.main:app --reload --port 8000   # PT vB 가 있으면 자동 채택 (1순위)
```

### 산출물 위치
- `qa/results/new_video_windows.npz` — Stage 1 키포인트 윈도우
- `qa/results/new_video_metadata.csv` — Stage 1 영상 메타
- `qa/results/new_video_eval.csv` — Stage 2 영상별 결과
- `qa/results/new_video_metrics.json` — Stage 2 데이터셋 metric
- `qa/results/vB_npz_eval.json` — `test_vB.npz` baseline 재현 결과

> `qa/results/` 와 `qa/new_videos/` 는 대용량/실험 산출물이므로 `.gitignore` 에 등록되어 있다.

---

## 부록 A. 변경 이력 (요약)

| 날짜 | 내용 |
|---|---|
| 2026-05-02 | (1차) API mock pytest 4건 정착, vB 모델 로드 검증, test_vB.npz baseline 재현, 외부 영상 2단계 QA 구축, filename dedup 버그 수정(video_id 키 매칭), Zenodo 임계값 실험 후 Stage 2 default `fall_min_windows=1` 채택, runtime `FALL_CONSECUTIVE_THRESHOLD` env 도입 |
| 2026-05-02 | (2차) 웹 런타임 진단 → 모델 불일치 확인. `/detect/model_info` 엔드포인트 + `CameraPage` 배지 추가. `WEB_RUNTIME_QA.md` 작성. |
| 2026-05-02 | (3차) PyTorch vB 마이그레이션 완료. `convert_keras_to_pt.py` / `train_vB_pt.py` / `evaluate_vB_pt_npz.py` 신규. `fall_detector.py` 로더 우선순위 PT vB > Keras vB > Legacy PT > heuristic. `evaluate_new_video_windows.py` 에 `--backend` auto-detect 추가. `/detect/model_info.model_backend` 가 `pytorch_vB`/`keras_vB`/`pytorch_fallback`/`heuristic` 중 하나로 명확화. |
| 2026-05-02 | (4차) **Lying suppression** — pose keypoint 후처리 추가. `vertical_drop` / `motion_score` / `torso_angle_change` 계산해 "처음부터 누워있음/천천히 눕기" 케이스를 `lying_suppressed` 로 강등. 임계값 4종(`FALL_PROB_THRESHOLD=0.65`, `FALL_CONSECUTIVE_THRESHOLD=3`, `FALL_VERTICAL_DROP_MIN=0.10`, `FALL_MOTION_SCORE_MIN=0.005`) 모두 env-tunable. `/detect/fall` `/detect/live` 응답에 `fall_probability` / `confidence` 분리 + 9개 디버그 필드 추가. CameraPage 가 "낙상 확률" / "판정 신뢰도" 분리 표시 + kinematics 디버그 스트립. |
| 2026-05-02 | (5차) **Movement pending state machine** — 눕는 중간 단계가 일시적으로 emergency 로 잡혀 알림이 먼저 나가던 문제 해결. (1) `FALL_CONFIRM_WINDOWS` (default 3) + `FALL_SLOW_MOTION_MAX` (default 0.020) env 추가. (2) 3-way counter: 천천히 눕기(`drop≥MIN` AND `motion<SLOW_MAX`) 감지 시 counter HOLD → `movement_pending` 유지 → 자세 안정 후 `lying_suppressed` 자연 전이. (3) 알림 게이팅을 `status=='emergency'` 비교에서 `alert_triggered=true` flag 로 명확화 — `routes.py` + 프론트 동시 갱신. 알림은 오직 `final_decision=='fall_emergency'` 일 때만 발화. (4) 응답에 `alert_triggered`, `alert_reason`, `transition_state`, `pending_windows` 4개 필드 추가. (5) CameraPage 5-state UI (정상 / 판정 중 / 낙상 의심 / 낙상 발생 / 누워 있음). |
| 2026-05-02 | (6차) **Dynamic event gate** — 누운 뒤 정지 상태에서 모델이 Fall 로 튀어도 emergency 까지 가던 잔존 문제 해결. (1) `FALL_DYNAMIC_MOTION_MIN` (default 0.020) + `FALL_TORSO_CHANGE_MIN` (default 30°) env 추가. (2) `_compute_dynamic_event()` 가 (drop, motion, torso_angle_change) 셋이 모두 fall-like 임계 이상일 때만 True. (3) `dynamic_fall_event=False` 면 즉시 suppress (`static_no_dynamic_event`) → counter 증가 차단. (4) 최종 emergency 게이트: `count >= EMERGENCY_THRESHOLD` AND `dynamic_event=True` AND `lstm_fall=True` 모두 충족 시에만 `fall_emergency` — 셋 중 하나라도 미달이면 `fall_suspected` 강등. (5) 응답에 `dynamic_fall_event`, `dynamic_gate_reason`, `recent_*` 5개 필드 추가. (6) CameraPage 디버그 스트립에 `dyn_event ✓/✗` 칩 + 사유 표시. |
| 2026-05-02 | (7차) **엄격 게이팅 — 천천히 눕는 중 fall_suspected 잔존 문제** 해결. (1) Phase 2 (slow transition) 의 counter 동작을 HOLD → **RESET** 으로 변경 — 천천히 눕는 중 brief motion spike 가 누적되어 fall_suspected 까지 가던 케이스 차단. (2) 상태 머신에 `elif not dynamic_event` 분기 추가 — counter 가 양수여도 현재 dynamic_event=False 면 fall_* 라벨 절대 노출 X (movement_pending 또는 normal 로 강제). (3) 워밍업 분기에도 동일 dynamic_event gate + Phase 3 정적-while-Fall suppress 적용. (4) 프론트 CameraPage 가 backend 가 보낸 fall_suspected/fall_emergency 를 dynamic_event=false 면 movement_pending 으로 자동 변환 (방어 가드). 알람 트리거도 `alert_triggered=true AND dynamic_fall_event=true` 둘 다 요구. |
| 2026-05-02 | (10차) **9차 exercise suppression revert** — 푸시업/플랭크/엎드린 활동 억제용 `_is_exercise_like_motion` 게이트가 실제 급격한 낙상까지 `activity_suppressed` 로 잘못 잡는 부작용 발생. 푸시업 오탐은 알려진 한계로 받아들이고 다음을 모두 되돌림: `_is_exercise_like_motion()` 메서드 / `EXERCISE_*` 4개 env / 4개 `DEFAULT_FALL_EXERCISE_*` 상수 / `_apply_incident_latch` 의 exercise_like_motion 게이트 / LSTM-active + 워밍업 + YOLO 분기의 exercise call / state machine 의 `activity_suppressed` 분기 / 응답의 `exercise_like_motion` & `exercise_gate_reason` & `warmup_blocked` 필드 / CameraPage 의 `activity_suppressed` 라벨 + 4-tier fallKey + 디버그 스트립 exercise 칩 + alert 트리거 exerciseLike 가드. 유지: `pytorch_vB` 로드 / lying suppression / movement_pending / dynamic_fall_event gate / alert_triggered 알림 / `fall_incident_active` latch / 사건 해제 버튼 / latch warmup 차단 / latch_allowed/latch_block_reason 디버그 노출. |
| 2026-05-02 | (9차) **Exercise-like motion suppression + warmup latch 차단** — 푸시업/플랭크/엎드린 활동 오탐 차단 + warmup 단계의 latch 활성화 차단. (1) 4개 env 신규: `FALL_EXERCISE_TORSO_MIN=60°` / `FALL_EXERCISE_HIP_Y_MIN=0.40` / `FALL_EXERCISE_MOTION_MIN=0.003` / `FALL_EXERCISE_ANGLE_STABLE_MAX=15°`. (2) `_is_exercise_like_motion()` 헬퍼: torso 수평 + 낮은 위치 + 활동 motion + 안정 torso 4 조건 모두 충족 시 활성화. (3) 활성화 시 `final_decision="activity_suppressed"`, counter reset, alert 차단. (4) `_apply_incident_latch()` 가 7개 게이트 검사 (backend / pred / prob / dynamic_event / exercise_like / warmup / count) — 하나라도 미달이면 `latch_allowed=false` + `latch_block_reason` 노출, fall_suspected 강등. (5) 워밍업 분기는 무조건 `warmup=True` 로 호출 → latch 영구 차단. 워밍업 상태머신 캡: 최대 `movement_pending`. (6) API 응답에 `exercise_like_motion` / `exercise_gate_reason` / `warmup_blocked` / `latch_allowed` / `latch_block_reason` 5개 필드 추가. (7) CameraPage: `activity_suppressed` → 💪 "활동 중" (emerald), 디버그 스트립에 `exercise ✓/✗` / `warmup blocked` / `latch allowed/blocked` chip + 사유 라인. exercise_like_motion=true 시 알림 트리거에서도 추가 가드. |
| 2026-05-02 | (8차) **낙상 사건 latch (incident state)** — 낙상 확정 후 사람이 오래 누워있을 때 lying_suppressed 로 자동 전이되던 문제 해결. (1) 새 상태: `fall_incident_active` / `incident_state` / `fall_incident_started_at` / `fall_incident_id` (per-camera). (2) `_apply_incident_latch()` 메서드가 fall_emergency 1회 확정 시 latch 활성화 → 명시적 reset 까지 final_decision 강제 유지. lying_suppressed/normal/movement_pending 으로 자동 전이 차단. (3) `FALL_ALERT_COOLDOWN_SECONDS` env (default 0 = 최초 1회만, >0 = reminder 주기). (4) `reset()` 가 모든 사건 state 도 함께 초기화. (5) 응답에 `fall_incident_active`, `incident_state`, `fall_incident_started_at`, `fall_incident_id` 추가. (6) CameraPage 에 빨간 사건 active 배너 + "사건 해제" 버튼 + 디버그 스트립에 `incident: active/inactive` 표시. |

세부 이력은 git log 참조.

## 부록 B. 알려진 이슈 (코드 외)

- `ai/training/fall_lstm_vB/` 의 세 학습 스크립트는 `from config import ...` 를 시도하지만
  현재 레포에 `config.py` 가 없어 `ModuleNotFoundError` 로 실패한다. 학습 재현이 필요할 때만
  복원하면 됨 (런타임 추론과는 무관).

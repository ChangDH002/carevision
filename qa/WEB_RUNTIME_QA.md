# CareVision — Web Runtime QA Status

> 본 문서는 **실제 웹 런타임(uvicorn AI 서버 + 프론트)** 에서 사용 중인 낙상감지 모델의
> 현황과, QA 단계에서 검증한 모델과의 차이를 명시한다.
>
> 자매 문서: [`README_QA.md`](./README_QA.md) — 배치 평가 / 학습 분포 재현 결과 (vB Keras 기준).

---

## 1. 핵심 결론

> **2026-05-02 업데이트**: PyTorch vB 변환/재학습 도구 + fall_detector 로더 갱신을 추가.
> 운영자가 `qa/convert_keras_to_pt.py` 또는 `qa/train_vB_pt.py` 로 `fall_lstm_vB.pt` 를
> 만들면 웹 런타임이 자동으로 그것을 사용한다 (1순위). 아래 표의 두 번째 컬럼은
> 변환/재학습 전의 상태고, 세 번째 컬럼이 목표 상태다.

| 항목 | QA 단계 (배치 평가) | 웹 런타임 (변환 전) | 웹 런타임 (PyTorch vB 적용 후 — 목표) |
|---|---|---|---|
| 모델 | `fall_lstm_vB_best.keras` | `fall_lstm.pt` (legacy) | `fall_lstm_vB.pt` |
| backend | keras_vB | pytorch_fallback | **pytorch_vB** |
| sequence_len | 30 | 15 | 30 |
| num_classes | 5 | 2 | 5 |
| Fall class index | 0 | 1 | 0 |
| 검증 정확도 | `accuracy 0.7402`, `Fall recall 0.7143` | **미검증** | baseline 재현 가능 |
| qa_validated | ✅ | ❌ | ✅ (체크 후) |

→ **현재 데이터셋을 바꾸기보다 웹 런타임 모델 통일이 1차 원인**. 데이터셋 보강은
PyTorch vB 단일 런타임 정착 이후 외부 영상 metric 이 여전히 낮을 때 진행한다.
**최종 목표: `.venv311` 단일 환경에서 MediaPipe + PyTorch vB.**

---

## 2. 현재 상태 (확인 로그)

uvicorn AI 서버 시작 시 stdout:
```
[FallDetector] LSTM 모델 로드 시작 — 우선순위: vB Keras > PT > 휴리스틱
[FallDetector] [FAIL] Keras (vB) 로드 실패 — PT 폴백 시도: ...
[FallDetector] [OK] PyTorch 모델 로드 완료 — path=C:\carevision\ai\models\fall_lstm.pt, window_size=15, num_classes=2, fall_class=1
```

Keras 로드 실패의 원인 메시지 (Windows 환경):
```
[WinError 267] 디렉터리 이름이 올바르지 않습니다:
  'C:\Users\...\AppData\Local\Temp\...\model.weights.h5'
```

`/detect/model_info` 엔드포인트(신규) 응답:
```json
{
  "model_backend": "pytorch_fallback",
  "model_path": "C:\\carevision\\ai\\models\\fall_lstm.pt",
  "version": "pt-2class",
  "sequence_len": 15,
  "num_features": 132,
  "num_classes": 2,
  "fall_class": 1,
  "fallback_reason": "vB Keras(.keras) 로드 실패 — PyTorch fallback 사용 중. QA 검증 결과(0.74 acc / 0.71 Fall recall)는 keras_vB 기준이며, 현재 런타임 성능은 다름.",
  "keras_fail_reason": "OSError: [WinError 267] 디렉터리 이름이 올바르지 않습니다: ...\\model.weights.h5",
  "pt_fail_reason": null,
  "qa_validated": false
}
```

---

## 3. Keras 로드 실패 원인 — Keras / TensorFlow 버전 호환 문제

`fall_detector.py` 의 `_build_and_load_model()` 은 `.keras` 파일을 zip 으로 열어
`model.weights.h5` 를 임시 디렉토리로 추출한 뒤 `model.load_weights()` 로 가중치를 복원한다:

```python
with zipfile.ZipFile(_KERAS_MODEL_PATH) as z:
    with tempfile.TemporaryDirectory() as tmpdir:
        z.extract("model.weights.h5", tmpdir)
        model.load_weights(os.path.join(tmpdir, "model.weights.h5"))
```

웹 런타임 환경에서는 다음 조합으로 위 경로가 깨진다:

- `.venv311` 의 TensorFlow / Keras 버전이 학습 시점(2026-04-22) 의 `.keras` zip 포맷과
  맞지 않아 `model.weights.h5` 추출 시 Windows API 가 `WinError 267` 을 반환.
- `.venv` 는 TF 2.21 로 `.keras` 로드 자체는 가능하지만 **MediaPipe 0.10.14 의 protobuf 와
  버전 충돌** 로 같은 프로세스에서 `mp.solutions.pose.Pose()` 가 import 단계에서 실패.

따라서 **한 venv 에서 MediaPipe + vB Keras 를 동시에 동작시킬 수 없는 것이 현재 환경의 한계**다.
이 충돌은 코드 결함이 아니라 라이브러리 의존성 문제다.

---

## 4. 실제 웹에서 QA 와 같은 성능을 내려면

다음 셋 중 하나가 필요하다.

### 옵션 A — vB Keras 를 `.venv311` 에서 로드 가능한 포맷으로 변환
- `.keras` → 단일 `.h5` (legacy) 또는 SavedModel 디렉토리로 export.
- TF/Keras 의 zip→임시 추출 경로를 우회 → Windows 임시 경로 이슈 회피 가능.
- 단, `.venv311` 의 Keras 버전이 학습 시 BiLSTM(128→64) 구조를 그대로 deserialize 할 수 있어야 함.
  실패 시 옵션 C 로 자동 진행.

### 옵션 B — MediaPipe 추출 ↔ 모델 추론 프로세스 분리
- MediaPipe (`.venv311`) 와 Keras (`.venv`) 를 별도 프로세스로 띄우고 IPC (HTTP / shared
  memory / named pipe) 로 keypoint 만 전달.
- 단일 venv 충돌은 회피하나 운영 복잡도 증가, 지연 증가, 디버깅 부담.

### 옵션 C — vB 5-class 모델을 PyTorch 로 재학습/변환하여 런타임 통일 ★ 권장
- `fall_lstm_vB_best.keras` 를 PyTorch `fall_lstm_vB.pt` 로 변환 (가중치 매핑) **또는**
  같은 데이터셋(test_vB.npz / labels_vB.json) 으로 PyTorch 재학습.
- `.venv311` 단일 환경에서 MediaPipe + PyTorch 가 모두 동작하므로 venv 충돌 자체가 사라진다.
- 기존 fallback 인 `fall_lstm.pt` (2-class) 와 **동일한 backend 종류** 이므로 `fall_detector.py`
  의 PT 로더는 그대로 재사용 가능 (`num_classes=5`, `window_size=30` 으로만 메타 변경).
- 마이그레이션 후 `/detect/model_info` 의 `model_backend` 가 `pytorch_vB` (또는 `keras_vB` 가
  계속 우선) 로 표시되도록 업데이트 필요.

---

## 5. 권장 마이그레이션 안

**옵션 C 채택을 권장한다.**

### 이유
1. **단일 venv (`.venv311`) 에서 동작** — 운영/배포가 가장 단순. 라즈베리파이 등 edge device 도 동일 환경.
2. **QA 검증 데이터셋(test_vB.npz / labels_vB.json) 을 그대로 재사용** 할 수 있어 baseline 재현 가능.
3. **`fall_detector.py` 의 backend 이미 PyTorch 로딩 경로를 갖고 있음** — 코드 변경 최소화.
4. **`.keras` zip 포맷 의존성 제거** — Windows 임시 경로 / Keras 버전 호환 이슈에서 해방.

### 작업 순서 (제안)
1. **변환 스크립트 작성**: `fall_lstm_vB_best.keras` 의 BiLSTM(128) → BiLSTM(64) → Dense(64) →
   Dense(5) 가중치를 PyTorch 동등 모듈로 매핑. (권장: 학습 데이터로 재학습이 더 안전)
2. `ai/models/fall_lstm_vB.pt` + `ai/models/fall_lstm_vB_meta.json` 생성. meta 는
   `{window_size: 30, input_dim: 132, hidden_dim: ..., num_layers: ..., num_classes: 5}`.
3. `fall_detector.py` 의 PT 로더 우선순위를 `(vB.pt, 5-class) > (legacy.pt, 2-class)` 로 정렬.
   `_PT_VB_MODEL_PATH` / `_PT_VB_META_PATH` 추가.
4. `qa/evaluate_vB_npz.py` 와 동일한 절차로 PyTorch vB 모델의 baseline 재현 (목표:
   `accuracy ≈ 0.7402`, `Fall recall ≈ 0.7143`).
5. 웹 런타임에서 `/detect/model_info.model_backend == "pytorch_vB"` 확인.
6. `qa/evaluate_new_video_windows.py` 도 새 PT 모델로 다시 실행하여 Zenodo metric 재산출.

### 작업하지 말 것
- runtime `CONSECUTIVE_THRESHOLD=3` 을 변경하지 말 것 (실시간 debouncing 의미. 자세한 내용은
  README_QA.md §11 참조).
- vB Keras `.keras` 파일을 삭제하지 말 것 — 향후 환경 정리(옵션 A) 시 재사용 가능.

---

## 6. 운영자가 즉시 확인 가능한 진단 도구

### 6.1 백엔드 (curl)
```bash
curl http://localhost:8000/detect/model_info
```
응답의 `model_backend` 가 `keras_vB` 가 아니면 **현재 런타임 결과는 QA metric 과 다르다.**

### 6.2 프론트엔드 배지
`CameraPage` 상단에 활성 backend 배지가 표시된다:
- 🟢 **QA 검증** — `keras_vB` 로드 성공
- 🟡 **폴백** — `pytorch_fallback` 또는 `heuristic` 사용 중. 호버 시 `fallback_reason` 표시.

배지 옆에 `seq=… · cls=… · fall=…` 으로 입력/출력 shape 와 Fall class index 도 함께 노출되어
QA 문서와 즉시 대조 가능.

### 6.3 detection 응답 자체
`/detect/fall` 와 `/detect/live` 응답에 `model_backend`, `qa_validated` 가 자동 첨부된다.
프론트가 별도 polling 없이도 매 프레임 응답에서 활성 backend 를 확인할 수 있다.

---

## 7. 마이그레이션 절차 (PyTorch vB 단일 런타임)

다음 절차를 따라 웹 런타임을 QA 검증 모델과 통일한다.

### 7.1 PyTorch vB 모델 생성

옵션 A — **직접 변환** (권장, .venv 사용 가능 시):
```bash
.\.venv\Scripts\activate
py qa/convert_keras_to_pt.py
```
- Keras vB 의 BiLSTM/Dense 가중치를 PyTorch 동등 모듈로 매핑.
- 무작위 입력으로 sanity check (`max|keras_softmax - pt_softmax| < 1e-3`).
- 통과 시 `ai/models/fall_lstm_vB.pt` + `fall_lstm_vB_meta.json` 저장.
- sanity check 실패 → 옵션 B 로 진행.

옵션 B — **재학습** (변환 실패 시 또는 .venv 에서 Keras 가 안 열릴 때):
```bash
.\.venv311\Scripts\activate  # 또는 PyTorch 가능한 venv
py qa/train_vB_pt.py
py qa/train_vB_pt.py --data-dir D:\custom\path  # 데이터 경로 다르면
```
- `train_vB.npz`, `val_vB.npz` 가 같은 디렉토리에 있어야 함.
- 동일한 모델 구조 (BiLSTM 128 → 64, Dense 64 → 5) 로 학습.

### 7.2 baseline 재현 검증
```bash
py qa/evaluate_vB_pt_npz.py
```
목표: `accuracy ≈ 0.7402`, `Fall recall ≈ 0.7143` 이내. ±0.05 초과 시 `qa_validated=false`
처리하고 원인 분석. 무리하게 적용하지 말 것.

### 7.3 외부 Zenodo metric 재산출 (선택)
```bash
py qa/evaluate_new_video_windows.py                       # auto: PT vB 우선
py qa/evaluate_new_video_windows.py --backend pytorch_vB
py qa/evaluate_new_video_windows.py --backend keras_vB    # 비교용
```
기존 `qa/results/new_video_windows.npz` 를 그대로 재사용 (Stage 1 재실행 불필요).
JSON 결과의 `backend.name` 으로 어떤 백엔드 결과인지 식별.

### 7.4 웹 서버 재시작 — PyTorch vB 자동 채택
`fall_detector.py` 의 로더 우선순위가 PT vB → Keras vB → Legacy PT → 휴리스틱 이므로
`fall_lstm_vB.pt` 만 존재하면 별도 설정 없이 자동으로 채택된다.
```bash
.\.venv311\Scripts\activate
uvicorn ai.main:app --reload --port 8000
```
첫 요청 시 stdout 에:
```
[FallDetector] [OK] PyTorch vB 모델 로드 완료 (V2.1-B-5class-coarse) — path=...fall_lstm_vB.pt, window_size=30, num_classes=5, fall_class=0
```
브라우저 `CameraPage` 의 배지가 🟢 "QA 검증" + `pytorch_vB` 로 변경되어야 정상.

### 7.5 데이터셋 보강 (런타임 통일 후 옵션)
PT vB 로 통일된 후에도 외부 영상 metric 이 여전히 낮으면 그제서야 데이터셋 보강
(외부 영상 추가 학습, 도메인 적응) 을 검토한다. 런타임 불일치 상태에서 데이터셋부터
바꾸는 것은 잘못된 진단을 유발하므로 피한다.

---

## 8. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-05-02 | (1차) 웹 런타임 PyTorch fallback 사용 사실 발견. `get_model_info()` + `/detect/model_info` + CameraPage 배지 추가. |
| 2026-05-02 | (2차) PyTorch vB 변환/재학습/검증 도구 추가 (`convert_keras_to_pt.py`, `train_vB_pt.py`, `evaluate_vB_pt_npz.py`). `fall_detector.py` 로더 우선순위 갱신 — PT vB → Keras vB → Legacy PT → 휴리스틱. `evaluate_new_video_windows.py` 에 `--backend` 자동 검출 추가. |
| 2026-05-02 | (3차) **Lying suppression 추가** — 실 서비스에서 누워서 자는 상태가 Fall 로 오탐되는 문제 해결. pose keypoint 후처리 (vertical_drop / motion_score / torso_angle_change) 로 "처음부터 누워있음 / 천천히 눕기" 강등. 임계값 4종 env-tunable (`FALL_PROB_THRESHOLD`, `FALL_CONSECUTIVE_THRESHOLD`, `FALL_VERTICAL_DROP_MIN`, `FALL_MOTION_SCORE_MIN`). API 응답에 `fall_probability` / `confidence` / `model_prediction` / `final_decision` / `suppression_reason` / kinematics 5종 추가. 자세한 사용법은 `README_QA.md` §11 참조. |
| 2026-05-02 | (4차) **Movement pending state machine** — 눕는 중간 단계 알림 방지. `FALL_CONFIRM_WINDOWS=3`, `FALL_SLOW_MOTION_MAX=0.020` env 추가. 3-way counter (inc/hold/reset) 로 천천히 눕기 감지 시 counter HOLD → emergency 까지 도달 못함. 알림 게이팅을 `alert_triggered` flag 로 분리 — `routes.py` 가 이 flag 만 보고 보호자 알림 발화 결정. fall_suspected 는 화면 경고만, fall_emergency 만 알림. CameraPage 5-state UI + 디버그 스트립에 alert/transition/pending 노출. 자세한 내용은 `README_QA.md` §11 참조. |
| 2026-05-02 | (5차) **Dynamic event gate** — 누운 뒤 정지 상태에서 모델이 Fall 로 튀어도 emergency 까지 가던 잔존 문제 해결. `FALL_DYNAMIC_MOTION_MIN=0.020` + `FALL_TORSO_CHANGE_MIN=30°` env 추가. drop AND motion AND torso_angle_change 셋 다 fall-like 임계 이상이어야 `dynamic_fall_event=True`. False 면 즉시 `lying_suppressed` 처리. 최종 emergency 게이트는 `count>=EMERGENCY_THRESHOLD AND dynamic_event AND lstm_fall` 모두 충족할 때만 발화. 응답에 `dynamic_fall_event`, `dynamic_gate_reason`, `recent_*` 필드 추가. 자세한 내용은 `README_QA.md` §11 참조. |
| 2026-05-02 | (6차) **엄격 게이팅** — 천천히 눕는 중 brief motion spike 누적으로 fall_suspected 가 잠깐 표시되던 잔존 문제 해결. Phase 2 counter HOLD→RESET, 상태 머신에 `not dynamic_event` 가드 추가, 프론트 CameraPage 가 backend 의 fall_* 를 dynamic_event=false 면 movement_pending 으로 자동 변환. 알람 트리거도 `alert_triggered=true AND dynamic_fall_event=true` 둘 다 요구. |
| 2026-05-02 | (7차) **낙상 사건 latch** — 낙상 확정 후 사람이 오래 누워있으면 lying_suppressed 로 자동 전이되던 문제 해결. `fall_incident_active`/`incident_state`/`fall_incident_started_at`/`fall_incident_id` 도입. fall_emergency 1회 확정 시 latch 활성화 → 명시적 `/detect/reset` 까지 final_decision 강제 유지. `FALL_ALERT_COOLDOWN_SECONDS` env (default 0 = 최초 1회만, >0 = reminder). CameraPage 에 빨간 사건 active 배너 + "사건 해제" 버튼. |

# Fall Detection QA 가이드

이 문서는 CareVision 낙상감지 파이프라인의 **API 계약 검증(=mock 기반 단위 테스트)** 과
**실제 base64/영상 입력으로의 수동 QA** 절차를 모두 설명합니다.

> 핵심 앱 코드(`ai/` 내부)는 가급적 수정하지 않고, QA는 별도 파일에서 수행합니다.

---

## 1. 자동 테스트 (의존성 없는 mock 기반)

### 1.1 파일
- `qa/test_fall_api.py` — `ai.api.routes` import 시 무거운 의존성을 test double로 주입.

### 1.2 테스트 범위
1. `GET  /health` — 헬스체크 (`status == "ok"`)
2. `POST /detect/reset` — 상태 초기화 응답 (`success == True`, `cameraId` 에코)
3. `POST /detect/fall` — 기본 응답 구조 (`type/status/confidence/method` 키 존재)
4. `POST /detect/fall` — 잘못된 payload → `422`

### 1.3 실행
프로젝트 루트에서:

```bash
# Windows
py -m pytest -q qa/test_fall_api.py

# Linux/Mac
python -m pytest -q qa/test_fall_api.py
```

자세한 로그가 필요하면:

```bash
py -m pytest -vv qa/test_fall_api.py -rs
```

### 1.4 예상 결과
```
4 passed in 0.69s
```
(소요시간은 환경에 따라 다름)

### 1.5 mock을 쓰는 이유
테스트 환경에 `mediapipe`, `tensorflow`, `torch`, `ultralytics`, OpenCV, 모델 파일이
없어도 **엔드포인트 계약(상태코드 + JSON 구조)** 만 검증하기 위함. 모델/데이터셋
없이도 항상 4개 테스트가 통과해야 한다.

### 1.6 최소 필요 패키지 (자동 테스트 용)
- `pytest`
- `fastapi`
- `starlette`
- `pydantic`
- (선택) `httpx` — `services/backend_client` 가 mock되므로 자동 테스트에는 필수 아님.
- (선택) `python-multipart` — `UploadFile` 라우트 import 시 FastAPI 가 요구.
  현재 `requirements.txt` 6번 줄에 등록되어 있음.

---

## 2. 실제 base64 이미지로 낙상감지 수동 QA

자동 테스트는 mock 기반이라 **모델 로직** 자체는 검증하지 않는다. 실제 모델 동작을
확인하려면 AI 서버를 띄우고 base64 이미지를 직접 보내야 한다.

### 2.1 사전 준비
```bash
cd ai
.\venv\Scripts\activate         # Windows. Linux/Mac: source venv/bin/activate
pip install -r ..\requirements.txt
uvicorn main:app --reload --port 8000
```

서버가 떠 있는지 확인:
```bash
curl http://localhost:8000/health
# → {"status":"ok","message":"CareVision AI Server is running"}
```

### 2.2 base64 인코딩 (단일 이미지 → JSON payload 만들기)

**Linux / Mac / WSL:**
```bash
B64=$(base64 -w 0 ./test/fall_sample.jpg)
cat > /tmp/payload.json <<EOF
{"image":"$B64","cameraId":"qa-cam"}
EOF
```

**Windows PowerShell:**
```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes(".\test\fall_sample.jpg"))
@{ image = $b64; cameraId = "qa-cam" } | ConvertTo-Json -Compress > .\payload.json
```

> 데이터 URL 형식(`data:image/jpeg;base64,xxx`)도 `/detect/live` 는 자동 strip 하지만,
> `/detect/fall` 은 순수 base64 만 받는다. 안전하게 **prefix 없는 base64** 만 쓰자.

### 2.3 `/detect/fall` 호출

```bash
curl -X POST http://localhost:8000/detect/fall \
     -H "Content-Type: application/json" \
     -d @/tmp/payload.json
```

응답 예시 (휴리스틱 폴백 — 모델 파일 없거나 워밍업 중):
```json
{
  "detected": false,
  "confidence": 0.0,
  "type": "FALL",
  "status": "buffering",
  "buffered": 1,
  "consecutive_frames": 0,
  "method": "heuristic:upright"
}
```

응답 예시 (LSTM 활성 + 휴리스틱 결합):
```json
{
  "detected": true,
  "confidence": 0.87,
  "type": "FALL",
  "status": "emergency",
  "consecutive_frames": 3,
  "method": "lstm",
  "lstm_prob": 0.872,
  "heuristic": false
}
```

### 2.4 상태 머신 확인 절차
1. `POST /detect/reset {"cameraId":"qa-cam"}` — 초기화
2. 정자세 이미지 30프레임(=`SEQUENCE_LEN`) 이상 전송 → `status=buffering` → `normal`
3. 누운/엎드린 자세 이미지를 연속 2~3프레임 전송 → `caution` → `suspected` → `emergency`
4. 다시 `POST /detect/reset` — 다음 시나리오 격리

> `confidence`, `lstm_prob`, `heuristic` 필드 변화로 어떤 신호가 트리거됐는지 파악할 수 있다.

---

## 3. 실제 영상으로 낙상감지 수동 QA

영상은 OpenCV 로 프레임을 뽑은 뒤 base64 로 인코딩해 `/detect/fall` 또는
`/detect/live` 에 순서대로 POST 한다. 다음 스니펫을 `qa/run_video_qa.py` 같은
이름으로 저장해 사용한다.

```python
# qa/run_video_qa.py (예시)
import argparse, base64, json, time
import cv2
import urllib.request

def encode_jpeg(frame, quality=85):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--camera", default="qa-cam")
    ap.add_argument("--endpoint", default="/detect/fall",
                    choices=["/detect/fall", "/detect/live"])
    ap.add_argument("--every", type=int, default=1, help="N프레임마다 1번 전송")
    args = ap.parse_args()

    # 새 시나리오마다 카운터/버퍼 초기화
    post(args.server + "/detect/reset", {"cameraId": args.camera})

    cap = cv2.VideoCapture(args.video)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            payload = {"image": encode_jpeg(frame), "cameraId": args.camera}
            res = post(args.server + args.endpoint, payload)
            # /detect/fall 은 그대로, /detect/live 는 res["fall"] 만 출력
            fall = res.get("fall", res)
            print(f"[{idx:04d}] status={fall.get('status')} "
                  f"conf={fall.get('confidence')} method={fall.get('method')}")
        idx += 1
    cap.release()

if __name__ == "__main__":
    main()
```

실행:
```bash
# /detect/fall 로 보내기 (낙상 단독 테스트)
py qa/run_video_qa.py ai/test_videos/fall_sample.mp4 --every 2

# /detect/live 로 보내기 (낙상 + 복약 통합 응답)
py qa/run_video_qa.py ai/test_videos/fall_sample.mp4 --endpoint /detect/live --every 2
```

> `--every 2` 는 30fps 영상을 ≈15fps 로 다운샘플하여 서버 부하를 낮추기 위한 옵션.

### 3.1 기대 동작
- 영상 초반(서있는 구간): `method=heuristic:upright` 또는 `lstm`, `status=normal`
- 낙상 구간: `consecutive_frames` 가 증가하며 `caution → suspected → emergency`
- `_get_model()` 첫 로드 시 stdout 에 다음 중 하나가 한 번만 출력됨:
  - `[FallDetector] Keras LSTM 모델 로드 완료: ...fall_lstm_vB_best.keras`
  - `[FallDetector] PyTorch LSTM 모델 로드 완료: ...fall_lstm.pt (window_size=15)`
  - `[FallDetector] LSTM 모델 파일이 없어 휴리스틱(...)만 사용합니다.`

---

## 4. 모델 파일 / 백엔드 알림 관련 메모

### 4.1 낙상 모델 파일 우선순위 (현 구현 기준)
| 우선순위 | 파일 | 백엔드 | 비고 |
|---|---|---|---|
| 1 | `ai/models/fall_lstm_vB_best.keras` | Keras (BiLSTM, 5-class) | 현재 레포에는 미포함 |
| 2 | `ai/models/fall_lstm.pt` (+ `fall_lstm_meta.json`) | PyTorch (LSTM, 2-class) | `train_fall.py` 출력물 |
| 3 | (모두 없음) | 휴리스틱(MediaPipe Pose) + YOLO bbox 폴백 | 서버는 죽지 않음 |

`_get_model()` 은 위 순서대로 시도하며, **모두 실패해도 서버는 정상 동작** 하고
경고 로그는 한 번만 출력된다.

### 4.2 AI 서버 → 백엔드 알림 케이스
- AI 가 보내는 `detection_type` 은 `"FALL"` (대문자, Prisma `DetectionType` enum 과 일치).
- 백엔드 `detectionController.createLog` 는 `type.toLowerCase() === 'fall'` 로 비교하여
  대소문자 어느 쪽이 와도 FCM 알림이 발송되도록 수정됨.

### 4.3 `/detect/live` 와 `/detect/fall` 차이
| 항목 | `/detect/fall` | `/detect/live` |
|---|---|---|
| 입력 base64 prefix 처리 | **요구하지 않음** (순수 base64 만) | `data:image/...;base64,` prefix 자동 strip |
| 백엔드 알림 트리거 | `status == "emergency"` 이면 호출 | 호출하지 않음 (응답만 반환) |
| 응답 형태 | fall 결과 평면 JSON | `{ medication, hand_to_mouth, medication_score, fall, ... }` |

---

## 5. 변경 요약 (2026-05-02 QA 1차 후속)
- `requirements.txt` — `httpx`, `python-multipart` 등록 확인.
- `ai/pipelines/fall_detector.py` — `.keras → .pt → 휴리스틱` 순으로 안전하게 폴백.
  모델 파일이 없어도 서버가 죽지 않으며, 경고 로그는 한 번만 출력.
- `backend/src/controllers/detectionController.js` — FALL/fall 대소문자 비교 정규화.
- `qa/README_QA.md` — 본 문서. base64/영상 수동 QA 절차 추가.

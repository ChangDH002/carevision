import sys
import os
from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# 상위 디렉토리 import 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipelines.fall_detector import fall_detector
from pipelines import fall_detector as fall_detector_mod  # get_model_info() 접근용
from pipelines.medication_detector import medication_detector
from pipelines.hand_to_mouth_detector import hand_to_mouth_detector
from pipelines.medication_scorer import init_scorer
from detector import detector
from services.backend_client import notify_detection, notify_medication_log

# 스코어러 초기화 (약 감지 + 손→입 LSTM 결합)
medication_scorer = init_scorer(medication_detector, hand_to_mouth_detector)

router = APIRouter()


# 요청 모델 정의
class FallDetectRequest(BaseModel):
    image: str  # base64 인코딩된 이미지
    cameraId: str = "default"
    patientId: Optional[int] = None


class MedicationDetectRequest(BaseModel):
    image: str
    cameraId: str = "default"
    patientId: Optional[int] = None
    scheduleId: Optional[int] = None


class ImageRequest(BaseModel):
    image: str


# ===================================================
# MVP: 사진 업로드 → 약 감지
# ===================================================

@router.post("/detect/medication/upload")
async def detect_medication_upload(file: UploadFile = File(...)):
    """
    [MVP] 사진 파일 업로드 → 약 객체 감지
    - 지원 형식: JPG, PNG
    - 반환: 감지 결과 + 바운딩박스 이미지 (base64)
    """
    contents = await file.read()
    result = medication_detector.detect_from_file(contents)

    return {
        "success": True,
        "data": result,
        "message": result["message"],
    }


@router.get("/test/medication", response_class=HTMLResponse)
async def test_medication_page():
    """MVP 테스트 페이지 — 사진 업로드해서 약 감지 결과 확인"""
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CareVision — 복약 감지 MVP 테스트</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a; color: #e2e8f0;
            min-height: 100vh; padding: 2rem;
        }
        h1 { text-align: center; margin-bottom: 0.5rem; font-size: 1.8rem; }
        .subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; }
        .container { max-width: 900px; margin: 0 auto; }

        .upload-area {
            border: 2px dashed #475569; border-radius: 16px;
            padding: 3rem; text-align: center; cursor: pointer;
            transition: all 0.3s; margin-bottom: 1.5rem;
        }
        .upload-area:hover { border-color: #3b82f6; background: #1e293b; }
        .upload-area.dragover { border-color: #22c55e; background: #1e293b; }
        .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
        .upload-text { font-size: 1.1rem; color: #94a3b8; }

        .btn {
            background: #3b82f6; color: white; border: none;
            padding: 12px 32px; border-radius: 8px; font-size: 1rem;
            cursor: pointer; transition: background 0.2s;
        }
        .btn:hover { background: #2563eb; }
        .btn:disabled { background: #475569; cursor: not-allowed; }

        .result-section { margin-top: 2rem; }
        .result-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 1.5rem; margin-top: 1rem;
        }
        @media (max-width: 768px) { .result-grid { grid-template-columns: 1fr; } }

        .card {
            background: #1e293b; border-radius: 12px;
            padding: 1.5rem; border: 1px solid #334155;
        }
        .card h3 { margin-bottom: 1rem; color: #3b82f6; }
        .card img {
            width: 100%; border-radius: 8px;
            border: 1px solid #334155;
        }

        .status-badge {
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-weight: 600; font-size: 0.9rem; margin-bottom: 1rem;
        }
        .status-detected { background: #166534; color: #4ade80; }
        .status-none { background: #7c2d12; color: #fb923c; }

        .detection-item {
            background: #0f172a; border-radius: 8px;
            padding: 10px 14px; margin-bottom: 8px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .detection-label { font-weight: 600; }
        .detection-conf {
            background: #334155; padding: 2px 8px;
            border-radius: 4px; font-size: 0.85rem;
        }

        .loading { display: none; text-align: center; padding: 2rem; }
        .spinner {
            width: 40px; height: 40px; border: 4px solid #334155;
            border-top-color: #3b82f6; border-radius: 50%;
            animation: spin 0.8s linear infinite; margin: 0 auto 1rem;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .preview-img {
            max-height: 200px; border-radius: 8px;
            margin-top: 1rem; border: 1px solid #334155;
        }
        .model-info {
            background: #1e293b; border-radius: 8px;
            padding: 12px 16px; margin-bottom: 1.5rem;
            border-left: 3px solid #f59e0b; font-size: 0.9rem; color: #fbbf24;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CareVision 복약 감지 MVP</h1>
        <p class="subtitle">사진을 업로드하면 약 관련 객체를 감지합니다</p>

        <div class="model-info" id="modelInfo"></div>

        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📷</div>
            <div class="upload-text">클릭하거나 이미지를 드래그해서 업로드</div>
            <div style="color: #64748b; margin-top: 0.5rem; font-size: 0.85rem;">JPG, PNG 지원</div>
            <input type="file" id="fileInput" accept="image/*" style="display:none">
            <img id="previewImg" class="preview-img" style="display:none">
        </div>

        <div style="text-align: center;">
            <button class="btn" id="detectBtn" onclick="detect()" disabled>감지 시작</button>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>분석 중...</div>
        </div>

        <div class="result-section" id="resultSection" style="display:none">
            <h2>감지 결과</h2>
            <div class="result-grid">
                <div class="card">
                    <h3>분석 이미지</h3>
                    <img id="resultImg">
                </div>
                <div class="card">
                    <h3>감지 결과</h3>
                    <div id="statusBadge"></div>
                    <div id="resultMessage" style="margin-bottom: 1rem; color: #94a3b8;"></div>
                    <div id="detectionList"></div>

                    <h3 style="margin-top: 1.5rem;">전체 감지 객체</h3>
                    <div id="allObjectsList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewImg = document.getElementById('previewImg');
        const detectBtn = document.getElementById('detectBtn');

        // 드래그 앤 드롭
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) handleFile(e.target.files[0]);
        });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                previewImg.style.display = 'block';
            };
            reader.readAsDataURL(file);
            detectBtn.disabled = false;
            document.getElementById('resultSection').style.display = 'none';
        }

        async function detect() {
            if (!selectedFile) return;

            detectBtn.disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const res = await fetch('/detect/medication/upload', {
                    method: 'POST',
                    body: formData,
                });
                const json = await res.json();
                showResult(json.data);
            } catch (err) {
                alert('오류 발생: ' + err.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                detectBtn.disabled = false;
            }
        }

        function showResult(data) {
            document.getElementById('resultSection').style.display = 'block';

            // 분석 이미지
            if (data.annotated_image) {
                document.getElementById('resultImg').src = 'data:image/jpeg;base64,' + data.annotated_image;
            }

            // 상태 배지
            const badge = document.getElementById('statusBadge');
            if (data.detected) {
                badge.innerHTML = '<span class="status-badge status-detected">약 객체 감지됨</span>';
            } else {
                badge.innerHTML = '<span class="status-badge status-none">약 객체 미감지</span>';
            }

            // 메시지
            document.getElementById('resultMessage').textContent = data.message;

            // 모델 정보
            const modelInfo = document.getElementById('modelInfo');
            if (data.model_type === 'coco_base') {
                modelInfo.textContent = '현재 기본 COCO 모델 사용 중 — bottle, cup만 감지 가능. 파인튜닝 후 pill/medicine 감지 가능';
                modelInfo.style.display = 'block';
            } else {
                modelInfo.textContent = '파인튜닝 모델 사용 중 — pill, medicine, tablet 등 감지 가능';
                modelInfo.style.borderLeftColor = '#22c55e';
                modelInfo.style.color = '#4ade80';
                modelInfo.style.display = 'block';
            }

            // 약 감지 리스트
            const medList = document.getElementById('detectionList');
            medList.innerHTML = '';
            if (data.medication_objects.length > 0) {
                data.medication_objects.forEach(det => {
                    medList.innerHTML += `
                        <div class="detection-item">
                            <span class="detection-label" style="color: #4ade80;">${det.label}</span>
                            <span class="detection-conf">${(det.confidence * 100).toFixed(0)}%</span>
                        </div>`;
                });
            } else {
                medList.innerHTML = '<div style="color: #64748b;">없음</div>';
            }

            // 전체 객체 리스트
            const allList = document.getElementById('allObjectsList');
            allList.innerHTML = '';
            if (data.all_objects.length > 0) {
                data.all_objects.forEach(det => {
                    allList.innerHTML += `
                        <div class="detection-item">
                            <span class="detection-label">${det.label}</span>
                            <span class="detection-conf">${(det.confidence * 100).toFixed(0)}%</span>
                        </div>`;
                });
            } else {
                allList.innerHTML = '<div style="color: #64748b;">감지된 객체 없음</div>';
            }
        }
    </script>
</body>
</html>
"""


# ===================================================
# 기존 API (영상 모드 / base64)
# ===================================================

@router.post("/detect/fall")
async def detect_fall(request: FallDetectRequest, background_tasks: BackgroundTasks):
    result = fall_detector.detect(request.image, request.cameraId)

    # 알림 게이팅 — alert_triggered=True (즉 final_decision="fall_emergency") 일 때만 보호자 알림.
    # movement_pending / fall_suspected / lying_suppressed 는 절대 알림 보내지 않음.
    if result.get("alert_triggered") is True and request.patientId:
        background_tasks.add_task(
            notify_detection,
            patient_id=request.patientId,
            detection_type="FALL",
            # confidence 컬럼에는 fall_probability 를 넣어 의미를 명확히 한다.
            confidence=result.get("fall_probability", result.get("confidence", 0.0)),
            status=result.get("status", "emergency"),
            camera_id=request.cameraId,
        )

    # 응답에 현재 활성 모델 backend 짧은 태그를 첨부 — 클라이언트가 어느 백엔드의 결과인지 식별
    info = fall_detector_mod.get_model_info()
    result["model_backend"] = info.get("model_backend")
    result["qa_validated"] = info.get("qa_validated")
    return result


@router.get("/detect/model_info")
def detect_model_info():
    """현재 런타임에서 활성화된 낙상 모델의 backend / 경로 / shape / fallback 사유.

    웹 UI 와 운영자가 'QA 에서 검증된 vB Keras 가 실제 사용 중인가' 를 즉시 확인할 수
    있게 한다. fallback (PyTorch / 휴리스틱) 인 경우 그 사유를 명시.
    """
    return fall_detector_mod.get_model_info()


@router.post("/detect/medication")
async def detect_medication(request: MedicationDetectRequest, background_tasks: BackgroundTasks):
    """[영상 모드] base64 이미지로 연속 프레임 복약 감지"""
    result = medication_detector.detect_stream(request.image, request.cameraId)

    if request.scheduleId is not None:
        result["scheduleId"] = request.scheduleId

    if result["status"] in ("completed", "missed") and request.patientId:
        background_tasks.add_task(
            notify_detection,
            patient_id=request.patientId,
            detection_type="MEDICATION",
            confidence=result["confidence"],
            status=result["status"],
            camera_id=request.cameraId,
        )

        if request.scheduleId:
            background_tasks.add_task(
                notify_medication_log,
                patient_id=request.patientId,
                schedule_id=request.scheduleId,
                status=result["status"],
            )

    return result


# ===================================================
# 라이브 웹캠 감지 (복약 + 낙상 통합)
# ===================================================

class LiveDetectRequest(BaseModel):
    image: str  # base64 (data URL 또는 순수 base64)
    cameraId: str = "webcam"


@router.post("/detect/live")
def detect_live(request: LiveDetectRequest):
    """
    프론트엔드 웹캠용 통합 감지 엔드포인트.
    - 약 객체 감지 (YOLO/ONNX)
    - 손→입 동작 감지 (MediaPipe Hands + LSTM)
    - 낙상 자세 판정 (MediaPipe Pose)
    - 두 복약 신호를 시간창에서 결합해 TAKEN 판정 산출
    """
    img_b64 = request.image
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]

    # 복약 스코어링 (내부에서 medication_detector + hand_to_mouth_detector 호출)
    score_result = medication_scorer.update(img_b64, request.cameraId)

    # 낙상은 별도 파이프라인
    fall_result = fall_detector.detect(img_b64, request.cameraId)

    # 현재 활성 낙상 모델 backend 메타 (UI 가 'QA 검증 모델 사용 중인지' 즉시 식별)
    fall_info = fall_detector_mod.get_model_info()
    fall_result["model_backend"] = fall_info.get("model_backend")
    fall_result["qa_validated"] = fall_info.get("qa_validated")

    return {
        "success": True,
        "medication": {
            "detected": len(score_result["medication_objects"]) > 0,
            "objects": score_result["medication_objects"],
        },
        "hand_to_mouth": {
            "detected": bool(score_result["signals"]["hand_to_mouth_prob"]
                             >= medication_scorer.MOTION_POS_THRESHOLD),
            "confidence": score_result["signals"]["hand_to_mouth_prob"],
            "status": score_result["signals"]["hand_status"],
        },
        "medication_score": {
            "taken": score_result["taken"],
            "score": score_result["score"],
            "status": score_result["status"],
            "window": score_result["window"],
        },
        "fall": fall_result,
        "model_type": medication_detector.model_type,
        "fall_model_backend": fall_info.get("model_backend"),
    }


class ResetRequest(BaseModel):
    cameraId: str = "webcam"


@router.post("/detect/reset")
def detect_reset(request: ResetRequest):
    """카메라의 모든 스코어러/카운터 초기화."""
    medication_scorer.reset(request.cameraId)
    fall_detector.reset(request.cameraId)
    return {"success": True, "cameraId": request.cameraId, "message": "reset ok"}


# --- 일반 객체 감지 ---
@router.post("/detect")
def detect_objects(request: ImageRequest):
    detections = detector.detect_from_base64(request.image)
    return {
        "success": True,
        "detections": detections,
        "count": len(detections),
    }


# --- 헬스 체크 ---
@router.get("/health")
def health_check():
    return {"status": "ok", "message": "CareVision AI Server is running"}

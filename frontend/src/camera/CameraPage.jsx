import { useState, useEffect, useRef, useCallback } from 'react';
import { store } from '../store';
import { emergencyBeep } from '../notify';

const AI_URL = import.meta.env.VITE_AI_BASE_URL || 'http://localhost:8000';
const FRAME_INTERVAL_MS = 500; // 2 fps 정도로 전송 (MVP 가시성 확인 용)

export default function CameraPage({ patient, patients = [], onSelectPatient, onClose }) {
    const [selectedLocal, setSelectedLocal] = useState(patient);

    useEffect(() => {
        setSelectedLocal(patient);
    }, [patient]);

    // ─────────────── 환자 미선택 → 선택 목록 ───────────────
    if (!selectedLocal) {
        return (
            <div className="min-h-screen bg-slate-100 max-w-[480px] mx-auto font-sans">
                <div className="bg-red-600 px-4 pt-5 pb-4">
                    <span className="text-white font-bold text-[17px]">🚨 긴급 모니터링</span>
                </div>
                <div className="p-3">
                    <p className="text-sm font-semibold text-gray-600 mb-3">
                        모니터링할 환자를 선택해주세요
                    </p>
                    <div className="flex flex-col gap-2">
                        {patients.map((p) => (
                            <div
                                key={p.id}
                                className="bg-white rounded-xl px-4 py-3 flex items-center gap-3 border border-gray-200 shadow-sm cursor-pointer"
                                onClick={() => {
                                    onSelectPatient?.(p);
                                    setSelectedLocal(p);
                                }}
                            >
                                <div className="w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-base shrink-0">
                                    {p.name?.[0]}
                                </div>
                                <div>
                                    <p className="text-sm font-bold text-gray-900 m-0">{p.name}</p>
                                    <p className="text-[11px] text-gray-500 m-0">
                                        {p.age}세 · {p.address}
                                    </p>
                                </div>
                                <span className="ml-auto text-gray-400">›</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    // ─────────────── 환자 선택 후 → 웹캠 + 실시간 감지 ───────────────
    return (
        <LiveDetectionView
            patient={selectedLocal}
            onChangePatient={() => setSelectedLocal(null)}
        />
    );
}

function LiveDetectionView({ patient, onChangePatient }) {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);
    const captureCanvasRef = useRef(null);
    const intervalRef = useRef(null);
    const inflightRef = useRef(false);
    const fileInputRef = useRef(null);
    const videoObjectUrlRef = useRef(null);

    const [streaming, setStreaming] = useState(false);
    const [source, setSource] = useState('webcam'); // 'webcam' | 'file'
    const [videoFileName, setVideoFileName] = useState('');
    const [videoPaused, setVideoPaused] = useState(false);
    const [error, setError] = useState(null);
    const [medObjects, setMedObjects] = useState([]);
    const [fall, setFall] = useState({ status: 'idle', confidence: 0 });
    const [modelType, setModelType] = useState('-');
    const [modelInfo, setModelInfo] = useState(null);
        // { model_backend, model_path, version, sequence_len, num_classes, fall_class,
        //   fallback_reason, qa_validated, ... }
    const [lastLatency, setLastLatency] = useState(0);
    const [alerts, setAlerts] = useState([]); // 상단 토스트 알림
    const [medScore, setMedScore] = useState(null);     // {taken, score, status, signals, window}
    const [handMouth, setHandMouth] = useState(null);   // {confidence, status}
    const prevFallStatus = useRef('idle');
    const prevMedCount = useRef(0);
    const prevTaken = useRef(false);
    const prevVideoTime = useRef(0);

    // 컴포넌트 mount 시 현재 활성 낙상 모델 backend 정보를 한 번 받아옴
    // — UI 가 'QA 검증된 vB Keras' 가 실제 사용 중인지 즉시 보여준다.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const res = await fetch(`${AI_URL}/detect/model_info`);
                if (!res.ok) return;
                const json = await res.json();
                if (!cancelled) setModelInfo(json);
            } catch {
                /* AI 서버 미기동 시 무시 — 추후 첫 detect 응답에서 갱신됨 */
            }
        })();
        return () => { cancelled = true; };
    }, []);

    // 감지기 상태 리셋 (루프 재시작 / 수동 리셋 공용)
    const resetDetectionState = useCallback(async () => {
        try {
            await fetch(`${AI_URL}/detect/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cameraId: `patient-${patient.id}` }),
            });
        } catch {}
        prevFallStatus.current = 'idle';
        prevMedCount.current = 0;
        prevTaken.current = false;
        setFall({ status: 'idle', confidence: 0 });
        setMedScore(null);
    }, [patient.id]);

    // 비디오가 루프되어 currentTime이 뒤로 점프하면 감지기 상태 리셋
    const onVideoTimeUpdate = useCallback(() => {
        const v = videoRef.current;
        if (!v) return;
        const t = v.currentTime;
        // 0.5초 이상 뒤로 점프 → 루프 재시작으로 간주
        if (prevVideoTime.current - t > 0.5) {
            resetDetectionState();
        }
        prevVideoTime.current = t;
    }, [resetDetectionState]);

    const pushAlert = useCallback((type, text) => {
        const id = Date.now() + Math.random();
        setAlerts((prev) => [...prev, { id, type, text }]);
        setTimeout(() => {
            setAlerts((prev) => prev.filter((a) => a.id !== id));
        }, 4000);
    }, []);

    // 현재 source와 상관없이 기존 스트림/파일 리소스를 깨끗하게 정리
    const teardownVideo = () => {
        const v = videoRef.current;
        if (v) {
            const stream = v.srcObject;
            if (stream && stream.getTracks) stream.getTracks().forEach((t) => t.stop());
            v.srcObject = null;
            try { v.pause(); } catch {}
            v.removeAttribute('src');
            v.load();
        }
        if (videoObjectUrlRef.current) {
            try { URL.revokeObjectURL(videoObjectUrlRef.current); } catch {}
            videoObjectUrlRef.current = null;
        }
    };

    const startCamera = async () => {
        setError(null);
        teardownVideo();
        setSource('webcam');
        setVideoFileName('');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: false,
            });
            videoRef.current.srcObject = stream;
            videoRef.current.loop = false;
            videoRef.current.muted = true;
            await videoRef.current.play();
            setStreaming(true);
            setVideoPaused(false);
        } catch (e) {
            setError('웹캠 접근 실패: ' + e.message);
        }
    };

    // 비디오 파일로 시연 (파일 업로드 또는 프리셋 URL)
    const startVideoFile = async (fileOrUrl) => {
        setError(null);
        teardownVideo();
        await resetDetectionState();
        prevVideoTime.current = 0;

        try {
            let src;
            let name;
            if (fileOrUrl instanceof File) {
                src = URL.createObjectURL(fileOrUrl);
                videoObjectUrlRef.current = src;
                name = fileOrUrl.name;
            } else {
                src = fileOrUrl;
                name = fileOrUrl.split('/').pop();
            }
            const v = videoRef.current;
            v.src = src;
            v.loop = false;       // 시연 편의용 루프
            v.muted = true;
            v.playsInline = true;
            await v.play();
            setSource('file');
            setVideoFileName(name);
            setStreaming(true);
            setVideoPaused(false);
        } catch (e) {
            setError('비디오 재생 실패: ' + e.message);
        }
    };

    const onPickFile = (e) => {
        const f = e.target.files?.[0];
        if (f) startVideoFile(f);
        // 같은 파일 재선택 허용
        e.target.value = '';
    };

    const togglePause = () => {
        const v = videoRef.current;
        if (!v) return;
        if (v.paused) {
            v.play();
            setVideoPaused(false);
        } else {
            v.pause();
            setVideoPaused(true);
        }
    };

    const stopCamera = () => {
        teardownVideo();
        setStreaming(false);
        setVideoPaused(false);
        setVideoFileName('');
        setMedObjects([]);
        setFall({ status: 'idle', confidence: 0 });
        setMedScore(null);
        setHandMouth(null);
        clearOverlay();
    };

    const resetScorer = async () => {
        try {
            await fetch(`${AI_URL}/detect/reset`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cameraId: `patient-${patient.id}` }),
            });
            prevTaken.current = false;
            setMedScore(null);
            pushAlert('info', '🔄 스코어러 리셋 완료');
        } catch (e) {
            pushAlert('warn', '리셋 실패: ' + e.message);
        }
    };

    const clearOverlay = () => {
        const cv = overlayRef.current;
        if (!cv) return;
        cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
    };

    const drawOverlay = (objects, vw, vh) => {
        const cv = overlayRef.current;
        if (!cv) return;
        cv.width = vw;
        cv.height = vh;
        const ctx = cv.getContext('2d');
        ctx.clearRect(0, 0, vw, vh);
        ctx.lineWidth = 3;
        ctx.font = 'bold 16px sans-serif';
        objects.forEach((d) => {
            const { x1, y1, x2, y2 } = d.bbox;
            ctx.strokeStyle = '#22c55e';
            ctx.fillStyle = '#22c55e';
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const label = `${d.label} ${(d.confidence * 100).toFixed(0)}%`;
            const tw = ctx.measureText(label).width + 8;
            ctx.fillRect(x1, Math.max(0, y1 - 20), tw, 20);
            ctx.fillStyle = '#0f172a';
            ctx.fillText(label, x1 + 4, Math.max(14, y1 - 5));
        });
    };

    const captureAndDetect = async () => {
        if (inflightRef.current) return;
        const video = videoRef.current;
        if (!video || video.readyState < 2) return;

        const w = video.videoWidth;
        const h = video.videoHeight;
        if (!w || !h) return;

        const canvas = captureCanvasRef.current;
        canvas.width = w;
        canvas.height = h;
        canvas.getContext('2d').drawImage(video, 0, 0, w, h);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

        inflightRef.current = true;
        const t0 = performance.now();
        try {
            const res = await fetch(`${AI_URL}/detect/live`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl, cameraId: `patient-${patient.id}` }),
            });
            const json = await res.json();
            if (json.success) {
                const objs = json.medication?.objects || [];
                setMedObjects(objs);
                setFall(json.fall || { status: 'unknown', confidence: 0 });
                setModelType(json.model_type);
                setMedScore(json.medication_score || null);
                setHandMouth(json.hand_to_mouth || null);
                drawOverlay(objs, w, h);
                setLastLatency(Math.round(performance.now() - t0));

                // 낙상 알림 게이팅 — alert_triggered=true (backend 의 사건 latch 가 이미
                // 최초 1회만 true 로 발화). dynamic_fall_event=true 도 추가 가드로 요구.
                const fallObj = json.fall || {};
                const newDecision = fallObj.final_decision || fallObj.status;
                const dynamicEvt = fallObj.dynamic_fall_event === true;
                const alertNow = fallObj.alert_triggered === true && dynamicEvt;
                const prevDecision = prevFallStatus.current;
                if (alertNow) {
                    // 사건 latch 의 최초 알림 (또는 cooldown reminder) → 보호자 알림 + 진동 + store 기록
                    pushAlert('danger', '🚨 낙상 감지! 위급 상황');
                    emergencyBeep();
                    store.recordDetection({
                        type: 'FALL',
                        confidence: fallObj.fall_probability ?? fallObj.confidence ?? 0.9,
                        patient,
                        extra: {
                            location: '거실',
                            source: 'camera',
                            incident_id: fallObj.fall_incident_id || null,
                        },
                    });
                } else if (newDecision !== prevDecision
                    && newDecision === 'fall_suspected'
                    && dynamicEvt
                    && fallObj.fall_incident_active !== true) {
                    // 화면 경고만 — 알림 / 진동 X. dynamic event 가 있고 아직 사건 latch 가 아닐 때만 노출.
                    pushAlert('warn', '⚠️ 낙상 의심 — 관찰 중');
                }
                // movement_pending / lying_suppressed / dynamic_event=false 인 fall_* :
                // 화면 라벨 변경만 (pushAlert 없음, beep 없음, store 기록 없음)
                prevFallStatus.current = newDecision;
                // 약 처음 감지되거나 개수가 늘어났을 때 알림
                if (objs.length > prevMedCount.current) {
                    pushAlert('info', `💊 약 ${objs.length}개 감지`);
                }
                prevMedCount.current = objs.length;

                // 복약 완료 판정 순간 한 번만 기록 (스토어에서 자동으로 스케줄 매칭 + 로그)
                const takenNow = !!json.medication_score?.taken;
                if (takenNow && !prevTaken.current) {
                    pushAlert('info', '✅ 복약 완료 확정!');
                    store.recordDetection({
                        type: 'MEDICATION',
                        confidence: json.medication_score?.score ?? 0.8,
                        patient,
                        extra: { source: 'camera', status: 'taken' },
                    });
                }
                prevTaken.current = takenNow;
            }
        } catch (e) {
            // 서버 죽어있는 경우 조용히 pass
        } finally {
            inflightRef.current = false;
        }
    };

    useEffect(() => {
        if (!streaming) return;
        intervalRef.current = setInterval(captureAndDetect, FRAME_INTERVAL_MS);
        return () => clearInterval(intervalRef.current);
        // eslint-disable-next-line
    }, [streaming]);

    useEffect(() => () => stopCamera(), []);

    // 낙상 상태별 색상/라벨 — final_decision 우선, 없으면 status 로 fallback.
    // 5-state UI:
    //   normal              → 정상
    //   movement_pending    → 판정 중 (알림 X)
    //   fall_suspected      → 낙상 의심 (화면 경고만, 알림 X)
    //   fall_emergency      → 낙상 발생 (보호자 알림 O — alert_triggered=true)
    //   lying_suppressed    → 정상/누워 있음 (알림 X)
    //
    // 사건 latch: fall_incident_active=true 인 동안에는 lying_suppressed 가 와도
    // 화면 라벨을 "낙상 발생 — 확인 필요" 로 강제 유지.
    // Defense-in-depth: backend 가 fall_suspected/fall_emergency 를 보내도
    // dynamic_fall_event=false 면 절대 화면에 "낙상" 라벨을 띄우지 않는다.
    const _rawKey = fall.final_decision || fall.status;
    const _isFallLabel = _rawKey === 'fall_suspected' || _rawKey === 'fall_emergency'
        || _rawKey === 'emergency' || _rawKey === 'suspected';
    let fallKey;
    if (fall.fall_incident_active === true) {
        // 사건 latch 활성 — 화면은 항상 낙상 발생 라벨 유지 (lying_suppressed 무시)
        fallKey = 'fall_incident_active';
    } else if (_isFallLabel && fall.dynamic_fall_event === false) {
        // dynamic_event 가 없으면 fall_* 대신 판정 중 표시
        fallKey = 'movement_pending';
    } else {
        fallKey = _rawKey;
    }
    const fallMeta = {
        // 새 final_decision 값
        fall_incident_active: { color: 'bg-red-700 text-white animate-pulse', label: '🚨 낙상 발생 — 확인 필요' },
        fall_emergency:   { color: 'bg-red-600 text-white',     label: '🚨 낙상 발생' },
        fall_suspected:   { color: 'bg-orange-500 text-white',  label: '⚠️ 낙상 의심' },
        movement_pending: { color: 'bg-yellow-400 text-gray-900', label: '⏳ 판정 중' },
        checking_transition: { color: 'bg-yellow-400 text-gray-900', label: '⏳ 판정 중' },
        lying_suppressed: { color: 'bg-blue-500 text-white',    label: '🛏 누워 있음' },
        normal:           { color: 'bg-green-500 text-white',   label: '정상' },
        warmup:           { color: 'bg-gray-300 text-gray-700', label: '준비 중' },
        no_person:        { color: 'bg-gray-300 text-gray-700', label: '사람 없음' },
        // legacy status 값 (fallback — final_decision 미존재 시)
        emergency:        { color: 'bg-red-600 text-white',     label: '🚨 낙상 발생' },
        suspected:        { color: 'bg-orange-500 text-white',  label: '⚠️ 낙상 의심' },
        checking:         { color: 'bg-yellow-400 text-gray-900', label: '⏳ 판정 중' },
        caution:          { color: 'bg-yellow-300 text-gray-900', label: '주의' },
        buffering:        { color: 'bg-gray-300 text-gray-700', label: '준비 중' },
        idle:             { color: 'bg-gray-200 text-gray-600', label: '대기 중' },
    }[fallKey] || { color: 'bg-gray-200 text-gray-600', label: fallKey || '대기 중' };

    return (
        <div className="min-h-screen bg-slate-100 max-w-[480px] mx-auto font-sans relative">
            {/* 헤더 */}
            <div className="bg-[#FF6B3D] px-4 pt-5 pb-4 flex justify-between items-center">
                <span className="text-white font-bold text-[17px]">🚨 실시간 모니터링</span>
                <button
                    className="bg-white text-[#FF6B3D] border-none rounded-lg px-3 py-1.5 font-semibold cursor-pointer text-sm"
                    onClick={onChangePatient}
                >
                    환자 변경
                </button>
            </div>

            {/* 환자 정보 */}
            <div className="px-4 py-3 bg-white border-b border-gray-200 flex items-center gap-3">
                <div className="w-9 h-9 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold text-sm shrink-0">
                    {patient.name?.[0]}
                </div>
                <div>
                    <p className="text-sm font-bold text-gray-900 m-0">{patient.name} 님 실시간 영상</p>
                    <p className="text-[11px] text-gray-500 m-0">
                        {patient.age}세 · {patient.address}
                    </p>
                </div>
            </div>

            {/* 토스트 알림 영역 */}
            {alerts.length > 0 && (
                <div className="fixed top-3 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 w-[90%] max-w-[440px]">
                    {alerts.map((a) => (
                        <div
                            key={a.id}
                            className={`rounded-lg px-3 py-2 text-sm font-semibold shadow-lg animate-pulse ${
                                a.type === 'danger'
                                    ? 'bg-red-600 text-white'
                                    : a.type === 'warn'
                                    ? 'bg-orange-500 text-white'
                                    : 'bg-blue-600 text-white'
                            }`}
                        >
                            {a.text}
                        </div>
                    ))}
                </div>
            )}

            {/* 웹캠 + 오버레이 */}
            <div className="p-3">
                <div className="relative bg-black rounded-2xl overflow-hidden aspect-[4/3]">
                    <video
                        ref={videoRef}
                        className="absolute inset-0 w-full h-full object-contain"
                        muted
                        playsInline
                        onTimeUpdate={onVideoTimeUpdate}
                    />
                    <canvas
                        ref={overlayRef}
                        className="absolute inset-0 w-full h-full pointer-events-none"
                    />
                    <canvas ref={captureCanvasRef} className="hidden" />

                    {!streaming && (
                        <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
                            카메라 꺼짐 — 아래 버튼으로 시작
                        </div>
                    )}

                    {/* 상단 상태 배지 */}
                    {streaming && (
                        <div className="absolute top-2 left-2 flex gap-2">
                            <span className={`text-[11px] font-bold rounded-md px-2 py-1 ${fallMeta.color}`}>
                                {fallMeta.label}
                            </span>
                            {medObjects.length > 0 && (
                                <span className="text-[11px] font-bold rounded-md px-2 py-1 bg-green-600 text-white">
                                    💊 {medObjects.length}
                                </span>
                            )}
                        </div>
                    )}
                </div>

                {/* 제어 버튼 */}
                <div className="mt-3 flex gap-2">
                    {!streaming ? (
                        <>
                            <button
                                onClick={startCamera}
                                className="flex-1 bg-[#FF6B3D] text-white rounded-xl py-3 font-bold text-sm border-none cursor-pointer"
                            >
                                ▶ 웹캠 시작
                            </button>
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="flex-1 bg-blue-600 text-white rounded-xl py-3 font-bold text-sm border-none cursor-pointer"
                            >
                                🎬 영상 선택
                            </button>
                        </>
                    ) : (
                        <>
                            <button
                                onClick={stopCamera}
                                className="flex-1 bg-red-500 text-white rounded-xl py-3 font-bold text-sm border-none cursor-pointer"
                            >
                                ■ 정지
                            </button>
                            {source === 'file' && (
                                <button
                                    onClick={togglePause}
                                    className="bg-gray-800 text-white rounded-xl py-3 px-4 font-bold text-sm border-none cursor-pointer"
                                >
                                    {videoPaused ? '▶' : '⏸'}
                                </button>
                            )}
                        </>
                    )}
                    <button
                        onClick={resetScorer}
                        className="bg-gray-200 text-gray-800 rounded-xl py-3 px-4 font-bold text-sm border-none cursor-pointer"
                    >
                        🔄 리셋
                    </button>
                    {/* 숨겨진 파일 input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={onPickFile}
                    />
                </div>

                {/* 시연 영상 모드 알림 배너 */}
                {streaming && source === 'file' && (
                    <div className="mt-3 bg-blue-50 border border-blue-200 rounded-xl p-3 flex items-center gap-2">
                        <span className="text-lg">🎬</span>
                        <div className="flex-1 min-w-0">
                            <p className="text-[11px] font-bold text-blue-700 m-0">시연 모드 (파일 재생)</p>
                            <p className="text-[10px] text-blue-600 m-0 truncate">{videoFileName}</p>
                        </div>
                        <span className="text-[10px] text-blue-500 font-mono">
                            {videoPaused ? 'PAUSED' : 'LIVE'}
                        </span>
                    </div>
                )}

                {/* 시작 전 힌트 */}
                {!streaming && (
                    <div className="mt-3 bg-white rounded-xl p-3 border border-dashed border-gray-300 text-[11px] text-gray-600">
                        <p className="m-0 mb-1 font-semibold text-gray-700">📹 시연 방법</p>
                        <p className="m-0 leading-relaxed">
                            • <b>웹캠 시작</b>: 실시간 웹캠으로 감지<br/>
                            • <b>영상 선택</b>: 준비된 mp4/웹m 파일을 재생하며 AI가 프레임을 분석
                            (낙상·복약 시연용). 루프 재생되며 하단 컨트롤로 일시정지 가능.
                        </p>
                    </div>
                )}

                {/* 수동 테스트 (AI 서버 없이 알림/기록 흐름 확인용) */}
                <div className="mt-3 bg-white rounded-xl p-3 border border-dashed border-gray-300">
                    <p className="text-[11px] font-semibold text-gray-500 m-0 mb-2">🧪 테스트용 시뮬레이션</p>
                    <div className="flex gap-2">
                        <button
                            onClick={() => store.recordDetection({
                                type: 'FALL', confidence: 0.87, patient,
                                extra: { location: '거실', source: 'manual' },
                            })}
                            className="flex-1 bg-[#E53935] text-white rounded-lg py-2 text-xs font-bold border-none cursor-pointer"
                        >
                            낙상 시뮬
                        </button>
                        <button
                            onClick={() => {
                                store.recordDetection({
                                    type: 'MEDICATION', confidence: 0.92, patient,
                                    extra: { source: 'manual', status: 'taken' },
                                });
                                const meds = store.getMedications(patient.id);
                                if (meds.length > 0) {
                                    store.logMedication(patient.id, meds[0].id, 'TAKEN');
                                }
                            }}
                            className="flex-1 bg-[#10B981] text-white rounded-lg py-2 text-xs font-bold border-none cursor-pointer"
                        >
                            복약 시뮬
                        </button>
                    </div>
                </div>

                {/* 복약 완료 큰 배너 */}
                {medScore?.taken && (
                    <div className="mt-3 bg-green-600 text-white rounded-xl p-4 text-center shadow-lg">
                        <p className="text-[11px] m-0 opacity-90">MEDICATION TAKEN</p>
                        <p className="text-lg font-bold m-0 mt-1">✅ 복약 완료</p>
                        <p className="text-[10px] m-0 mt-1 opacity-90">
                            score {(medScore.score ?? 0).toFixed(2)} · status {medScore.status}
                        </p>
                    </div>
                )}

                {/* 복약 스코어 상세 카드 */}
                {medScore && (
                    <div className="mt-3 bg-white rounded-xl p-3 border border-gray-200">
                        <div className="flex justify-between items-center mb-2">
                            <p className="text-[12px] font-bold text-gray-800 m-0">📊 복약 스코어</p>
                            <span
                                className={`text-[10px] font-bold rounded px-2 py-0.5 ${
                                    medScore.status === 'taken'
                                        ? 'bg-green-100 text-green-700'
                                        : medScore.status === 'preparing'
                                        ? 'bg-yellow-100 text-yellow-700'
                                        : medScore.status === 'eating_unknown'
                                        ? 'bg-orange-100 text-orange-700'
                                        : 'bg-gray-100 text-gray-600'
                                }`}
                            >
                                {medScore.status}
                            </span>
                        </div>

                        {/* 총점 게이지 */}
                        <div className="mb-2">
                            <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                                <span>score</span>
                                <span className="font-mono">{(medScore.score ?? 0).toFixed(3)}</span>
                            </div>
                            <div className="w-full h-2 bg-gray-100 rounded">
                                <div
                                    className="h-2 rounded bg-green-500 transition-all"
                                    style={{ width: `${Math.min(100, (medScore.score ?? 0) * 100)}%` }}
                                />
                            </div>
                        </div>

                        {/* 박스 프레임 진행도 */}
                        <div className="mb-2">
                            <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                                <span>약통 감지 프레임</span>
                                <span className="font-mono">
                                    {medScore.window.box_frames} / {medScore.window.total_frames}
                                    {'  (min '}{medScore.window.box_min}{')'}
                                </span>
                            </div>
                            <div className="w-full h-1.5 bg-gray-100 rounded">
                                <div
                                    className={`h-1.5 rounded transition-all ${
                                        medScore.window.box_frames >= medScore.window.box_min
                                            ? 'bg-green-500'
                                            : 'bg-blue-400'
                                    }`}
                                    style={{
                                        width: `${Math.min(
                                            100,
                                            (medScore.window.box_frames /
                                                Math.max(1, medScore.window.box_min)) *
                                                100
                                        )}%`,
                                    }}
                                />
                            </div>
                        </div>

                        {/* 모션 프레임 진행도 */}
                        <div className="mb-2">
                            <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                                <span>손→입 동작 프레임</span>
                                <span className="font-mono">
                                    {medScore.window.motion_frames} / {medScore.window.total_frames}
                                    {'  (min '}{medScore.window.motion_min}{')'}
                                </span>
                            </div>
                            <div className="w-full h-1.5 bg-gray-100 rounded">
                                <div
                                    className={`h-1.5 rounded transition-all ${
                                        medScore.window.motion_frames >= medScore.window.motion_min
                                            ? 'bg-green-500'
                                            : 'bg-purple-400'
                                    }`}
                                    style={{
                                        width: `${Math.min(
                                            100,
                                            (medScore.window.motion_frames /
                                                Math.max(1, medScore.window.motion_min)) *
                                                100
                                        )}%`,
                                    }}
                                />
                            </div>
                        </div>

                        <p className="text-[10px] text-gray-400 m-0 mt-1">
                            시간창 {medScore.window.seconds}s · 박스 최고 conf{' '}
                            {((medScore.signals?.box_top_confidence ?? 0) * 100).toFixed(0)}%
                        </p>
                    </div>
                )}

                {/* 손→입 LSTM 카드 */}
                {handMouth && (
                    <div className="mt-3 bg-white rounded-xl p-3 border border-gray-200">
                        <div className="flex justify-between items-center mb-2">
                            <p className="text-[12px] font-bold text-gray-800 m-0">✋ 손→입 동작 (LSTM)</p>
                            <span
                                className={`text-[10px] font-bold rounded px-2 py-0.5 ${
                                    handMouth.detected
                                        ? 'bg-red-100 text-red-700'
                                        : (handMouth.status || '').includes('suspect')
                                        ? 'bg-orange-100 text-orange-700'
                                        : 'bg-gray-100 text-gray-600'
                                }`}
                            >
                                {handMouth.status || '-'}
                            </span>
                        </div>
                        <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                            <span>확률</span>
                            <span className="font-mono">
                                {((handMouth.confidence ?? 0) * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="w-full h-2 bg-gray-100 rounded">
                            <div
                                className={`h-2 rounded transition-all ${
                                    (handMouth.confidence ?? 0) >= 0.5 ? 'bg-red-500' : 'bg-purple-400'
                                }`}
                                style={{
                                    width: `${Math.min(100, (handMouth.confidence ?? 0) * 100)}%`,
                                }}
                            />
                        </div>
                    </div>
                )}

                {error && (
                    <div className="mt-3 bg-red-100 text-red-700 text-xs p-3 rounded-lg">
                        {error}
                    </div>
                )}

                {/* 활성 낙상 모델 backend 배지 — QA 검증 모델(vB Keras) 인지 즉시 식별 */}
                {modelInfo && (
                    <div
                        className={`mt-3 rounded-xl p-3 border ${
                            modelInfo.qa_validated
                                ? 'bg-green-50 border-green-200'
                                : 'bg-amber-50 border-amber-300'
                        }`}
                        title={modelInfo.fallback_reason || '현재 활성 낙상 모델'}
                    >
                        <div className="flex items-center gap-2">
                            <span
                                className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                                    modelInfo.qa_validated
                                        ? 'bg-green-600 text-white'
                                        : 'bg-amber-600 text-white'
                                }`}
                            >
                                {modelInfo.qa_validated ? 'QA 검증' : '폴백'}
                            </span>
                            <span className="text-xs font-semibold text-gray-800">
                                낙상 모델: {modelInfo.model_backend}
                            </span>
                            <span className="text-[10px] text-gray-500 ml-auto">
                                seq={modelInfo.sequence_len ?? '-'} · cls={modelInfo.num_classes ?? '-'}
                                {modelInfo.fall_class != null && ` · fall=${modelInfo.fall_class}`}
                            </span>
                        </div>
                        {!modelInfo.qa_validated && modelInfo.fallback_reason && (
                            <p className="text-[10px] text-amber-800 m-0 mt-1 leading-snug">
                                ⚠ {modelInfo.fallback_reason}
                            </p>
                        )}
                    </div>
                )}

                {/* 감지 상태 패널 */}
                <div className="mt-3 grid grid-cols-2 gap-2">
                    <div className="bg-white rounded-xl p-3 border border-gray-200">
                        <p className="text-[11px] text-gray-500 m-0 mb-1">💊 복약 감지</p>
                        <p className="text-lg font-bold text-gray-900 m-0">
                            {medObjects.length}개
                        </p>
                        <p className="text-[10px] text-gray-400 m-0 mt-1">
                            모델: {modelType}
                        </p>
                    </div>
                    <div className="bg-white rounded-xl p-3 border border-gray-200">
                        <p className="text-[11px] text-gray-500 m-0 mb-1">🚨 자세/낙상</p>
                        <p className="text-lg font-bold text-gray-900 m-0">
                            {fallMeta.label}
                        </p>
                        {/*
                            * 낙상 확률 (fall_probability)        : 모델이 'Fall' 클래스에 부여한 raw softmax 확률
                            * 판정 신뢰도 (confidence)            : 모델이 자신의 top prediction 에 대해 얼마나 확신하는지
                            *                                       (top class 가 Fall 이 아닐 수도 있음 — Lying_Down 등)
                        */}
                        <p className="text-[10px] text-gray-500 m-0 mt-1 leading-tight">
                            낙상 확률 {(((fall.fall_probability ?? 0)) * 100).toFixed(0)}%
                            <span className="text-gray-400">
                                {' · '}판정 신뢰도 {((fall.confidence ?? 0) * 100).toFixed(0)}%
                            </span>
                        </p>
                        {fall.model_prediction && (
                            <p className="text-[10px] text-gray-400 m-0 mt-0.5">
                                예측: {fall.model_prediction}
                                {modelInfo?.model_backend && (
                                    <span> · {modelInfo.model_backend}</span>
                                )}
                            </p>
                        )}
                    </div>
                </div>

                {/* 낙상 사건 active 배너 — 명시적 reset 까지 유지 */}
                {fall.fall_incident_active === true && (
                    <div className="mt-2 rounded-lg p-3 border border-red-300 bg-red-100 flex items-start gap-2">
                        <span className="text-lg">🚨</span>
                        <div className="flex-1 min-w-0">
                            <p className="text-[12px] font-bold text-red-800 m-0 leading-tight">
                                낙상 발생 — 확인 필요
                            </p>
                            <p className="text-[10px] text-red-700 m-0 mt-0.5 leading-tight">
                                자동으로 정상/누움 상태로 돌아가지 않습니다. 보호자 확인 후 아래 버튼으로 해제하세요.
                            </p>
                            {fall.fall_incident_id && (
                                <p className="text-[9px] text-red-600 m-0 mt-0.5 break-all">
                                    incident_id: {fall.fall_incident_id}
                                </p>
                            )}
                        </div>
                        <button
                            type="button"
                            onClick={resetDetectionState}
                            className="bg-white text-red-700 border border-red-400 rounded-md px-2 py-1 text-[11px] font-semibold cursor-pointer shrink-0 hover:bg-red-50"
                        >
                            사건 해제
                        </button>
                    </div>
                )}

                {/* 낙상 디버그 스트립 — 테스트 케이스(서있음/걷기/앉기/눕기/자기/낙상) 구분용 */}
                {fall.final_decision && (
                    <div
                        className={`mt-2 rounded-lg p-2 border text-[10px] ${
                            fall.final_decision === 'fall_emergency'
                                ? 'bg-red-50 border-red-200'
                                : fall.final_decision === 'fall_suspected'
                                    ? 'bg-orange-50 border-orange-200'
                                    : fall.final_decision === 'movement_pending' || fall.final_decision === 'checking_transition'
                                        ? 'bg-yellow-50 border-yellow-300'
                                        : fall.final_decision === 'lying_suppressed'
                                            ? 'bg-blue-50 border-blue-200'
                                            : 'bg-slate-50 border-slate-200'
                        }`}
                    >
                        <div className="flex flex-wrap gap-x-3 gap-y-1 text-gray-700 leading-snug">
                            <span><b>final</b>: {fall.final_decision}</span>
                            <span><b>alert</b>: {fall.alert_triggered ? '🔔 ON' : 'off'}</span>
                            <span>
                                <b>dyn_event</b>:{' '}
                                <span className={fall.dynamic_fall_event ? 'text-red-700 font-semibold' : 'text-gray-500'}>
                                    {fall.dynamic_fall_event ? '✓' : '✗'}
                                </span>
                            </span>
                            <span><b>P(Fall)</b>: {((fall.fall_probability ?? 0) * 100).toFixed(1)}%</span>
                            <span><b>conf</b>: {((fall.confidence ?? 0) * 100).toFixed(1)}%</span>
                            <span><b>pred</b>: {fall.model_prediction ?? '-'}</span>
                            <span><b>drop</b>: {(fall.vertical_drop ?? 0).toFixed(3)}</span>
                            <span><b>motion</b>: {(fall.motion_score ?? 0).toFixed(4)}</span>
                            <span><b>Δθ</b>: {(fall.torso_angle_change ?? 0).toFixed(1)}°</span>
                            <span><b>seq</b>: {fall.consecutive_fall_windows ?? 0}</span>
                            {(fall.pending_windows ?? 0) > 0 && (
                                <span><b>to_emerg</b>: {fall.pending_windows}</span>
                            )}
                            {fall.incident_state && (
                                <span>
                                    <b>incident</b>:{' '}
                                    <span className={fall.incident_state === 'active' ? 'text-red-700 font-semibold' : 'text-gray-500'}>
                                        {fall.incident_state}
                                    </span>
                                </span>
                            )}
                            {fall.latch_allowed !== undefined && (
                                <span>
                                    <b>latch</b>:{' '}
                                    <span className={fall.latch_allowed ? 'text-red-700 font-semibold' : 'text-gray-500'}>
                                        {fall.latch_allowed ? 'allowed' : 'blocked'}
                                    </span>
                                </span>
                            )}
                        </div>
                        {fall.latch_block_reason && (
                            <p className="text-[10px] text-amber-700 m-0 mt-1 leading-tight">
                                ⛔ latch_blocked: {fall.latch_block_reason}
                            </p>
                        )}
                        {fall.dynamic_gate_reason && (
                            <p className={`text-[10px] m-0 mt-1 leading-tight ${
                                fall.dynamic_fall_event ? 'text-red-700' : 'text-gray-600'
                            }`}>
                                {fall.dynamic_fall_event ? '⚡ dynamic_event:' : '○ static_state:'} {fall.dynamic_gate_reason}
                            </p>
                        )}
                        {fall.suppression_reason && (
                            <p className="text-[10px] text-blue-700 m-0 mt-1 leading-tight">
                                ⓘ suppress: {fall.suppression_reason}
                            </p>
                        )}
                        {fall.transition_state && (
                            <p className="text-[10px] text-yellow-800 m-0 mt-1 leading-tight">
                                ⏳ transition: {fall.transition_state}
                            </p>
                        )}
                        {fall.alert_reason && (
                            <p className="text-[10px] text-red-700 m-0 mt-1 leading-tight">
                                🔔 alert: {fall.alert_reason}
                            </p>
                        )}
                    </div>
                )}

                {/* 복약 객체 리스트 */}
                {medObjects.length > 0 && (
                    <div className="mt-3 bg-white rounded-xl p-3 border border-gray-200">
                        <p className="text-[11px] font-semibold text-gray-500 m-0 mb-2">감지 객체</p>
                        <div className="flex flex-col gap-1">
                            {medObjects.map((d, i) => (
                                <div
                                    key={i}
                                    className="flex justify-between text-xs bg-slate-50 rounded px-2 py-1"
                                >
                                    <span className="font-semibold text-green-700">{d.label}</span>
                                    <span className="text-gray-500">
                                        {(d.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {streaming && (
                    <p className="text-center text-[10px] text-gray-400 mt-2">
                        전송 주기 {FRAME_INTERVAL_MS}ms · 응답 {lastLatency}ms
                    </p>
                )}
            </div>
        </div>
    );
}

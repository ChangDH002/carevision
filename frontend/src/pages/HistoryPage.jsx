import { useEffect, useState } from 'react';
import { useStore } from '../store';
import { CV, SHADOW } from '../styles/cv';

function EmergencyDetail({ detection, patient, onBack, onOpenCamera }) {
    const startAt = detection?.detectedAt || detection?.sentAt || new Date().toISOString();
    const [elapsed, setElapsed] = useState(Math.floor((Date.now() - new Date(startAt).getTime()) / 1000));

    useEffect(() => {
        const t = setInterval(() => setElapsed((e) => e + 1), 1000);
        return () => clearInterval(t);
    }, []);

    const min = Math.floor(elapsed / 60);
    const sec = elapsed % 60;
    const confidence = Math.round((detection?.confidence ?? 0.874) * 100 * 10) / 10;
    const time = new Date(startAt).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });

    return (
        <div className="min-h-screen">
            {/* Danger hero */}
            <div
                className="text-white relative overflow-hidden"
                style={{
                    background: CV.dangerGrad,
                    padding: '20px 22px 28px',
                    borderRadius: '0 0 32px 32px',
                }}
            >
                <span className="absolute pointer-events-none" style={{ left: -60, bottom: -80, width: 280, height: 280, borderRadius: '50%', background: 'rgba(255,255,255,.06)' }} />
                <span className="absolute pointer-events-none" style={{ right: -40, top: -40, width: 160, height: 160, borderRadius: '50%', background: 'rgba(255,255,255,.07)' }} />

                <div className="relative">
                    <button
                        onClick={onBack}
                        className="cursor-pointer border-none flex items-center justify-center mt-3"
                        style={{
                            background: 'rgba(255,255,255,.18)',
                            color: '#fff',
                            width: 40, height: 40, borderRadius: '50%',
                        }}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                            <path d="M15 6l-6 6 6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                    </button>
                    <p className="m-0 mt-4 font-semibold opacity-90" style={{ fontSize: 13 }}>응급 감지</p>
                    <h1 className="m-0 mt-1 font-extrabold" style={{ fontSize: 26, letterSpacing: '-0.02em' }}>낙상 의심 상황</h1>
                    <p className="m-0 mt-1.5 opacity-90" style={{ fontSize: 12 }}>
                        위험도 3/5 · 발생 {4 + min}분 {37 + sec}초 경과
                    </p>
                    <div className="flex gap-1 mt-3">
                        {[1, 2, 3, 4, 5].map((i) => (
                            <span
                                key={i}
                                className="flex-1"
                                style={{
                                    height: 6,
                                    borderRadius: 9999,
                                    background: i <= 3 ? '#fff' : 'rgba(255,255,255,.3)',
                                }}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Live preview */}
            <div className="px-4 mt-3.5">
                <button
                    onClick={onOpenCamera}
                    className="w-full relative overflow-hidden border-none cursor-pointer"
                    style={{
                        background: CV.cameraBg,
                        borderRadius: 24,
                        aspectRatio: '16 / 10',
                        boxShadow: SHADOW.card,
                    }}
                >
                    <span
                        className="absolute inset-0 pointer-events-none"
                        style={{ background: 'radial-gradient(circle at 30% 30%, rgba(239,68,68,.25), transparent 60%)' }}
                    />
                    <span
                        className="absolute text-white font-extrabold"
                        style={{
                            top: 14, left: 14,
                            background: CV.danger, fontSize: 10,
                            padding: '3px 10px', borderRadius: 6, letterSpacing: '.05em',
                        }}
                    >
                        ● LIVE
                    </span>
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
                        <svg width="40" height="60" viewBox="0 0 40 60" fill="none">
                            <circle cx="20" cy="10" r="6" stroke="white" strokeWidth="2"/>
                            <line x1="20" y1="16" x2="20" y2="36" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                            <line x1="20" y1="22" x2="8" y2="30" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                            <line x1="20" y1="22" x2="32" y2="30" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                            <line x1="20" y1="36" x2="12" y2="52" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                            <line x1="20" y1="36" x2="28" y2="52" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                        <span className="text-white opacity-80" style={{ fontSize: 12 }}>실시간 카메라 스트림</span>
                    </div>
                </button>
            </div>

            {/* detection grid */}
            <div className="px-4 mt-3.5 grid grid-cols-2 gap-2.5">
                {[
                    { label: '감지 종류', value: '낙상', color: CV.dangerDeep },
                    { label: '발생 위치', value: '거실', color: CV.fg },
                    { label: 'LSTM 신뢰도', value: `${confidence}%`, color: CV.fg },
                    { label: '감지 시각', value: time, color: CV.fg },
                ].map((c) => (
                    <div
                        key={c.label}
                        style={{
                            background: '#fff',
                            borderRadius: 16,
                            padding: 14,
                            boxShadow: SHADOW.card,
                            border: '1px solid rgba(15,23,42,.04)',
                        }}
                    >
                        <p
                            className="m-0 font-bold uppercase"
                            style={{ fontSize: 10, color: CV.fgMuted, letterSpacing: '.06em' }}
                        >
                            {c.label}
                        </p>
                        <p className="m-0 mt-1 font-extrabold" style={{ fontSize: 18, color: c.color }}>{c.value}</p>
                    </div>
                ))}
            </div>

            {/* actions */}
            <div className="px-4 mt-3.5 flex flex-col gap-2.5">
                <button
                    className="w-full cursor-pointer font-bold border-none flex items-center justify-center gap-2"
                    style={{
                        background: CV.dangerGrad,
                        color: '#fff',
                        padding: '15px 16px',
                        borderRadius: 16,
                        fontSize: 15,
                        boxShadow: '0 8px 16px rgba(220,38,38,.25)',
                        fontFamily: 'inherit',
                    }}
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                        <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"
                              stroke="#fff" strokeWidth="2" strokeLinejoin="round"/>
                    </svg>
                    119 바로 전화
                </button>
                <button
                    className="w-full cursor-pointer font-bold"
                    style={{
                        background: '#fff',
                        color: CV.primary,
                        padding: '15px 16px',
                        borderRadius: 16,
                        fontSize: 15,
                        border: `1.5px solid ${CV.primary}`,
                        fontFamily: 'inherit',
                    }}
                >
                    다른 보호자에게 알리기
                </button>
            </div>

            <div className="text-center mt-3.5">
                <button
                    className="bg-transparent border-none cursor-pointer underline"
                    style={{ color: CV.fgMuted, fontSize: 12, fontFamily: 'inherit' }}
                >
                    오탐지로 표시
                </button>
            </div>
        </div>
    );
}

export default function HistoryPage({ patient, focus, onClearFocus, onOpenCamera, onBack }) {
    const allDetections = useStore((s) => s.detections);
    const detections = patient
        ? allDetections.filter((d) => !d.patient || d.patient.id === patient.id)
        : allDetections;
    const [view, setView] = useState(focus ? 'detail' : 'list');
    const [current, setCurrent] = useState(focus || null);

    useEffect(() => {
        if (focus) {
            setCurrent(focus);
            setView('detail');
        }
    }, [focus]);

    if (view === 'detail') {
        const matchedDetection = current?.detectedAt
            ? current
            : allDetections.find((d) =>
                  d.type === (current?.type || 'FALL') &&
                  Math.abs(new Date(d.detectedAt).getTime() - new Date(current?.sentAt || 0).getTime()) < 60000
              ) || current;

        return (
            <EmergencyDetail
                detection={matchedDetection}
                patient={patient}
                onBack={() => {
                    if (focus && onBack) {
                        onBack();
                    } else {
                        setView('list');
                        onClearFocus?.();
                    }
                }}
                onOpenCamera={onOpenCamera}
            />
        );
    }

    const TONE = {
        FALL:       { label: '낙상', bg: CV.dangerTint, color: CV.dangerDeep },
        MEDICATION: { label: '복약', bg: CV.successTint, color: CV.successText },
        NORMAL:     { label: '정상', bg: CV.primaryTint, color: CV.primaryText },
    };
    const typeLabel = (t) => TONE[t] || TONE.NORMAL;

    return (
        <div className="min-h-screen">
            {/* Hero */}
            <div
                className="text-white relative overflow-hidden"
                style={{
                    background: CV.primaryGradHero,
                    padding: '20px 22px 28px',
                    borderRadius: '0 0 32px 32px',
                }}
            >
                <span className="absolute pointer-events-none" style={{ left: -60, bottom: -80, width: 280, height: 280, borderRadius: '50%', background: 'rgba(255,255,255,.06)' }} />
                <span className="absolute pointer-events-none" style={{ right: -40, top: -40, width: 160, height: 160, borderRadius: '50%', background: 'rgba(255,255,255,.07)' }} />

                <div className="relative">
                    <h1 className="m-0 mt-3 font-extrabold" style={{ fontSize: 24, letterSpacing: '-0.01em' }}>감지 이력</h1>
                    <p className="m-0 mt-1 opacity-90" style={{ fontSize: 12 }}>
                        {patient?.name} 님 · 최근 7일
                    </p>
                </div>
            </div>

            {/* List */}
            <div className="px-4 mt-3.5">
                <div
                    style={{
                        background: '#fff',
                        borderRadius: 20,
                        padding: 6,
                        boxShadow: SHADOW.card,
                        border: '1px solid rgba(15,23,42,.04)',
                    }}
                >
                    {detections.length === 0 && (
                        <p className="text-center py-10 m-0" style={{ color: CV.fgFaint, fontSize: 13 }}>감지 이력이 없습니다</p>
                    )}
                    {detections.map((d, i, arr) => {
                        const t = typeLabel(d.type);
                        return (
                            <button
                                key={d.id}
                                onClick={() => { setCurrent(d); setView('detail'); }}
                                className="w-full text-left bg-transparent border-none cursor-pointer flex items-center gap-3"
                                style={{
                                    padding: '14px 12px',
                                    borderBottom: i === arr.length - 1 ? 'none' : `1px solid ${CV.divider}`,
                                    fontFamily: 'inherit',
                                }}
                            >
                                <span
                                    className="font-bold"
                                    style={{
                                        fontSize: 11, padding: '4px 10px',
                                        borderRadius: 9999, background: t.bg, color: t.color,
                                    }}
                                >
                                    {t.label}
                                </span>
                                <div className="flex-1 min-w-0">
                                    <p className="m-0 font-bold" style={{ fontSize: 14 }}>
                                        {d.type === 'FALL' ? '낙상 의심 상황' : d.type === 'MEDICATION' ? '복약 감지됨' : '정상 동작'}
                                    </p>
                                    <p className="m-0 mt-0.5" style={{ fontSize: 11, color: CV.fgFaint }}>
                                        신뢰도 {Math.round(d.confidence * 100)}% · {new Date(d.detectedAt).toLocaleString('ko-KR')}
                                    </p>
                                </div>
                                <span style={{ color: CV.fgFaint, fontSize: 16 }}>›</span>
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}

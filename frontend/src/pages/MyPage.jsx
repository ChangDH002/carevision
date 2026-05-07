import { useState } from 'react';
import { CV, SHADOW } from '../styles/cv';

function Toggle({ on, onChange }) {
    return (
        <button
            onClick={() => onChange(!on)}
            className="cursor-pointer border-none relative"
            style={{
                width: 44, height: 24, borderRadius: 9999,
                background: on ? CV.primary : '#CBD5E1',
                transition: 'background-color 0.15s ease',
            }}
        >
            <span
                className="absolute"
                style={{
                    top: 2, left: on ? 22 : 2,
                    width: 20, height: 20, borderRadius: '50%',
                    background: '#fff',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
                    transition: 'left 0.15s ease',
                }}
            />
        </button>
    );
}

export default function MyPage({ user, patients, onLogout }) {
    const [emergencyAlert, setEmergencyAlert] = useState(true);
    const [medAlert, setMedAlert] = useState(true);

    const cardStyle = {
        background: '#fff',
        borderRadius: 20,
        boxShadow: SHADOW.card,
        border: '1px solid rgba(15,23,42,.04)',
        overflow: 'hidden',
    };
    const sectionLabel = { fontSize: 12, color: CV.fgMuted, fontWeight: 600 };

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
                    <h1 className="m-0 mt-3 font-extrabold" style={{ fontSize: 24, letterSpacing: '-0.01em' }}>내 정보</h1>
                    <div className="flex items-center gap-3.5 mt-4">
                        <div
                            className="flex items-center justify-center font-extrabold shrink-0"
                            style={{
                                width: 56, height: 56, borderRadius: '50%',
                                background: '#fff', color: CV.primary, fontSize: 22,
                                boxShadow: '0 0 0 3px rgba(255,255,255,.3)',
                            }}
                        >
                            {user?.name?.[0] || '박'}
                        </div>
                        <div className="min-w-0">
                            <p className="m-0 font-extrabold" style={{ fontSize: 18 }}>{user?.name || '박보호'} 님</p>
                            <p className="m-0 mt-0.5 opacity-85" style={{ fontSize: 12 }}>
                                {user?.email || 'guardian@example.com'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* 피보호자 관리 */}
            <div className="px-4 mt-3.5">
                <p className="m-0 mb-2 px-1" style={sectionLabel}>피보호자 관리</p>
                <div style={cardStyle}>
                    {patients.map((p, i) => (
                        <div
                            key={p.id}
                            className="flex items-center gap-3"
                            style={{
                                padding: '14px 16px',
                                borderBottom: i === patients.length - 1 ? 'none' : `1px solid ${CV.divider}`,
                            }}
                        >
                            <div
                                className="flex items-center justify-center font-extrabold shrink-0"
                                style={{
                                    width: 40, height: 40, borderRadius: '50%',
                                    background: CV.primaryTint, color: CV.primary, fontSize: 14,
                                }}
                            >
                                {p.name?.[0] || '?'}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="m-0 font-bold" style={{ fontSize: 14 }}>{p.name} ({p.age}세)</p>
                                <p className="m-0 mt-0.5" style={{ fontSize: 12, color: CV.fgMuted }}>
                                    {p.cameraId ? '카메라 연결됨' : '카메라 미연결'} · {p.address?.split(' ').pop() || '거실'}
                                </p>
                            </div>
                            {p.cameraId ? (
                                <span
                                    className="inline-flex items-center gap-1.5 font-bold"
                                    style={{
                                        fontSize: 11,
                                        padding: '5px 12px',
                                        borderRadius: 9999,
                                        background: CV.successTint,
                                        color: CV.successText,
                                    }}
                                >
                                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: CV.success }} />
                                    연결
                                </span>
                            ) : (
                                <button
                                    className="bg-transparent border-none cursor-pointer font-bold"
                                    style={{ color: CV.primary, fontSize: 12 }}
                                >
                                    연결 ›
                                </button>
                            )}
                        </div>
                    ))}
                    <button
                        className="w-full text-center cursor-pointer bg-transparent font-bold"
                        style={{
                            padding: '14px 0',
                            color: CV.primary,
                            fontSize: 13,
                            borderTop: `1px solid ${CV.divider}`,
                            border: 'none',
                            borderTopWidth: 1,
                            borderTopStyle: 'solid',
                            borderTopColor: CV.divider,
                            fontFamily: 'inherit',
                        }}
                    >
                        + 피보호자 추가
                    </button>
                </div>
            </div>

            {/* 알림 설정 */}
            <div className="px-4 mt-3.5">
                <p className="m-0 mb-2 px-1" style={sectionLabel}>알림 설정</p>
                <div style={cardStyle}>
                    <div className="flex items-center gap-3" style={{ padding: '14px 16px', borderBottom: `1px solid ${CV.divider}` }}>
                        <div className="flex-1 min-w-0">
                            <p className="m-0 font-bold" style={{ fontSize: 14 }}>응급 알림</p>
                            <p className="m-0 mt-0.5" style={{ fontSize: 12, color: CV.fgMuted }}>낙상·무동작 즉시 알림</p>
                        </div>
                        <Toggle on={emergencyAlert} onChange={setEmergencyAlert} />
                    </div>
                    <div className="flex items-center gap-3" style={{ padding: '14px 16px', borderBottom: `1px solid ${CV.divider}` }}>
                        <div className="flex-1 min-w-0">
                            <p className="m-0 font-bold" style={{ fontSize: 14 }}>복약 누락 알림</p>
                            <p className="m-0 mt-0.5" style={{ fontSize: 12, color: CV.fgMuted }}>미복용 30분 후 알림</p>
                        </div>
                        <Toggle on={medAlert} onChange={setMedAlert} />
                    </div>
                    <div className="flex items-center gap-3" style={{ padding: '14px 16px' }}>
                        <div className="flex-1 min-w-0">
                            <p className="m-0 font-bold" style={{ fontSize: 14 }}>야간 무음</p>
                            <p className="m-0 mt-0.5" style={{ fontSize: 12, color: CV.fgMuted }}>23:00 ~ 07:00</p>
                        </div>
                        <button
                            className="bg-transparent border-none cursor-pointer font-bold"
                            style={{ color: CV.primary, fontSize: 12 }}
                        >
                            설정 ›
                        </button>
                    </div>
                </div>
            </div>

            {/* 계정 */}
            <div className="px-4 mt-3.5 mb-4">
                <p className="m-0 mb-2 px-1" style={sectionLabel}>계정</p>
                <div style={cardStyle}>
                    <button
                        className="w-full flex items-center justify-between bg-transparent border-none cursor-pointer text-left"
                        style={{ padding: '14px 16px', borderBottom: `1px solid ${CV.divider}`, fontFamily: 'inherit' }}
                    >
                        <p className="m-0 font-semibold" style={{ fontSize: 14 }}>비밀번호 변경</p>
                        <span style={{ color: CV.fgFaint }}>›</span>
                    </button>
                    <button
                        onClick={onLogout}
                        className="w-full text-left bg-transparent border-none cursor-pointer font-semibold"
                        style={{ padding: '14px 16px', color: CV.dangerDeep, fontSize: 14, fontFamily: 'inherit' }}
                    >
                        로그아웃
                    </button>
                </div>
            </div>
        </div>
    );
}

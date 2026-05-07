import { useState } from 'react';
import { store, useStore } from '../store';
import { CV, SHADOW } from '../styles/cv';

const CATEGORIES = [
    { id: 'all',        label: '전체' },
    { id: 'emergency',  label: '응급' },
    { id: 'medication', label: '복약' },
    { id: 'system',     label: '시스템' },
];

const TYPE_META = {
    FALL: {
        category: 'emergency',
        title: '낙상 의심 감지',
        bg: CV.dangerTint,
        color: CV.dangerDeep,
        icon: (color) => (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L2 22h20L12 2z" stroke={color} strokeWidth="2" strokeLinejoin="round"/>
                <line x1="12" y1="10" x2="12" y2="15" stroke={color} strokeWidth="2" strokeLinecap="round"/>
                <circle cx="12" cy="18" r="1" fill={color}/>
            </svg>
        ),
    },
    MEDICATION: {
        category: 'medication',
        title: '복약 알림',
        bg: CV.primaryTint,
        color: CV.primary,
        icon: (color) => (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <rect x="2" y="8" width="20" height="8" rx="4" stroke={color} strokeWidth="2"/>
                <line x1="12" y1="8" x2="12" y2="16" stroke={color} strokeWidth="2"/>
            </svg>
        ),
    },
    NORMAL: {
        category: 'system',
        title: '정상 동작',
        bg: CV.successTint,
        color: CV.successText,
        icon: (color) => (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="9" stroke={color} strokeWidth="2"/>
                <path d="M8 12l3 3 5-6" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
        ),
    },
    SYSTEM: {
        category: 'system',
        title: '시스템',
        bg: CV.primaryTintSoft,
        color: CV.primaryText,
        icon: (color) => (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="5" width="18" height="12" rx="2" stroke={color} strokeWidth="2"/>
                <circle cx="18" cy="11" r="1" fill={color}/>
            </svg>
        ),
    },
};

function timeAgo(iso) {
    const diff = Date.now() - new Date(iso).getTime();
    const m = Math.floor(diff / 60000);
    if (m < 1) return '방금 전';
    if (m < 60) return `${m}분 전`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}시간 전`;
    const d = Math.floor(h / 24);
    return `${d}일 전`;
}

export default function NotificationsPage({ onOpenEmergency }) {
    const [cat, setCat] = useState('all');
    const items = useStore((s) => s.notifications);

    const markAll = () => store.markAllRead();

    const normalizedType = (n) => {
        const t = (n.type || 'NORMAL').toUpperCase();
        return TYPE_META[t] ? t : 'NORMAL';
    };

    const filtered = cat === 'all'
        ? items
        : items.filter((n) => TYPE_META[normalizedType(n)].category === cat);

    const unreadEmergency = items.find((n) => !n.isRead && normalizedType(n) === 'FALL');

    const formatMessage = (n) => {
        const t = normalizedType(n);
        const patient = n.patient?.name;
        if (t === 'FALL') return `${patient ?? ''} 님 거실에서 낙상 패턴이 감지되었습니다. 즉시 확인해주세요.`;
        if (t === 'MEDICATION') return n.message || `${patient ?? ''} 님 복약이 완료되었습니다.`;
        return n.message || '알림';
    };

    const titleFor = (n, t) => {
        if (t === 'FALL') return '낙상 의심 감지';
        if (t === 'MEDICATION') return n.message?.includes('완료') ? '복약 완료 확인' : '복약 누락 알림';
        return TYPE_META[t].title;
    };

    return (
        <div className="min-h-screen">
            {/* Hero (ink/dark) */}
            <div
                className="text-white relative overflow-hidden"
                style={{
                    background: CV.inkGrad,
                    padding: '20px 22px 28px',
                    borderRadius: '0 0 32px 32px',
                }}
            >
                <span className="absolute pointer-events-none" style={{ left: -60, bottom: -80, width: 280, height: 280, borderRadius: '50%', background: 'rgba(255,255,255,.04)' }} />
                <span className="absolute pointer-events-none" style={{ right: -40, top: -40, width: 160, height: 160, borderRadius: '50%', background: 'rgba(255,255,255,.05)' }} />

                <div className="relative">
                    <div className="flex justify-between items-end mt-3">
                        <h1 className="m-0 font-extrabold" style={{ fontSize: 24, letterSpacing: '-0.01em' }}>알림</h1>
                        <button
                            onClick={markAll}
                            className="cursor-pointer border-none bg-transparent font-semibold"
                            style={{ color: 'rgba(255,255,255,.85)', fontSize: 12, fontFamily: 'inherit' }}
                        >
                            모두 읽음 처리
                        </button>
                    </div>

                    {/* category chips */}
                    <div className="mt-4 flex gap-1.5 overflow-x-auto scrollbar-hide">
                        {CATEGORIES.map((c) => {
                            const on = cat === c.id;
                            return (
                                <button
                                    key={c.id}
                                    onClick={() => setCat(c.id)}
                                    className="cursor-pointer border-none font-bold shrink-0"
                                    style={{
                                        padding: '6px 14px',
                                        borderRadius: 9999,
                                        background: on ? '#fff' : 'rgba(255,255,255,.12)',
                                        color: on ? CV.ink : '#fff',
                                        fontSize: 12,
                                        fontFamily: 'inherit',
                                        transition: 'background-color 0.15s ease',
                                    }}
                                >
                                    {c.label}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Emergency CTA */}
            {unreadEmergency && (
                <div className="px-4 mt-4">
                    <div
                        className="flex items-center gap-3.5"
                        style={{
                            background: CV.dangerGrad,
                            color: '#fff',
                            borderRadius: 20,
                            padding: 14,
                            boxShadow: '0 12px 28px rgba(220,38,38,.18)',
                        }}
                    >
                        <div
                            className="flex items-center justify-center shrink-0"
                            style={{
                                width: 44, height: 44, borderRadius: 14,
                                background: 'rgba(255,255,255,.18)',
                            }}
                        >
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                                <path d="M12 2L2 22h20L12 2z" stroke="white" strokeWidth="2" strokeLinejoin="round"/>
                                <line x1="12" y1="10" x2="12" y2="15" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                                <circle cx="12" cy="18" r="1" fill="white"/>
                            </svg>
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="m-0 font-extrabold" style={{ fontSize: 14 }}>응급 알림 확인 필요</p>
                            <p className="m-0 mt-0.5 opacity-90" style={{ fontSize: 11 }}>낙상 감지 후 3분 경과. 지금 확인하세요.</p>
                        </div>
                        <button
                            onClick={() => onOpenEmergency?.(unreadEmergency)}
                            className="cursor-pointer border-none font-bold shrink-0"
                            style={{
                                background: '#fff',
                                color: CV.dangerDeep,
                                borderRadius: 9999,
                                padding: '8px 14px',
                                fontSize: 12,
                                fontFamily: 'inherit',
                            }}
                        >
                            확인
                        </button>
                    </div>
                </div>
            )}

            {/* list card */}
            <div className="px-4 mt-4">
                <div
                    style={{
                        background: '#fff',
                        borderRadius: 20,
                        padding: 6,
                        boxShadow: SHADOW.card,
                        border: '1px solid rgba(15,23,42,.04)',
                    }}
                >
                    {filtered.length === 0 && (
                        <p className="text-center py-10 m-0" style={{ color: CV.fgFaint, fontSize: 13 }}>알림이 없습니다</p>
                    )}
                    {filtered.map((n, i, arr) => {
                        const t = normalizedType(n);
                        const meta = TYPE_META[t];
                        return (
                            <button
                                key={n.id}
                                onClick={() => {
                                    store.markRead(n.id);
                                    if (t === 'FALL') onOpenEmergency?.(n);
                                }}
                                className="w-full text-left bg-transparent border-none cursor-pointer flex gap-3"
                                style={{
                                    padding: '14px 12px',
                                    borderBottom: i === arr.length - 1 ? 'none' : `1px solid ${CV.divider}`,
                                    fontFamily: 'inherit',
                                }}
                            >
                                <div
                                    className="flex items-center justify-center shrink-0"
                                    style={{
                                        width: 44, height: 44, borderRadius: 14,
                                        background: meta.bg, color: meta.color,
                                    }}
                                >
                                    {meta.icon(meta.color)}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex justify-between items-start gap-2">
                                        <p className="m-0 font-bold" style={{ fontSize: 14, color: CV.fg }}>{titleFor(n, t)}</p>
                                        {!n.isRead && (
                                            <span
                                                className="shrink-0"
                                                style={{ width: 8, height: 8, borderRadius: '50%', background: CV.primary, marginTop: 6 }}
                                            />
                                        )}
                                    </div>
                                    <p className="m-0 mt-0.5" style={{ fontSize: 12, color: CV.fgMuted, lineHeight: 1.45 }}>
                                        {formatMessage(n)}
                                    </p>
                                    <p className="m-0 mt-1.5" style={{ fontSize: 11, color: CV.fgFaint }}>{timeAgo(n.sentAt)}</p>
                                </div>
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}

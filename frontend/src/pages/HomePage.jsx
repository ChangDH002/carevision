import { useStore } from '../store';
import { CV, SHADOW, meal as mealLabel } from '../styles/cv';

function Hero({ children, tall = false, variant = 'primary' }) {
    const grad = variant === 'danger' ? CV.dangerGrad
               : variant === 'ink' ? CV.inkGrad
               : CV.primaryGradHero;
    return (
        <div
            className="text-white relative overflow-hidden"
            style={{
                background: grad,
                padding: tall ? '20px 22px 110px' : '20px 22px 28px',
                borderRadius: '0 0 32px 32px',
            }}
        >
            <span className="absolute pointer-events-none" style={{ left: -60, bottom: -80, width: 280, height: 280, borderRadius: '50%', background: 'rgba(255,255,255,.06)' }} />
            <span className="absolute pointer-events-none" style={{ right: -40, top: -40, width: 160, height: 160, borderRadius: '50%', background: 'rgba(255,255,255,.07)' }} />
            <div className="relative">{children}</div>
        </div>
    );
}

function StatusPill({ children, dotColor = CV.success }) {
    return (
        <span
            className="inline-flex items-center gap-2 rounded-full text-xs font-semibold"
            style={{ background: 'rgba(255,255,255,.18)', backdropFilter: 'blur(8px)', padding: '6px 14px 6px 6px' }}
        >
            <span
                className="inline-flex items-center justify-center"
                style={{ width: 22, height: 22, borderRadius: '50%', background: dotColor }}
            >
                <span style={{ width: 8, height: 8, background: '#fff', borderRadius: '50%' }} />
            </span>
            {children}
        </span>
    );
}

function Tile({ children, featured = false, onClick }) {
    return (
        <button
            onClick={onClick}
            className="text-left font-inherit cursor-pointer flex flex-col justify-between"
            style={{
                background: featured ? CV.inkGrad : '#fff',
                color: featured ? '#fff' : CV.fg,
                borderRadius: 24,
                padding: 18,
                minHeight: 144,
                boxShadow: SHADOW.card,
                border: '1px solid rgba(15,23,42,.04)',
            }}
        >
            {children}
        </button>
    );
}

function TileIcon({ name, bg, color, featured = false }) {
    const c = featured ? '#fff' : (color || CV.primary);
    const b = featured ? 'rgba(255,255,255,.12)' : (bg || CV.primaryTint);
    let inner = null;
    switch (name) {
        case 'warning':
            inner = (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 22h20L12 2z" stroke={c} strokeWidth="2" strokeLinejoin="round"/>
                    <line x1="12" y1="10" x2="12" y2="15" stroke={c} strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="12" cy="18" r="1" fill={c}/>
                </svg>
            );
            break;
        case 'pill':
            inner = (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <rect x="2" y="8" width="20" height="8" rx="4" stroke={c} strokeWidth="2"/>
                    <line x1="12" y1="8" x2="12" y2="16" stroke={c} strokeWidth="2"/>
                </svg>
            );
            break;
        case 'check':
            inner = (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="9" stroke={c} strokeWidth="2"/>
                    <path d="M8 12l3 3 5-6" stroke={c} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
            );
            break;
        case 'clock':
            inner = (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="9" stroke={c} strokeWidth="2"/>
                    <path d="M12 7v5l3 2" stroke={c} strokeWidth="2" strokeLinecap="round"/>
                </svg>
            );
            break;
        default:
            inner = null;
    }
    return (
        <div
            className="flex items-center justify-center"
            style={{ width: 44, height: 44, borderRadius: 14, background: b, color: c }}
        >
            {inner}
        </div>
    );
}

function TileBody({ title, sub, num, suffix, featured = false }) {
    return (
        <div>
            <h3 className="font-extrabold m-0 mt-2" style={{ fontSize: 15, letterSpacing: '-0.01em', color: 'inherit' }}>{title}</h3>
            {sub && (
                <p
                    className="m-0 leading-tight"
                    style={{ fontSize: 11, color: featured ? 'rgba(255,255,255,.7)' : CV.fgMuted, marginTop: 2 }}
                >
                    {sub}
                </p>
            )}
            <div className="flex items-center justify-between mt-2.5">
                <span className="font-extrabold" style={{ fontSize: 22, letterSpacing: '-0.02em', color: 'inherit' }}>
                    {num}
                    {suffix && (
                        <small
                            className="font-semibold ml-0.5"
                            style={{ fontSize: 11, color: featured ? 'rgba(255,255,255,.6)' : CV.fgMuted }}
                        >
                            {suffix}
                        </small>
                    )}
                </span>
                <span style={{ color: featured ? 'rgba(255,255,255,.6)' : CV.fgFaint }}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                </span>
            </div>
        </div>
    );
}

function LiveTile({ label = '거실 카메라', subLabel = '방금 전', onClick }) {
    return (
        <button
            onClick={onClick}
            className="w-full flex items-center gap-3.5 text-left font-inherit cursor-pointer"
            style={{
                background: '#fff',
                borderRadius: 24,
                padding: 14,
                boxShadow: SHADOW.card,
                border: '1px solid rgba(15,23,42,.04)',
            }}
        >
            <div
                className="relative overflow-hidden flex items-center justify-center shrink-0"
                style={{ width: 84, height: 84, borderRadius: 18, background: CV.inkGrad }}
            >
                <span
                    className="absolute inset-0"
                    style={{ background: 'radial-gradient(circle at 30% 30%,rgba(47,107,255,.4),transparent 60%)' }}
                />
                <span
                    className="absolute text-white font-extrabold"
                    style={{ top: 6, left: 6, background: CV.danger, fontSize: 8, padding: '2px 6px', borderRadius: 4, letterSpacing: '.06em' }}
                >
                    LIVE
                </span>
                <div
                    className="relative flex items-center justify-center"
                    style={{ width: 32, height: 32, borderRadius: '50%', background: '#fff', zIndex: 1 }}
                >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill={CV.primary}>
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                </div>
            </div>
            <div className="flex-1 min-w-0">
                <div
                    className="font-bold uppercase"
                    style={{ fontSize: 10, color: CV.primary, letterSpacing: '.06em' }}
                >
                    실시간 카메라
                </div>
                <h4 className="m-0.5 font-extrabold" style={{ fontSize: 16, margin: '2px 0 4px' }}>{label}</h4>
                <div style={{ fontSize: 11, color: CV.fgMuted }}>{subLabel} · 보러가기 ›</div>
            </div>
        </button>
    );
}

function StatusBadge({ tone = 'success', children }) {
    const tones = {
        success: { bg: CV.successTint, fg: CV.successText },
        danger:  { bg: CV.dangerTint,  fg: '#B91C1C' },
        warn:    { bg: CV.warnTint,    fg: CV.warnText },
        info:    { bg: CV.primaryTint, fg: CV.primaryText },
    };
    const t = tones[tone] || tones.success;
    return (
        <span
            className="inline-flex items-center gap-1 font-bold rounded-full"
            style={{ fontSize: 11, padding: '5px 12px', background: t.bg, color: t.fg }}
        >
            {children}
        </span>
    );
}

export default function HomePage({
    user,
    patients,
    selectedPatient,
    onSelectPatient,
    onGoNotifications,
    onGoMedication,
    onOpenCamera,
    unreadCount,
}) {
    const meds = useStore((s) => (selectedPatient ? s.medications[selectedPatient.id] || [] : []));
    const logs = useStore((s) => (selectedPatient ? s.logs[selectedPatient.id] || [] : []));
    const detections = useStore((s) => s.detections);

    const recentFall = detections.find(
        (d) => d.type === 'FALL' && Date.now() - new Date(d.detectedAt).getTime() < 30 * 60 * 1000
    );

    const sortedMeds = [...meds].sort((a, b) => a.scheduleTime.localeCompare(b.scheduleTime));

    const todayStr = (() => {
        const d = new Date();
        return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    })();

    const taken = logs.filter((l) => l.status === 'TAKEN' && l.loggedAt?.slice(0, 10) === todayStr).length;
    const total = meds.length;

    const nextMed = sortedMeds.find((m) => {
        const log = [...logs]
            .filter((l) => l.medicationId === m.id && l.loggedAt?.slice(0, 10) === todayStr)
            .sort((a, b) => new Date(b.loggedAt) - new Date(a.loggedAt))[0];
        return !log || log.status !== 'TAKEN';
    });

    const cyclePatient = () => {
        if (patients.length < 2) return;
        const i = patients.findIndex((p) => p.id === selectedPatient?.id);
        onSelectPatient(patients[(i + 1) % patients.length]);
    };

    const guardianName = user?.name || '보호자';
    const patientLabel = selectedPatient?.relation
        ? `${selectedPatient.relation} ${selectedPatient.name}`
        : selectedPatient?.name || '피보호자';

    return (
        <div className="min-h-screen">
            <Hero tall>
                {/* greeting + bell */}
                <div className="flex items-center gap-3 mt-4">
                    <div
                        className="flex items-center justify-center font-extrabold shrink-0"
                        style={{
                            width: 40, height: 40, borderRadius: '50%',
                            background: '#fff', color: CV.primary, fontSize: 16,
                            boxShadow: '0 0 0 3px rgba(255,255,255,.3)',
                        }}
                    >
                        {selectedPatient?.name?.[0] || '?'}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="m-0 opacity-90" style={{ fontSize: 12 }}>
                            안녕하세요, {guardianName} 님 👋
                        </p>
                        <button
                            onClick={cyclePatient}
                            className="inline-flex items-center gap-1 font-semibold cursor-pointer mt-1"
                            style={{
                                background: 'rgba(255,255,255,.18)',
                                backdropFilter: 'blur(8px)',
                                border: 'none', color: '#fff',
                                padding: '4px 12px', borderRadius: 9999,
                                fontSize: 12, fontFamily: 'inherit',
                            }}
                        >
                            {patientLabel} 님 ▾
                        </button>
                    </div>
                    <button
                        onClick={onGoNotifications}
                        className="relative flex items-center justify-center cursor-pointer"
                        style={{
                            width: 40, height: 40, borderRadius: '50%',
                            background: 'rgba(255,255,255,.18)',
                            backdropFilter: 'blur(8px)',
                            border: 'none', color: '#fff',
                        }}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                            <path d="M6 8a6 6 0 1 1 12 0c0 5 2 6 2 6H4s2-1 2-6z"
                                  stroke="white" strokeWidth="2" strokeLinejoin="round"/>
                            <path d="M10 19a2 2 0 0 0 4 0" stroke="white" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                        {unreadCount > 0 && (
                            <span
                                className="absolute"
                                style={{
                                    top: 4, right: 4, width: 8, height: 8,
                                    background: CV.accent, borderRadius: '50%',
                                    boxShadow: '0 0 0 2px rgba(255,255,255,.3)',
                                }}
                            />
                        )}
                    </button>
                </div>

                <h1 className="font-extrabold m-0 mt-5" style={{ fontSize: 26, letterSpacing: '-0.02em', lineHeight: 1.2 }}>
                    오늘도 {patientLabel}은<br />
                    <span style={{ color: recentFall ? '#fff' : CV.accent }}>
                        {recentFall ? '확인이 필요해요' : '안전하게'}
                    </span>{' '}
                    {recentFall ? '' : '계세요'}
                </h1>

                <div className="mt-3.5">
                    <StatusPill dotColor={recentFall ? CV.danger : CV.success}>
                        {recentFall ? '낙상 의심 감지 · 확인 필요' : '실시간 안전 모니터링 중 · 3분 전 활동 감지'}
                    </StatusPill>
                </div>
            </Hero>

            {/* tile grid (overlaps hero) */}
            <div
                className="grid grid-cols-2 gap-3 relative z-10"
                style={{ margin: '-72px 16px 0' }}
            >
                <Tile featured onClick={onOpenCamera}>
                    <TileIcon name="warning" featured />
                    <TileBody
                        title="낙상 감지"
                        sub={recentFall ? '방금 감지됨' : '지난 7일 · 0건'}
                        num={recentFall ? '확인' : '정상'}
                        featured
                    />
                </Tile>
                <Tile onClick={onGoMedication}>
                    <TileIcon name="pill" />
                    <TileBody
                        title="오늘 복약"
                        sub={nextMed ? `다음: ${mealLabel(nextMed.scheduleTime)} ${nextMed.name} · ${nextMed.scheduleTime}` : '오늘 일정 없음'}
                        num={taken}
                        suffix={`/${total || 0}`}
                    />
                </Tile>
                <Tile>
                    <TileIcon name="check" bg={CV.successTint} color={CV.successText} />
                    <TileBody title="활동 점수" sub="지난 24시간" num="87" suffix="/100" />
                </Tile>
                <Tile>
                    <TileIcon name="clock" bg={CV.warnTint} color={CV.warnText} />
                    <TileBody title="무동작 감지" sub="마지막 활동 이후" num="12" suffix="분" />
                </Tile>
            </div>

            {/* live row */}
            <div className="mt-3.5 mx-4">
                <LiveTile onClick={onOpenCamera} />
            </div>

            {/* today's meds quick view */}
            <div className="mt-3.5 mx-4">
                <div className="flex justify-between items-center px-1 pb-2.5">
                    <h3 className="font-extrabold m-0" style={{ fontSize: 15 }}>오늘 복약</h3>
                    <button
                        onClick={onGoMedication}
                        className="bg-transparent border-none cursor-pointer font-bold"
                        style={{ color: CV.primary, fontSize: 12 }}
                    >
                        전체보기 ›
                    </button>
                </div>
                <div
                    style={{
                        background: '#fff', borderRadius: 20, padding: 6,
                        boxShadow: SHADOW.card, border: '1px solid rgba(15,23,42,.04)',
                    }}
                >
                    {sortedMeds.length === 0 && (
                        <p className="text-center py-6" style={{ color: CV.fgFaint, fontSize: 13 }}>
                            등록된 복약이 없습니다
                        </p>
                    )}
                    {sortedMeds.slice(0, 4).map((m, i, arr) => {
                        const log = [...logs]
                            .filter((l) => l.medicationId === m.id && l.loggedAt?.slice(0, 10) === todayStr)
                            .sort((a, b) => new Date(b.loggedAt) - new Date(a.loggedAt))[0];
                        const isTaken = log?.status === 'TAKEN';
                        const isMissed = log?.status === 'MISSED';
                        const tone = isTaken ? 'success' : isMissed ? 'danger' : 'info';
                        const lbl = isTaken ? '완료' : isMissed ? '누락' : '대기';
                        return (
                            <div
                                key={m.id}
                                className="flex items-center gap-3"
                                style={{
                                    padding: '12px',
                                    borderBottom: i === arr.length - 1 ? 'none' : `1px solid ${CV.divider}`,
                                }}
                            >
                                <div
                                    className="flex items-center justify-center shrink-0"
                                    style={{
                                        width: 36, height: 36, borderRadius: 10,
                                        background: isTaken ? CV.successTint : CV.primaryTint,
                                        color: isTaken ? CV.successText : CV.primary,
                                    }}
                                >
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                                        <rect x="2" y="8" width="20" height="8" rx="4" stroke="currentColor" strokeWidth="2"/>
                                        <line x1="12" y1="8" x2="12" y2="16" stroke="currentColor" strokeWidth="2"/>
                                    </svg>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="m-0 font-bold" style={{ fontSize: 14 }}>
                                        {mealLabel(m.scheduleTime)} {m.name}
                                    </p>
                                    <p className="m-0 mt-0.5" style={{ fontSize: 11, color: CV.fgFaint }}>
                                        {m.scheduleTime} · {m.dosage}
                                    </p>
                                </div>
                                <StatusBadge tone={tone}>{lbl}</StatusBadge>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}

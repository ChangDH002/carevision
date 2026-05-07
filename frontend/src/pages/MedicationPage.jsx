import { useState } from 'react';
import { store, useStore } from '../store';
import { CV, SHADOW, meal as mealLabel } from '../styles/cv';

const DAYS_KR = { MON: '월', TUE: '화', WED: '수', THU: '목', FRI: '금', SAT: '토', SUN: '일' };
const ALL_DAYS = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'];

function MedicationFormModal({ onClose, onSubmit, initial }) {
    const [form, setForm] = useState(
        initial || { name: '', dosage: '1정', scheduleTime: '08:00', days: [...ALL_DAYS] }
    );

    const toggleDay = (d) =>
        setForm((f) => ({
            ...f,
            days: f.days.includes(d) ? f.days.filter((x) => x !== d) : [...f.days, d],
        }));

    const submit = () => {
        if (!form.name.trim()) return alert('약 이름을 입력해주세요');
        if (form.days.length === 0) return alert('요일을 하나 이상 선택해주세요');
        onSubmit({ ...form, days: form.days.join(',') });
    };

    const inputStyle = {
        width: '100%',
        padding: '12px 14px',
        border: `1px solid ${CV.border}`,
        borderRadius: 14,
        fontSize: 14,
        outline: 'none',
        background: CV.surfaceInput,
        boxSizing: 'border-box',
        fontFamily: 'inherit',
    };

    return (
        <div className="fixed inset-0 z-[70] flex items-end justify-center" style={{ background: 'rgba(0,0,0,0.5)' }}>
            <div
                className="w-full max-w-[480px] animate-[slideUp_0.25s_ease-out]"
                style={{
                    background: '#fff',
                    borderRadius: '32px 32px 0 0',
                    padding: '24px 22px 32px',
                }}
            >
                <div className="flex justify-between items-center mb-4">
                    <h2 className="m-0 font-extrabold" style={{ fontSize: 20, color: CV.fg }}>복약 스케줄 추가</h2>
                    <button
                        onClick={onClose}
                        className="bg-transparent border-none cursor-pointer leading-none"
                        style={{ fontSize: 24, color: CV.fgFaint }}
                    >
                        ×
                    </button>
                </div>

                <div className="flex flex-col gap-3">
                    <div>
                        <label className="block mb-1.5 font-semibold" style={{ fontSize: 12, color: CV.fgMuted }}>약 이름</label>
                        <input
                            style={inputStyle}
                            placeholder="예: 혈압약"
                            value={form.name}
                            onChange={(e) => setForm({ ...form, name: e.target.value })}
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="block mb-1.5 font-semibold" style={{ fontSize: 12, color: CV.fgMuted }}>용량</label>
                            <input
                                style={inputStyle}
                                placeholder="1정"
                                value={form.dosage}
                                onChange={(e) => setForm({ ...form, dosage: e.target.value })}
                            />
                        </div>
                        <div>
                            <label className="block mb-1.5 font-semibold" style={{ fontSize: 12, color: CV.fgMuted }}>복용 시각</label>
                            <input
                                type="time"
                                style={inputStyle}
                                value={form.scheduleTime}
                                onChange={(e) => setForm({ ...form, scheduleTime: e.target.value })}
                            />
                        </div>
                    </div>
                    <div>
                        <label className="block mb-1.5 font-semibold" style={{ fontSize: 12, color: CV.fgMuted }}>복용 요일</label>
                        <div className="flex gap-1.5 flex-wrap">
                            {ALL_DAYS.map((d) => {
                                const on = form.days.includes(d);
                                return (
                                    <button
                                        key={d}
                                        onClick={() => toggleDay(d)}
                                        className="cursor-pointer font-bold"
                                        style={{
                                            width: 40, height: 40, borderRadius: '50%',
                                            background: on ? CV.primary : '#fff',
                                            color: on ? '#fff' : CV.fgMuted,
                                            border: `1px solid ${on ? CV.primary : CV.border}`,
                                            fontSize: 13, fontFamily: 'inherit',
                                            transition: 'background-color 0.15s ease',
                                        }}
                                    >
                                        {DAYS_KR[d]}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                    <button
                        onClick={submit}
                        className="w-full cursor-pointer font-bold border-none mt-3"
                        style={{
                            padding: '15px 16px',
                            background: CV.primaryGrad,
                            color: '#fff',
                            borderRadius: 16,
                            fontSize: 15,
                            fontFamily: 'inherit',
                            boxShadow: SHADOW.cta,
                        }}
                    >
                        추가하기
                    </button>
                </div>
            </div>
        </div>
    );
}

export default function MedicationPage({ patient, patients, onSelectPatient }) {
    const [tab, setTab] = useState('today');
    const [showForm, setShowForm] = useState(false);
    const [showPatientList, setShowPatientList] = useState(false);

    const medications = useStore((s) => (patient ? s.medications[patient.id] || [] : []));
    const logs = useStore((s) => (patient ? s.logs[patient.id] || [] : []));

    const sorted = [...medications].sort((a, b) => a.scheduleTime.localeCompare(b.scheduleTime));

    const today = (() => {
        const d = new Date();
        return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    })();

    const taken = logs.filter((l) => l.status === 'TAKEN' && l.loggedAt?.slice(0, 10) === today).length;
    const total = medications.length;
    const pct = total > 0 ? (taken / total) * 100 : 0;

    const getStatus = (m) => {
        const log = [...logs]
            .filter((l) => l.medicationId === m.id && l.loggedAt?.slice(0, 10) === today)
            .sort((a, b) => new Date(b.loggedAt) - new Date(a.loggedAt))[0];
        if (log?.status === 'TAKEN') {
            const at = new Date(log.loggedAt);
            const hh = String(at.getHours()).padStart(2, '0');
            const mm = String(at.getMinutes()).padStart(2, '0');
            return {
                kind: 'taken',
                label: log.source === 'manual' ? '수동 체크' : 'AI 감지',
                tone: 'success',
                line: `${m.dosage} · ${hh}:${mm} 감지됨`,
            };
        }
        if (log?.status === 'MISSED') {
            return { kind: 'missed', label: '누락', tone: 'danger', line: `${m.dosage} · 복용 누락` };
        }
        return { kind: 'pending', label: '대기중', tone: 'info', line: `${m.dosage} · 복용 예정` };
    };

    const handleAdd = async (data) => {
        if (!patient) return;
        await store.addMedication(patient.id, data);
        setShowForm(false);
    };

    const handleDelete = (medId) => {
        if (!patient) return;
        if (!confirm('이 스케줄을 삭제하시겠습니까?')) return;
        store.deleteMedication(patient.id, medId);
    };

    const handleManualCheck = (medId) => {
        if (!patient) return;
        store.logMedication(patient.id, medId, 'TAKEN');
    };

    const TONE = {
        success: { bg: CV.successTint, fg: CV.successText },
        danger:  { bg: CV.dangerTint,  fg: '#B91C1C' },
        info:    { bg: CV.primaryTint, fg: CV.primaryText },
    };

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
                    <div className="flex justify-between items-start">
                        <div>
                            <p className="m-0 opacity-85" style={{ fontSize: 13 }}>{patient?.name} 님</p>
                            <h1 className="m-0 font-extrabold mt-1" style={{ fontSize: 24, letterSpacing: '-0.01em' }}>
                                오늘 복약은<br />
                                <span style={{ color: CV.accent }}>{taken}/{total || 0} 완료</span>
                            </h1>
                        </div>
                        <button
                            onClick={() => setShowForm(true)}
                            className="cursor-pointer font-semibold border-none"
                            style={{
                                background: 'rgba(255,255,255,.25)',
                                color: '#fff',
                                fontSize: 13,
                                borderRadius: 9999,
                                padding: '8px 14px',
                                fontFamily: 'inherit',
                            }}
                        >
                            + 추가
                        </button>
                    </div>

                    {/* patient switcher */}
                    {patients?.length > 1 && (
                        <div className="mt-3">
                            <button
                                onClick={() => setShowPatientList((v) => !v)}
                                className="cursor-pointer font-semibold border-none inline-flex items-center gap-1"
                                style={{
                                    background: 'rgba(255,255,255,.18)',
                                    backdropFilter: 'blur(8px)',
                                    color: '#fff',
                                    fontSize: 12,
                                    borderRadius: 9999,
                                    padding: '4px 12px',
                                    fontFamily: 'inherit',
                                }}
                            >
                                {patient?.name} 님 ▾
                            </button>
                            {showPatientList && (
                                <div
                                    className="mt-2 p-1.5"
                                    style={{
                                        background: '#fff',
                                        borderRadius: 14,
                                        boxShadow: SHADOW.card,
                                    }}
                                >
                                    {patients.map((p) => (
                                        <button
                                            key={p.id}
                                            onClick={() => { onSelectPatient(p); setShowPatientList(false); }}
                                            className="w-full text-left cursor-pointer border-none bg-transparent"
                                            style={{
                                                padding: '10px 12px',
                                                borderRadius: 10,
                                                fontSize: 13,
                                                color: patient?.id === p.id ? CV.primaryText : CV.fg,
                                                fontWeight: patient?.id === p.id ? 700 : 500,
                                                background: patient?.id === p.id ? CV.primaryTint : 'transparent',
                                                fontFamily: 'inherit',
                                            }}
                                        >
                                            {p.name} ({p.age}세)
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* progress bar */}
                    <div className="mt-4">
                        <div
                            className="overflow-hidden"
                            style={{ height: 8, background: 'rgba(255,255,255,.2)', borderRadius: 9999 }}
                        >
                            <div
                                style={{
                                    width: `${pct}%`,
                                    height: '100%',
                                    background: CV.accent,
                                    borderRadius: 9999,
                                    transition: 'width 0.4s ease',
                                }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* segmented tabs */}
            <div className="px-4 mt-4">
                <div
                    className="inline-flex p-1"
                    style={{ background: '#fff', borderRadius: 9999, boxShadow: SHADOW.card }}
                >
                    {[{ id: 'today', label: '오늘' }, { id: 'all', label: '전체 스케줄' }].map((t) => (
                        <button
                            key={t.id}
                            onClick={() => setTab(t.id)}
                            className="cursor-pointer font-bold border-none"
                            style={{
                                padding: '8px 18px',
                                borderRadius: 9999,
                                background: tab === t.id ? CV.primary : 'transparent',
                                color: tab === t.id ? '#fff' : CV.fgMuted,
                                fontSize: 13,
                                fontFamily: 'inherit',
                                transition: 'background-color 0.15s ease',
                            }}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* timeline / list card */}
            <div className="px-4 mt-3.5">
                <div
                    style={{
                        background: '#fff',
                        borderRadius: 20,
                        padding: 16,
                        boxShadow: SHADOW.card,
                        border: '1px solid rgba(15,23,42,.04)',
                    }}
                >
                    <h3 className="m-0 mb-3 font-extrabold" style={{ fontSize: 15 }}>
                        {tab === 'today' ? '복약 타임라인' : '전체 스케줄'}
                    </h3>

                    {sorted.length === 0 && (
                        <div className="text-center py-6">
                            <p className="m-0 mb-3" style={{ color: CV.fgFaint, fontSize: 13 }}>등록된 복약 스케줄이 없습니다</p>
                            <button
                                onClick={() => setShowForm(true)}
                                className="cursor-pointer font-bold border-none"
                                style={{
                                    background: CV.primaryGrad,
                                    color: '#fff',
                                    borderRadius: 14,
                                    padding: '10px 18px',
                                    fontSize: 13,
                                    boxShadow: SHADOW.cta,
                                    fontFamily: 'inherit',
                                }}
                            >
                                + 첫 스케줄 추가
                            </button>
                        </div>
                    )}

                    {tab === 'today' && sorted.map((m, i, arr) => {
                        const s = getStatus(m);
                        const isTaken = s.kind === 'taken';
                        const tone = TONE[s.tone];
                        return (
                            <div
                                key={m.id}
                                className="flex items-start gap-3.5"
                                style={{
                                    padding: '14px 0',
                                    borderBottom: i === arr.length - 1 ? 'none' : `1px solid ${CV.divider}`,
                                }}
                            >
                                <div
                                    className="flex flex-col items-center justify-center shrink-0 font-bold"
                                    style={{
                                        width: 48, height: 48, borderRadius: 14,
                                        background: isTaken ? CV.successTint : CV.primaryTintSoft,
                                        color: isTaken ? CV.successText : CV.primary,
                                        lineHeight: 1.1,
                                    }}
                                >
                                    <span className="font-extrabold" style={{ fontSize: 14 }}>{m.scheduleTime.split(':')[0]}</span>
                                    <span style={{ fontSize: 9 }}>{m.scheduleTime.split(':')[1]}</span>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between gap-2">
                                        <p className="m-0 font-bold" style={{ fontSize: 14 }}>
                                            {mealLabel(m.scheduleTime)} {m.name}
                                        </p>
                                        <span
                                            className="font-bold"
                                            style={{ fontSize: 11, padding: '4px 10px', borderRadius: 9999, background: tone.bg, color: tone.fg }}
                                        >
                                            {s.label}
                                        </span>
                                    </div>
                                    <p className="m-0 mt-1" style={{ fontSize: 12, color: CV.fgMuted }}>{s.line}</p>
                                    {s.kind === 'pending' && (
                                        <button
                                            onClick={() => handleManualCheck(m.id)}
                                            className="bg-transparent border-none cursor-pointer p-0 mt-2 font-bold"
                                            style={{ color: CV.primary, fontSize: 12 }}
                                        >
                                            수동으로 완료 처리 ›
                                        </button>
                                    )}
                                </div>
                            </div>
                        );
                    })}

                    {tab === 'all' && sorted.map((m, i, arr) => (
                        <div
                            key={m.id}
                            className="flex items-center gap-3"
                            style={{
                                padding: '14px 0',
                                borderBottom: i === arr.length - 1 ? 'none' : `1px solid ${CV.divider}`,
                            }}
                        >
                            <div className="font-extrabold" style={{ width: 56, fontSize: 14, color: CV.primary }}>
                                {m.scheduleTime}
                            </div>
                            <div className="flex-1 min-w-0">
                                <p className="m-0 font-bold" style={{ fontSize: 14 }}>{m.name}</p>
                                <p className="m-0 mt-0.5" style={{ fontSize: 12, color: CV.fgMuted }}>
                                    {m.dosage} · {m.days?.split(',').map((d) => DAYS_KR[d]).join(' ')}
                                </p>
                            </div>
                            <button
                                onClick={() => handleDelete(m.id)}
                                className="bg-transparent border-none cursor-pointer font-semibold"
                                style={{ color: CV.dangerDeep, fontSize: 12 }}
                            >
                                삭제
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* AI 복약 감지 dark featured banner */}
            <div className="px-4 mt-3.5 mb-4">
                <div
                    className="flex items-center gap-3.5"
                    style={{
                        background: CV.inkGrad,
                        color: '#fff',
                        borderRadius: 20,
                        padding: 16,
                        boxShadow: SHADOW.card,
                    }}
                >
                    <div
                        className="flex items-center justify-center shrink-0"
                        style={{
                            width: 44, height: 44, borderRadius: 14,
                            background: CV.primaryGrad, boxShadow: SHADOW.fab,
                        }}
                    >
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                            <rect x="2" y="8" width="20" height="8" rx="4" stroke="#fff" strokeWidth="2"/>
                            <line x1="12" y1="8" x2="12" y2="16" stroke="#fff" strokeWidth="2"/>
                        </svg>
                    </div>
                    <div className="min-w-0">
                        <p className="m-0 font-extrabold" style={{ fontSize: 14 }}>AI 복약 감지 활성화됨</p>
                        <p className="m-0 mt-0.5" style={{ fontSize: 11, color: 'rgba(255,255,255,.7)' }}>
                            LIVE 모니터링 중 자동으로 감지됩니다
                        </p>
                    </div>
                </div>
            </div>

            {showForm && (
                <MedicationFormModal
                    onClose={() => setShowForm(false)}
                    onSubmit={handleAdd}
                />
            )}
        </div>
    );
}

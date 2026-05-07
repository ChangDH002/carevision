import { useState } from 'react';
import { api } from '../api/client';
import { CV, SHADOW } from '../styles/cv';

export default function LoginPage({ onLogin }) {
    const [mode, setMode] = useState('login');
    const [form, setForm] = useState({ email: '', password: '', name: '' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => setForm({ ...form, [e.target.name]: e.target.value });

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            if (mode === 'register') {
                await api.register(form);
                setMode('login');
                setError('회원가입 완료! 로그인해주세요.');
            } else {
                const data = await api.login({ email: form.email, password: form.password });
                localStorage.setItem('token', data.token);
                localStorage.setItem('user', JSON.stringify(data.user));
                onLogin(data.user);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const inputStyle = {
        padding: '15px 16px',
        border: `1.5px solid ${CV.border}`,
        borderRadius: 16,
        fontSize: 14,
        outline: 'none',
        width: '100%',
        background: CV.surfaceInput,
        boxSizing: 'border-box',
        fontFamily: 'inherit',
    };

    return (
        <div
            className="min-h-screen flex flex-col justify-end relative overflow-hidden max-w-[480px] mx-auto"
            style={{ background: CV.primaryGradHero }}
        >
            {/* decorative orbs */}
            <span className="absolute pointer-events-none" style={{ left: -80, top: 60, width: 240, height: 240, borderRadius: '50%', background: 'rgba(255,255,255,.08)' }} />
            <span className="absolute pointer-events-none" style={{ right: -60, top: 200, width: 180, height: 180, borderRadius: '50%', background: 'rgba(255,255,255,.06)' }} />

            {/* top brand */}
            <div className="flex-1 flex flex-col justify-center items-center text-white relative">
                <div
                    className="flex items-center justify-center mb-4"
                    style={{
                        width: 88, height: 88, borderRadius: 28,
                        background: 'rgba(255,255,255,.15)',
                        backdropFilter: 'blur(8px)',
                    }}
                >
                    <svg width="44" height="44" viewBox="0 0 24 24" fill="none">
                        <path d="M3 10.5L12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1v-9.5z"
                              stroke="#fff" strokeWidth="2.4" strokeLinejoin="round" fill="rgba(255,255,255,.2)"/>
                    </svg>
                </div>
                <h1 className="font-extrabold m-0" style={{ fontSize: 32, letterSpacing: '-0.02em' }}>CareVision</h1>
                <p className="m-0 opacity-90 mt-1" style={{ fontSize: 13 }}>가족의 안전을 지키는 가장 가까운 눈</p>
            </div>

            {/* sheet */}
            <div
                className="relative z-10"
                style={{
                    background: '#fff',
                    borderRadius: '32px 32px 0 0',
                    padding: '28px 24px 36px',
                }}
            >
                {/* segmented tabs */}
                <div
                    className="flex p-1 mb-4"
                    style={{ borderRadius: 9999, background: CV.divider }}
                >
                    {['login', 'register'].map((m) => (
                        <button
                            type="button"
                            key={m}
                            onClick={() => { setMode(m); setError(''); }}
                            className="flex-1 cursor-pointer font-bold border-none"
                            style={{
                                padding: '10px 0', borderRadius: 9999,
                                background: mode === m ? CV.primary : 'transparent',
                                color: mode === m ? '#fff' : CV.fgMuted,
                                fontSize: 14, fontFamily: 'inherit',
                                transition: 'background-color 0.15s ease',
                            }}
                        >
                            {m === 'login' ? '로그인' : '회원가입'}
                        </button>
                    ))}
                </div>

                <form onSubmit={handleSubmit} className="flex flex-col gap-2.5">
                    {mode === 'register' && (
                        <input
                            style={inputStyle}
                            name="name"
                            placeholder="이름"
                            value={form.name}
                            onChange={handleChange}
                            autoComplete="name"
                            required
                        />
                    )}
                    <input
                        style={inputStyle}
                        name="email"
                        type="email"
                        placeholder="이메일"
                        value={form.email}
                        onChange={handleChange}
                        autoComplete="email"
                        inputMode="email"
                        required
                    />
                    <input
                        style={inputStyle}
                        name="password"
                        type="password"
                        placeholder="비밀번호"
                        value={form.password}
                        onChange={handleChange}
                        autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                        required
                    />

                    {error && (
                        <p
                            className="text-center m-0 mt-1"
                            style={{
                                fontSize: 12,
                                color: error.includes('완료') ? CV.successText : CV.dangerDeep,
                            }}
                        >
                            {error}
                        </p>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full cursor-pointer font-bold border-none mt-2"
                        style={{
                            padding: '15px 16px',
                            background: loading ? '#94A3B8' : CV.primaryGrad,
                            color: '#fff',
                            borderRadius: 16,
                            fontSize: 15,
                            fontFamily: 'inherit',
                            boxShadow: loading ? 'none' : SHADOW.cta,
                            opacity: loading ? 0.6 : 1,
                        }}
                    >
                        {loading ? '처리 중...' : mode === 'login' ? '로그인' : '회원가입'}
                    </button>
                </form>
            </div>
        </div>
    );
}

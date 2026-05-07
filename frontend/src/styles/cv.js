// CareVision v2 — shared color / shadow constants for inline-style usage.
// CSS variables (in index.css) are the canonical source; these mirror them
// for places where inline styles are easier (gradients, dynamic tints, etc.).

export const CV = {
    primary:        '#2F6BFF',
    primaryPress:   '#1E4FE0',
    primaryTint:    '#DBEAFE',
    primaryTintSoft:'#EFF4FF',
    primaryText:    '#1E40AF',
    primaryGrad:    'linear-gradient(180deg,#3B7CFF 0%,#1E4FE0 100%)',
    primaryGradHero:'linear-gradient(160deg,#3B7CFF 0%,#1E4FE0 100%)',

    ink:            '#0F172A',
    inkSoft:        '#1E293B',
    inkGrad:        'linear-gradient(135deg,#0F172A 0%,#1E293B 100%)',

    accent:         '#FFD24A',

    danger:         '#EF4444',
    dangerDeep:     '#DC2626',
    dangerTint:     '#FEE2E2',
    dangerGrad:     'linear-gradient(180deg,#F87171 0%,#DC2626 100%)',

    success:        '#22C55E',
    successTint:    '#DCFCE7',
    successText:    '#15803D',

    warn:           '#F59E0B',
    warnTint:       '#FEF3C7',
    warnText:       '#B45309',

    bg:             '#F4F5F9',
    surface:        '#FFFFFF',
    surfaceInput:   '#F8FAFC',
    cameraBg:       '#0F172A',

    fg:             '#0F172A',
    fgMuted:        '#64748B',
    fgFaint:        '#94A3B8',

    divider:        '#EEF2F7',
    border:         '#E2E8F0',
};

export const SHADOW = {
    card:   '0 12px 28px rgba(30,79,224,.10), 0 2px 6px rgba(15,23,42,.04)',
    fab:    '0 8px 20px rgba(47,107,255,.40)',
    navUp:  '0 -8px 24px rgba(15,23,42,.06)',
    cta:    '0 8px 16px rgba(47,107,255,.25)',
};

export const meal = (t) => {
    const h = parseInt(t?.split(':')[0] || '0', 10);
    if (h < 11) return '아침';
    if (h < 16) return '점심';
    if (h < 19) return '저녁';
    return '밤';
};

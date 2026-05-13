"""
V2.1-B : 5-class coarse 버전 dataset builder.

원본 세부 라벨 → 새 coarse 라벨 매핑:
    Fall(0)                ← {0,1,2,3,4}    HEURISTIC
    Walking(1)             ← {5}            GROUND_TRUTH
    Lying_Down(2)          ← {6}            GROUND_TRUTH
    Sitting(3)             ← {7}            GROUND_TRUTH
    Standing_Transition(4) ← {10,11,12}     GROUND_TRUTH

특징:
 - keypoints 추출 결과 (keypoints/fall/*, keypoints/normal_video/{train,test}/*) 재사용
 - 추출 스크립트 미수정, build 단계에서만 remap 수행
 - total=0 인 클래스는 자동 제외
 - labels.json 에 새 클래스 정의 + 원본 → 신규 매핑 기록
 - 저장 경로: dataset/train_vB.npz, val_vB.npz, test_vB.npz, labels_vB.json
"""

from __future__ import annotations

import os
import sys
import glob
import json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import (
    FALL_KP_DIR, NORMAL_VIDEO_KP_DIR, DATASET_DIR,
    SEQUENCE_LEN, STRIDE, NUM_FEATURES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)

LEFT_HIP_IDX  = 23
RIGHT_HIP_IDX = 24

# ----- vB coarse class definitions -----
VB_LABELS = {
    0: "Fall",
    1: "Walking",
    2: "Lying_Down",
    3: "Sitting",
    4: "Standing_Transition",
}

VB_LABEL_SOURCE = {
    0: "HEURISTIC",       # LE2I fall subtype 전부가 heuristic 이므로
    1: "GROUND_TRUTH",
    2: "GROUND_TRUTH",
    3: "GROUND_TRUTH",
    4: "GROUND_TRUTH",
}

# 원본 id → vB id
ORIGINAL_TO_VB = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0,    # Fall
    5: 1,                             # Walking
    6: 2,                             # Lying_Down
    7: 3,                             # Sitting
    10: 4, 11: 4, 12: 4,              # Standing_Transition
}

# vB id → 원본 ids (역매핑, 로그용)
VB_TO_ORIGINAL = {}
for orig, vb in ORIGINAL_TO_VB.items():
    VB_TO_ORIGINAL.setdefault(vb, []).append(orig)


def _load_one(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"]
    y = int(d["y"]) if np.ndim(d["y"]) == 0 else int(d["y"].item())
    try:
        is_heur = bool(d["is_heuristic"])
    except Exception:
        is_heur = True
    return X, y, is_heur


def _load_dir(pattern):
    items = []
    for p in sorted(glob.glob(pattern)):
        try:
            X, y, h = _load_one(p)
            items.append((X, y, h, os.path.basename(p)))
        except Exception as e:
            print(f"[SKIP] {p}: {e}")
    return items


def normalize_sequence(seq):
    T = seq.shape[0]
    pts = seq.astype(np.float32).reshape(T, 33, 4).copy()
    hip = (pts[:, LEFT_HIP_IDX, :3] + pts[:, RIGHT_HIP_IDX, :3]) / 2.0
    pts[:, :, :3] -= hip[:, None, :]
    vis = pts[:, :, 3].reshape(-1)
    xy  = pts[:, :, :2].reshape(-1, 2)
    mask = vis > 0.1
    if mask.sum() > 10:
        xyv = xy[mask]
        scale = max(
            float(np.percentile(np.abs(xyv[:, 0]), 95)),
            float(np.percentile(np.abs(xyv[:, 1]), 95)),
            1e-3,
        )
    else:
        scale = 1.0
    pts[:, :, :3] /= scale
    return pts.reshape(T, NUM_FEATURES)


def make_windows(seq, label, heur, seq_len, stride):
    T = seq.shape[0]
    out = []
    if T < seq_len:
        pad = np.tile(seq[-1:], (seq_len - T, 1))
        out.append((np.concatenate([seq, pad], axis=0).astype(np.float32),
                    label, heur))
        return out
    for s in range(0, T - seq_len + 1, stride):
        out.append((seq[s:s + seq_len].astype(np.float32), label, heur))
    if (T - seq_len) % stride != 0:
        out.append((seq[-seq_len:].astype(np.float32), label, heur))
    return out


def stratified_split(indices, labels, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in sorted(set(labels.tolist())):
        idx = indices[labels == c]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(n * TRAIN_RATIO))
        nva = int(round(n * VAL_RATIO))
        tr.extend(idx[:ntr].tolist())
        va.extend(idx[ntr:ntr + nva].tolist())
        te.extend(idx[ntr + nva:].tolist())
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return (np.array(tr, dtype=np.int64),
            np.array(va, dtype=np.int64),
            np.array(te, dtype=np.int64))


def half_split(indices, labels, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 1)
    va, te = [], []
    for c in sorted(set(labels.tolist())):
        idx = indices[labels == c]
        rng.shuffle(idx)
        half = len(idx) // 2
        va.extend(idx[:half].tolist())
        te.extend(idx[half:].tolist())
    rng.shuffle(va); rng.shuffle(te)
    return np.array(va, dtype=np.int64), np.array(te, dtype=np.int64)


def process(items):
    """return Xs, ys(vB), hs, orig_seq_cnt, vb_win_cnt."""
    Xs, ys, hs = [], [], []
    orig_seq_cnt = Counter()
    vb_win_cnt   = Counter()
    for X, y_orig, heur, fname in items:
        if X.ndim != 2 or X.shape[1] != NUM_FEATURES:
            print(f"[SKIP] bad shape {X.shape}: {fname}")
            continue
        if y_orig not in ORIGINAL_TO_VB:
            continue
        y_vb = ORIGINAL_TO_VB[y_orig]
        orig_seq_cnt[y_orig] += 1
        norm = normalize_sequence(X)
        for w, lb, h in make_windows(norm, y_vb, heur, SEQUENCE_LEN, STRIDE):
            Xs.append(w); ys.append(lb); hs.append(1 if h else 0)
            vb_win_cnt[lb] += 1
    return Xs, ys, hs, orig_seq_cnt, vb_win_cnt


def main():
    fall_items = _load_dir(os.path.join(FALL_KP_DIR, "*.npz"))
    nv_train   = _load_dir(os.path.join(NORMAL_VIDEO_KP_DIR, "train", "*.npz"))
    nv_test    = _load_dir(os.path.join(NORMAL_VIDEO_KP_DIR, "test",  "*.npz"))
    print(f"[INFO] loaded fall={len(fall_items)}  "
          f"normal_video/train={len(nv_train)}  "
          f"normal_video/test={len(nv_test)}")

    Xf, yf, hf, f_seq_cnt, f_win_cnt = process(fall_items)
    Xn_tr, yn_tr, hn_tr, nv_tr_seq, nv_tr_win = process(nv_train)
    Xn_te, yn_te, hn_te, nv_te_seq, nv_te_win = process(nv_test)

    # stack
    def _stack(Xs, ys, hs):
        if not Xs:
            return (np.zeros((0, SEQUENCE_LEN, NUM_FEATURES), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int8))
        return (np.stack(Xs, axis=0),
                np.array(ys, dtype=np.int64),
                np.array(hs, dtype=np.int8))

    Xf_a, yf_a, hf_a = _stack(Xf, yf, hf)
    Xn_tr_a, yn_tr_a, hn_tr_a = _stack(Xn_tr, yn_tr, hn_tr)
    Xn_te_a, yn_te_a, hn_te_a = _stack(Xn_te, yn_te, hn_te)

    # fall stratified 70/15/15
    if len(Xf_a):
        idx = np.arange(len(Xf_a), dtype=np.int64)
        tr_f, va_f, te_f = stratified_split(idx, yf_a)
    else:
        tr_f = va_f = te_f = np.zeros((0,), dtype=np.int64)

    # normal test → half val/test
    if len(Xn_te_a):
        idx = np.arange(len(Xn_te_a), dtype=np.int64)
        va_n, te_n = half_split(idx, yn_te_a)
    else:
        va_n = te_n = np.zeros((0,), dtype=np.int64)

    X_train = np.concatenate([Xf_a[tr_f], Xn_tr_a], axis=0)
    y_train = np.concatenate([yf_a[tr_f], yn_tr_a], axis=0)
    h_train = np.concatenate([hf_a[tr_f], hn_tr_a], axis=0)

    X_val = np.concatenate([Xf_a[va_f], Xn_te_a[va_n]], axis=0)
    y_val = np.concatenate([yf_a[va_f], yn_te_a[va_n]], axis=0)
    h_val = np.concatenate([hf_a[va_f], hn_te_a[va_n]], axis=0)

    X_test = np.concatenate([Xf_a[te_f], Xn_te_a[te_n]], axis=0)
    y_test = np.concatenate([yf_a[te_f], yn_te_a[te_n]], axis=0)
    h_test = np.concatenate([hf_a[te_f], hn_te_a[te_n]], axis=0)

    if not (len(X_train) or len(X_val) or len(X_test)):
        print("[ERROR] 생성된 윈도우 없음. 01/02_video 추출 먼저 실행.")
        return

    rng = np.random.default_rng(RANDOM_SEED + 2)
    def _shuf(X, y, h):
        p = rng.permutation(len(X))
        return X[p], y[p], h[p]
    X_train, y_train, h_train = _shuf(X_train, y_train, h_train)
    X_val,   y_val,   h_val   = _shuf(X_val,   y_val,   h_val)
    X_test,  y_test,  h_test  = _shuf(X_test,  y_test,  h_test)

    # total=0 클래스 자동 제외
    per_class_total = {}
    for c in sorted(VB_LABELS):
        t = int((y_train == c).sum() + (y_val == c).sum() + (y_test == c).sum())
        per_class_total[c] = t
    active_vb = [c for c, t in per_class_total.items() if t > 0]
    dropped   = [c for c, t in per_class_total.items() if t == 0]
    if dropped:
        print(f"[INFO] drop empty vB classes: {dropped}")
        mask_tr = np.isin(y_train, active_vb)
        mask_va = np.isin(y_val,   active_vb)
        mask_te = np.isin(y_test,  active_vb)
        X_train, y_train, h_train = X_train[mask_tr], y_train[mask_tr], h_train[mask_tr]
        X_val,   y_val,   h_val   = X_val[mask_va],   y_val[mask_va],   h_val[mask_va]
        X_test,  y_test,  h_test  = X_test[mask_te],  y_test[mask_te],  h_test[mask_te]

    print(f"\n[SHAPE] train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    print("\n[DIST] per-vB-class window count")
    for c in active_vb:
        tr_c = int((y_train == c).sum())
        va_c = int((y_val   == c).sum())
        te_c = int((y_test  == c).sum())
        total = tr_c + va_c + te_c
        print(f"   {c} {VB_LABELS[c]:<20} "
              f"train={tr_c:>5}  val={va_c:>5}  test={te_c:>5}  "
              f"total={total:>5}  "
              f"src={VB_LABEL_SOURCE[c]}  "
              f"orig={VB_TO_ORIGINAL[c]}")

    np.savez_compressed(os.path.join(DATASET_DIR, "train_vB.npz"),
                        X=X_train, y=y_train, heuristic=h_train)
    np.savez_compressed(os.path.join(DATASET_DIR, "val_vB.npz"),
                        X=X_val, y=y_val, heuristic=h_val)
    np.savez_compressed(os.path.join(DATASET_DIR, "test_vB.npz"),
                        X=X_test, y=y_test, heuristic=h_test)

    meta = {
        "version": "V2.1-B-5class-coarse",
        "vB_labels": VB_LABELS,
        "vB_label_source": VB_LABEL_SOURCE,
        "original_to_vB": ORIGINAL_TO_VB,
        "vB_to_original": VB_TO_ORIGINAL,
        "active_vB_classes": active_vb,
        "dropped_empty_classes": dropped,
        "sequence_len": SEQUENCE_LEN,
        "stride": STRIDE,
        "num_features": NUM_FEATURES,
        "split_policy": {
            "fall_LE2I":    "stratified 70/15/15 → all remap to Fall(0)",
            "normal_video": "train → train; test → stratified half(val/test)",
        },
        "counts": {
            "fall_original_sequences": dict(f_seq_cnt),
            "fall_vB_windows":          dict(f_win_cnt),
            "normal_train_vB_windows":  dict(nv_tr_win),
            "normal_test_vB_windows":   dict(nv_te_win),
            "final_train_windows":      int(len(y_train)),
            "final_val_windows":        int(len(y_val)),
            "final_test_windows":       int(len(y_test)),
            "per_class_total":          per_class_total,
        },
        "notes": (
            "Fall(0) 은 LE2I 세부유형 5개(0~4) 를 하나로 합친 HEURISTIC 라벨. "
            "나머지 1~4 는 dataset_action_split 의 GROUND_TRUTH 라벨. "
            "세부유형 오분류 위험이 큰 상황에서 안정 성능을 우선하는 coarse 버전."),
    }
    with open(os.path.join(DATASET_DIR, "labels_vB.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("\n[DONE] vB dataset saved under:", DATASET_DIR)


if __name__ == "__main__":
    main()

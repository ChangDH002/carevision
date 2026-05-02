"""
05_train_lstm_vB_aug.py — 데이터 증강 추가한 vB 재학습 스크립트.

기존 04_train_lstm_vB.py 와 동일 모델 구조이지만,
학습 시 다음 augmentation 을 매 epoch 랜덤 적용:
  1) 좌우 flip (50%)
  2) ±15도 회전 (xy 평면)
  3) 시간 jitter (30%) — 프레임 인덱스 랜덤 재샘플
  4) keypoint 가우시안 노이즈
  5) 일부 관절 랜덤 가림 (30%) — occlusion 시뮬

기존 데이터(*_vB.npz)는 이미 hip-centering + 95% scale 정규화된 상태이므로
augmentation 도 정규화 후 좌표 기준으로 동작 (좌우 flip은 x → -x).

저장: models/fall_lstm_vB_aug_best.keras
실행:
    cd fall_lstm_project
    python Fall_Down_Detail/05_train_lstm_vB_aug.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from config import (
    DATASET_DIR, MODEL_DIR, LOG_DIR,
    SEQUENCE_LEN, NUM_FEATURES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    LSTM_UNITS, DROPOUT, PATIENCE, RANDOM_SEED,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ---- 좌/우 관절 swap (MediaPipe Pose) ----
_PAIRS = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
          (17,20),(18,21),(19,22),(23,24),(25,26),(27,28),(29,30),(31,32)]
LR_SWAP_IDX = list(range(33))
for a, b in _PAIRS:
    LR_SWAP_IDX[a], LR_SWAP_IDX[b] = b, a
LR_SWAP_IDX = np.array(LR_SWAP_IDX, dtype=np.int64)

VB_LABELS = {0:"Fall", 1:"Walking", 2:"Lying_Down", 3:"Sitting", 4:"Standing_Transition"}


def load_split(name):
    p = os.path.join(DATASET_DIR, f"{name}_vB.npz")
    d = np.load(p, allow_pickle=False)
    return d["X"], d["y"], d["heuristic"]


def build_dense_mapping(y_train):
    ids = sorted(set(y_train.tolist()))
    return {c: i for i, c in enumerate(ids)}, {i: c for i, c in enumerate(ids)}


def remap(y, id2dense):
    return np.array([id2dense[int(v)] for v in y], dtype=np.int64)


# ---- Augmentation ----
def augment_window(x: np.ndarray) -> np.ndarray:
    """x: (SEQUENCE_LEN, 132) 정규화된 시퀀스. 5가지 augmentation 랜덤 적용."""
    pts = x.reshape(SEQUENCE_LEN, 33, 4).copy()

    # 1) 좌우 flip
    if np.random.rand() < 0.5:
        pts = pts[:, LR_SWAP_IDX, :]
        pts[..., 0] = -pts[..., 0]   # 정규화된 좌표는 hip 기준 → x → -x

    # 2) ±15도 회전 (xy 평면)
    theta = np.deg2rad(np.random.uniform(-15, 15))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = pts[..., :2].reshape(-1, 2) @ R.T
    pts[..., :2] = xy.reshape(SEQUENCE_LEN, 33, 2)

    # 3) 시간 jitter
    if np.random.rand() < 0.3:
        idx = np.sort(np.random.choice(SEQUENCE_LEN, SEQUENCE_LEN, replace=True))
        pts = pts[idx]

    # 4) keypoint 가우시안 노이즈
    noise_scale = np.random.uniform(0.005, 0.025)
    pts[..., :3] += np.random.normal(0, noise_scale, pts[..., :3].shape).astype(np.float32)

    # 5) 일부 관절 가림 — Masking 레이어가 mask_value=0.0 으로 무시
    if np.random.rand() < 0.3:
        n_mask = np.random.randint(1, 5)
        mask_idx = np.random.choice(33, n_mask, replace=False)
        pts[:, mask_idx, :] = 0.0

    return pts.reshape(SEQUENCE_LEN, NUM_FEATURES).astype(np.float32)


def make_aug_dataset(X, y, batch_size):
    """매 epoch 마다 랜덤 augmentation 이 적용되는 무한 generator."""
    def gen():
        idxs = np.arange(len(X))
        while True:
            np.random.shuffle(idxs)
            for i in idxs:
                yield augment_window(X[i]), np.int64(y[i])
    sig = (
        tf.TensorSpec(shape=(SEQUENCE_LEN, NUM_FEATURES), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    inputs = layers.Input(shape=(SEQUENCE_LEN, NUM_FEATURES), name="pose_seq")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS[0], return_sequences=True))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS[1]))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(num_classes, activation="softmax", name="logits")(x)
    model = models.Model(inputs, out, name="fall_bilstm_vB_aug")
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weight(y_dense):
    cnt = Counter(y_dense.tolist())
    n = sum(cnt.values()); k = len(cnt)
    return {c: n / (k * v) for c, v in cnt.items()}


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    Xte, yte, _ = load_split("test")
    print(f"[INFO] train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    id2d, d2id = build_dense_mapping(ytr)
    K = len(id2d)
    ytr_d = remap(ytr, id2d); yva_d = remap(yva, id2d); yte_d = remap(yte, id2d)
    cw = compute_class_weight(ytr_d)
    print(f"[INFO] classes={K}, class_weight=", {d2id[k]: round(v, 3) for k, v in cw.items()})

    train_ds = make_aug_dataset(Xtr, ytr_d, BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((Xva, yva_d)).batch(BATCH_SIZE)

    model = build_model(K)
    model.summary(print_fn=lambda s: print("[MODEL]", s))

    best_path = os.path.join(MODEL_DIR, "fall_lstm_vB_aug_best.keras")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]

    steps_per_epoch = max(1, len(Xtr) // BATCH_SIZE)
    print(f"[INFO] steps_per_epoch={steps_per_epoch}, epochs={EPOCHS}")
    hist = model.fit(
        train_ds, validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS, class_weight=cw, callbacks=cbs, verbose=2,
    )

    with open(os.path.join(LOG_DIR, "training_history_vB_aug.json"),
              "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()},
                  f, ensure_ascii=False, indent=2)

    # 테스트 평가
    te_loss, te_acc = model.evaluate(Xte, yte_d, verbose=0)
    print(f"\n[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")

    pred_d = np.argmax(model.predict(Xte, batch_size=BATCH_SIZE, verbose=0), axis=1)
    pred_vb = np.array([d2id[int(v)] for v in pred_d], dtype=np.int64)
    fall_mask = (yte == 0)
    fall_acc = float((pred_vb[fall_mask] == 0).mean()) if fall_mask.any() else float("nan")
    print(f"[TEST] Fall recall = {fall_acc:.4f} (n={int(fall_mask.sum())})")

    print(f"\n[DONE] saved → {best_path}")


if __name__ == "__main__":
    main()

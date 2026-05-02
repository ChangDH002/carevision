"""
V2.1-B : 5-class coarse LSTM 학습 & 평가.

 - dataset/{train,val,test}_vB.npz 로드
 - dense 재매핑 (비어있는 vB id 자동 제외 대응)
 - BiLSTM(128→64) + Dropout → softmax
 - class_weight 로 imbalance 보정
 - best/final 모델 저장, history / test_report_vB / classification_report_vB / confusion_matrix_vB 저장
 - HEURISTIC(vB=0) vs GROUND_TRUTH(vB≥1) 그룹별 accuracy 분리 보고
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import (
    DATASET_DIR, MODEL_DIR, LOG_DIR,
    SEQUENCE_LEN, NUM_FEATURES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    LSTM_UNITS, DROPOUT, PATIENCE, RANDOM_SEED,
)

# vB labels (03_build_dataset_vB.py 와 동일)
VB_LABELS = {
    0: "Fall",
    1: "Walking",
    2: "Lying_Down",
    3: "Sitting",
    4: "Standing_Transition",
}
VB_LABEL_SOURCE = {0: "HEURISTIC", 1: "GROUND_TRUTH",
                   2: "GROUND_TRUTH", 3: "GROUND_TRUTH",
                   4: "GROUND_TRUTH"}

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
except ImportError:
    print("[ERROR] tensorflow 설치 필요: pip install tensorflow")
    raise

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SK = True
except ImportError:
    HAS_SK = False


def load_split(name):
    p = os.path.join(DATASET_DIR, f"{name}_vB.npz")
    d = np.load(p, allow_pickle=False)
    return d["X"], d["y"], d["heuristic"]


def build_dense_mapping(y_train):
    ids = sorted(set(y_train.tolist()))
    id2dense = {c: i for i, c in enumerate(ids)}
    dense2id = {i: c for c, i in id2dense.items()}
    return id2dense, dense2id


def remap(y, id2dense):
    return np.array([id2dense[int(v)] for v in y], dtype=np.int64)


def build_model(num_classes):
    inputs = layers.Input(shape=(SEQUENCE_LEN, NUM_FEATURES), name="pose_seq")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS[0], return_sequences=True))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS[1]))(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(num_classes, activation="softmax", name="logits")(x)
    model = models.Model(inputs, out, name="fall_bilstm_vB")
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


def compute_class_weight(y_dense):
    cnt = Counter(y_dense.tolist())
    n = sum(cnt.values())
    k = len(cnt)
    return {c: n / (k * v) for c, v in cnt.items()}


def group_accuracy(y_true_vb, y_pred_vb, class_ids):
    mask = np.isin(y_true_vb, list(class_ids))
    if mask.sum() == 0:
        return None, 0
    acc = float((y_true_vb[mask] == y_pred_vb[mask]).mean())
    return acc, int(mask.sum())


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    Xtr, ytr, htr = load_split("train")
    Xva, yva, hva = load_split("val")
    Xte, yte, hte = load_split("test")

    print(f"[INFO] train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    id2dense, dense2id = build_dense_mapping(ytr)
    num_classes = len(id2dense)
    print(f"[INFO] vB classes in train: {sorted(id2dense)}  -> dense 0..{num_classes - 1}")

    ytr_d = remap(ytr, id2dense)
    yva_d = remap(yva, id2dense)
    yte_d = remap(yte, id2dense)

    cw = compute_class_weight(ytr_d)
    print("[INFO] class_weight:", {dense2id[k]: round(v, 3) for k, v in cw.items()})

    model = build_model(num_classes)
    model.summary(print_fn=lambda s: print("[MODEL]", s))

    best_path  = os.path.join(MODEL_DIR, "fall_lstm_vB_best.keras")
    final_path = os.path.join(MODEL_DIR, "fall_lstm_vB_final.keras")

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True),
        callbacks.ModelCheckpoint(best_path, monitor="val_loss",
                                  save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=4, min_lr=1e-5),
    ]

    hist = model.fit(
        Xtr, ytr_d,
        validation_data=(Xva, yva_d),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=cw, callbacks=cbs, verbose=2)

    model.save(final_path)
    with open(os.path.join(LOG_DIR, "training_history_vB.json"),
              "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in hist.history.items()},
                  f, ensure_ascii=False, indent=2)

    # ========== TEST ==========
    te_loss, te_acc = model.evaluate(Xte, yte_d, verbose=0)
    print(f"\n[TEST] loss={te_loss:.4f}  acc={te_acc:.4f}")

    pred_d  = np.argmax(model.predict(Xte, batch_size=BATCH_SIZE, verbose=0),
                        axis=1)
    pred_vb = np.array([dense2id[int(v)] for v in pred_d], dtype=np.int64)

    heur_acc, heur_n = group_accuracy(yte, pred_vb, [0])           # Fall only
    gt_acc,   gt_n   = group_accuracy(yte, pred_vb, [1, 2, 3, 4])  # GT classes

    print(f"[TEST] Fall (HEURISTIC) acc = "
          f"{'--' if heur_acc is None else f'{heur_acc:.4f}'}  (n={heur_n})")
    print(f"[TEST] GROUND_TRUTH acc     = "
          f"{'--' if gt_acc   is None else f'{gt_acc:.4f}'}  (n={gt_n})")

    report = {
        "version": "V2.1-B-5class-coarse",
        "num_classes": num_classes,
        "vB_labels": VB_LABELS,
        "vB_label_source": VB_LABEL_SOURCE,
        "dense_to_vB_id": {int(k): int(v) for k, v in dense2id.items()},
        "test_loss": float(te_loss),
        "test_accuracy": float(te_acc),
        "fall_accuracy": heur_acc, "fall_n": heur_n,
        "ground_truth_accuracy": gt_acc, "ground_truth_n": gt_n,
        "n_test": int(len(yte)),
    }

    if HAS_SK:
        labels_sorted = sorted(set(yte.tolist()) | set(pred_vb.tolist()))
        cm = confusion_matrix(yte, pred_vb, labels=labels_sorted)
        import csv
        cm_path = os.path.join(LOG_DIR, "confusion_matrix_vB.csv")
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + [VB_LABELS.get(c, str(c)) for c in labels_sorted])
            for i, r in enumerate(cm):
                w.writerow([VB_LABELS.get(labels_sorted[i], str(labels_sorted[i]))]
                           + r.tolist())

        rep = classification_report(
            yte, pred_vb, labels=labels_sorted,
            target_names=[VB_LABELS.get(c, str(c)) for c in labels_sorted],
            output_dict=True, zero_division=0)
        cr_path = os.path.join(LOG_DIR, "classification_report_vB.csv")
        with open(cr_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1-score", "support"])
            for k, v in rep.items():
                if isinstance(v, dict):
                    w.writerow([k,
                                round(v.get("precision", 0), 4),
                                round(v.get("recall", 0), 4),
                                round(v.get("f1-score", 0), 4),
                                int(v.get("support", 0))])
        report["classification_report"] = rep
        report["confusion_matrix_labels"] = labels_sorted

    with open(os.path.join(LOG_DIR, "test_report_vB.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("\n[DONE] model saved:", best_path, "/", final_path)
    print("[DONE] logs saved under:", LOG_DIR)


if __name__ == "__main__":
    main()

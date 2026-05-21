"""
1안 손→입 행동 감지 실영상/웹캠 검증 하니스.

이 스크립트는 hand_to_mouth_detector.detect() (= /detect/live 가 읽는
바로 그 경로)를 영상/웹캠 프레임마다 실행해 다음을 집계한다.

  - final_detected (any / ratio)
  - 대표 detection_source
  - mean lstm_confidence
  - rule_detected ratio
  - mean rule_distance (값 있는 프레임만)
  - motion_accepted ratio
  - taken 관련 키가 생성되지 않음(불변식) 검증

사용:
  # 단일 영상
  ai\\venv\\Scripts\\python.exe validate_hand_to_mouth.py path\\to\\eat_apple.mp4 --label "사과 먹기"

  # 웹캠 (index)
  ai\\venv\\Scripts\\python.exe validate_hand_to_mouth.py 0 --label "웹캠 손→입" --max-frames 150

  # 폴더 일괄 (mp4/webm/mov/avi)
  ai\\venv\\Scripts\\python.exe validate_hand_to_mouth.py path\\to\\videos_dir --batch

출력: 마지막에 마크다운 표 1행(또는 N행)을 그대로 붙여넣을 수 있게 출력.
"""

from __future__ import annotations

import argparse
import base64
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipelines.hand_to_mouth_detector import hand_to_mouth_detector  # noqa: E402

# taken/복약완료 개념이 손→입 결과에 절대 섞이면 안 됨 (1안 불변식)
FORBIDDEN_KEYS = {"taken", "score", "medication_score", "medication_objects"}


def _b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def validate_source(source, label: str, max_frames: int, cam_id: str) -> dict:
    is_idx = isinstance(source, int)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if is_idx else cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return {"label": label, "error": f"소스를 열 수 없음: {source}"}

    hand_to_mouth_detector.reset(cam_id)

    n = 0
    final_true = 0
    rule_true = 0
    motion_true = 0
    lstm_confs: list[float] = []
    rule_dists: list[float] = []
    sources: Counter = Counter()
    statuses: Counter = Counter()
    forbidden_hit = set()

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        n += 1
        r = hand_to_mouth_detector.detect(_b64(frame), camera_id=cam_id)

        forbidden_hit |= (set(r.keys()) & FORBIDDEN_KEYS)
        statuses[r.get("status", "?")] += 1
        sources[r.get("detection_source", "none")] += 1
        if r.get("final_detected"):
            final_true += 1
        if r.get("rule_detected"):
            rule_true += 1
        if r.get("motion_accepted"):
            motion_true += 1
        lc = r.get("lstm_confidence")
        if isinstance(lc, (int, float)):
            lstm_confs.append(float(lc))
        rd = r.get("rule_distance")
        if isinstance(rd, (int, float)):
            rule_dists.append(float(rd))

    cap.release()
    if n == 0:
        return {"label": label, "error": "프레임 0개"}

    return {
        "label": label,
        "frames": n,
        "final_detected_any": final_true > 0,
        "final_detected_ratio": round(final_true / n, 3),
        "detection_source_top": sources.most_common(1)[0][0],
        "detection_source_dist": dict(sources),
        "lstm_confidence_mean": round(statistics.mean(lstm_confs), 4) if lstm_confs else None,
        "rule_detected_ratio": round(rule_true / n, 3),
        "rule_distance_mean": round(statistics.mean(rule_dists), 4) if rule_dists else None,
        "motion_accepted_ratio": round(motion_true / n, 3),
        "status_dist": dict(statuses),
        "forbidden_keys_present": sorted(forbidden_hit),  # 비어 있어야 정상
    }


def _row(res: dict) -> str:
    if res.get("error"):
        return f"| {res['label']} | ERROR: {res['error']} |"
    verdict = "정상" if not res["forbidden_keys_present"] else "불변식 위반!"
    return (
        f"| {res['label']} "
        f"| {res['final_detected_any']} (ratio {res['final_detected_ratio']}) "
        f"| {res['detection_source_top']} "
        f"| {res['lstm_confidence_mean']} "
        f"| ratio {res['rule_detected_ratio']} "
        f"| {res['rule_distance_mean']} "
        f"| ratio {res['motion_accepted_ratio']} "
        f"| {verdict} "
        f"| {'없음' if not res['forbidden_keys_present'] else res['forbidden_keys_present']} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="영상 경로 / 웹캠 index / 폴더(--batch)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--batch", action="store_true", help="폴더 내 영상 일괄")
    ap.add_argument("--max-frames", type=int, default=0, help="0=전체")
    args = ap.parse_args()

    if hand_to_mouth_detector is None:
        print("ERROR: hand_to_mouth_detector 미로드 (모델 파일 확인)")
        return

    header = (
        "| 영상/동작 | final_detected | detection_source | lstm_confidence "
        "| rule_detected | rule_distance | motion_accepted | 판단 | 문제 여부 |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    print(header)

    if args.batch:
        d = Path(args.source)
        vids = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in (".mp4", ".webm", ".mov", ".avi")
        )
        for i, v in enumerate(vids):
            res = validate_source(str(v), args.label or v.name, args.max_frames, f"val{i}")
            print(_row(res))
    else:
        src = int(args.source) if args.source.isdigit() else args.source
        label = args.label or (f"webcam-{src}" if isinstance(src, int) else Path(src).name)
        res = validate_source(src, label, args.max_frames, "val0")
        print(_row(res))
        print("\n[debug]", {k: res.get(k) for k in (
            "frames", "detection_source_dist", "status_dist", "forbidden_keys_present"
        )})


if __name__ == "__main__":
    main()

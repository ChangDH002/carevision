"""
1안 손→입 프레임별 debug 로거.

영상 전체를 hand_to_mouth_detector.detect() 로 돌리며 프레임별 신호를
출력하고, '손이 입에 가장 가까운 구간'(rule_distance 최소)을 찾아
그 구간에서 왜 막혔는지(motion gate / rule consecutive / distance) 분석.

사용:
  ai\\venv\\Scripts\\python.exe debug_hand_to_mouth.py <video> [--every N] [--max M]
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipelines.hand_to_mouth_detector import hand_to_mouth_detector  # noqa: E402
from pipelines.hand_to_mouth_fusion import (  # noqa: E402
    MOTION_MIN, RULE_DISTANCE_THRESHOLD, RULE_MIN_CONSECUTIVE,
    LSTM_POSITIVE_THRESHOLD, LSTM_MIN_CONSECUTIVE,
)


def _b64(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=10, help="N프레임마다 한 줄 출력")
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("OPEN_FAIL", args.video)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"# video={Path(args.video).name} fps={fps:.1f} "
          f"MOTION_MIN={MOTION_MIN} RULE_DIST_THR={RULE_DISTANCE_THRESHOLD} "
          f"RULE_CONSEC={RULE_MIN_CONSECUTIVE} LSTM_THR={LSTM_POSITIVE_THRESHOLD} "
          f"LSTM_CONSEC={LSTM_MIN_CONSECUTIVE}")
    hand_to_mouth_detector.reset("dbg")

    rows = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max and n >= args.max):
            break
        r = hand_to_mouth_detector.detect(_b64(frame), camera_id="dbg")
        rows.append((n, r))
        n += 1
    cap.release()

    hdr = ("idx  t(s)  status                 fin src   lstmC lstmD lConsec "
           "ruleD ruleDist rConsec motOK  motion  hand mouth")
    print(hdr)
    print("-" * len(hdr))

    def fmt(i, r):
        t = i / fps if fps else 0
        rd = r.get("rule_distance")
        rd_s = f"{rd:.3f}" if isinstance(rd, (int, float)) else "  -  "
        return (f"{i:4d} {t:5.1f}  {str(r.get('status','?'))[:22]:22s} "
                f"{str(r.get('final_detected'))[0]}  "
                f"{str(r.get('detection_source','none'))[:4]:4s} "
                f"{r.get('lstm_confidence',0):.3f} "
                f"{str(r.get('lstm_detected'))[0]}     "
                f"{r.get('lstm_consecutive',0):3d}    "
                f"{str(r.get('rule_detected'))[0]}     {rd_s:7s} "
                f"{r.get('consecutive_frames',0):3d}     "
                f"{str(r.get('motion_accepted'))[0]}   "
                f"{r.get('motion_score',0):.4f}  "
                f"{str(r.get('has_hand'))[0]}    {str(r.get('has_mouth'))[0]}")

    for i, r in rows:
        if i % args.every == 0:
            print(fmt(i, r))

    # ── 손이 입에 가장 가까운 구간 (has_hand & has_mouth & rule_distance) ──
    cand = [(i, r) for i, r in rows
            if isinstance(r.get("rule_distance"), (int, float))]
    print("\n# === 손-입 거리 분석 ===")
    if not cand:
        print("rule_distance 가 계산된 프레임이 0개 (손+입 동시 검출 프레임 없음).")
    else:
        cand.sort(key=lambda x: x[1]["rule_distance"])
        closest = cand[:8]
        print(f"손+입 동시검출 프레임 수: {len(cand)} / 전체 {len(rows)}")
        print(f"최소 rule_distance: {closest[0][1]['rule_distance']:.4f} "
              f"@frame {closest[0][0]} (threshold={RULE_DISTANCE_THRESHOLD})")
        print("가장 가까운 8프레임 상세:")
        print(hdr)
        for i, r in sorted(closest, key=lambda x: x[0]):
            print(fmt(i, r))
        # 근접(<=threshold) 프레임 통계
        near = [(i, r) for i, r in cand
                if r["rule_distance"] <= RULE_DISTANCE_THRESHOLD]
        near_motion_blocked = [
            1 for _, r in near if not r.get("motion_accepted")
        ]
        print(f"\n거리<=threshold 프레임: {len(near)}개")
        if near:
            print(f"  그 중 motion_accepted=False(게이트 차단): "
                  f"{sum(near_motion_blocked)}개")
            mx_consec = max(r.get("consecutive_frames", 0) for _, r in near)
            print(f"  rule 연속 누적 최대: {mx_consec} "
                  f"(필요 {RULE_MIN_CONSECUTIVE})")

    # ── 상태/소스 분포 ──
    from collections import Counter
    st = Counter(r.get("status") for _, r in rows)
    sc = Counter(r.get("detection_source") for _, r in rows)
    fin = sum(1 for _, r in rows if r.get("final_detected"))
    insuff = sum(1 for _, r in rows if r.get("status") == "insufficient_landmarks")
    forb = set()
    for _, r in rows:
        forb |= ({"taken", "score"} & set(r.keys()))
    print("\n# === 요약 ===")
    print(f"frames={len(rows)} final_detected_true={fin} "
          f"insufficient_landmarks={insuff}")
    print(f"status_dist={dict(st)}")
    print(f"detection_source_dist={dict(sc)}")
    print(f"INVARIANT forbidden_keys(taken/score) in hand_to_mouth: "
          f"{sorted(forb) if forb else 'NONE (정상)'}")


if __name__ == "__main__":
    main()

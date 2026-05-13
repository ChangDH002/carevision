# CareVision 낙상감지 QA 2차 정리

## 1. 목적

본 문서는 CareVision 낙상감지 기능의 웹 런타임 검증 과정에서 확인된 문제와 개선 사항을 정리한 QA 2차 보고용 README이다.

QA 1차에서는 웹 런타임에서 사용되는 낙상감지 모델이 QA 기준 모델과 불일치하는 문제가 확인되었고, 이를 `pytorch_fallback`에서 `pytorch_vB`로 통일하였다.

QA 2차에서는 실제 웹 시연 환경에서 발생한 정상 눕기 동작의 낙상 오탐 문제를 중심으로 검증하였다.

---

## 2. QA 진행 배경

초기 웹 런타임에서는 낙상감지 모델이 다음과 같이 legacy fallback 모델로 동작하였다.

```text
model_backend = pytorch_fallback
sequence_len = 15
num_classes = 2
```

이후 Keras vB 모델을 PyTorch 모델로 변환하여 웹 런타임에서 사용하도록 수정하였다.

최종 적용된 모델 정보는 다음과 같다.

```json
{
  "model_backend": "pytorch_vB",
  "sequence_len": 30,
  "num_features": 132,
  "num_classes": 5,
  "fall_class": 0,
  "qa_validated": true
}
```

생성된 주요 모델 파일은 다음과 같다.

```text
C:\carevision\ai\models\fall_lstm_vB.pt
C:\carevision\ai\models\fall_lstm_vB_meta.json
```

---

## 3. 외부 영상 평가 결과

PyTorch vB 모델 적용 후 외부 영상 50개를 대상으로 평가를 수행하였다.

```text
평가 영상 = 50 / 전체 50
accuracy = 0.8200
fall_precision = 0.9000
fall_recall = 0.7200
fall_f1 = 0.8000
normal_recall = 0.9200
```

Confusion matrix는 다음과 같다.

```text
행=true, 열=pred

true\pred   Fall   Normal
Fall          18       7
Normal         2      23
```

평가 지표만 보면 전체 성능은 나쁘지 않지만, 실제 웹 시연에서는 정상적인 눕기 및 누워 있는 상태에서 낙상으로 오탐되는 문제가 확인되었다.

---

## 4. 확인된 문제

### 4.1 정상 눕기 동작 오탐

실제 웹 테스트에서 사람이 천천히 침대에 눕는 과정에서 일시적으로 Fall로 판단되는 문제가 있었다.

문제 흐름은 다음과 같았다.

```text
천천히 눕기
→ 자세 변화 및 y축 하강 발생
→ 모델 또는 후처리 로직이 Fall 후보로 판단
→ 낙상 알림 또는 낙상 상태 표시
→ 이후 lying_suppressed 또는 정상으로 복귀
```

서비스 관점에서는 이 흐름이 부적절하다.

정상적인 기대 동작은 다음과 같다.

```text
천천히 눕는 중
→ 판정 중

천천히 누운 뒤 안정됨
→ 누워 있음

급격한 낙상
→ 낙상 의심
→ 낙상 발생
```

### 4.2 정적 누움 상태 오탐

누워 있는 상태에서는 motion과 vertical drop이 거의 없지만, 모델이 일시적으로 Fall을 예측하는 경우가 있었다.

이 경우에도 실제로는 낙상 사건이 아니므로, 최종 상태는 `lying_suppressed` 또는 `normal`이어야 한다.

### 4.3 화면 상태와 알림 조건의 분리 부족

초기 로직에서는 `fall_suspected`, `fall_emergency`, `alert_triggered`가 충분히 분리되지 않아, 낙상 후보 상태에서 화면 또는 알림이 과하게 발생할 수 있었다.

---

## 5. 개선 방향

QA 2차에서는 낙상을 단순한 자세가 아니라 “최근 급격한 동적 변화가 발생한 사건”으로 정의하였다.

즉, 낙상 확정 조건을 다음과 같이 강화하였다.

```text
모델이 Fall로 예측
+ Fall 확률이 threshold 이상
+ 최근 vertical drop 존재
+ 최근 motion spike 존재
+ torso angle 변화 존재
+ 연속 window 조건 충족
= 낙상 확정
```

반대로 다음 조건에서는 낙상으로 확정하지 않도록 하였다.

```text
모델이 Fall로 예측하더라도
+ dynamic_fall_event = false
+ motion_score 낮음
+ vertical_drop 부족
+ torso_angle_change 부족
= 낙상 표시 차단
```

---

## 6. 주요 개선 사항

## 6.1 pytorch_vB 웹 런타임 적용

기존 fallback 모델 대신 QA 기준 vB 모델을 웹 런타임에서 사용하도록 수정하였다.

확인 기준은 다음과 같다.

```json
{
  "model_backend": "pytorch_vB",
  "sequence_len": 30,
  "num_classes": 5,
  "fall_class": 0,
  "qa_validated": true
}
```

---

## 6.2 fall_probability와 confidence 분리

기존에는 신뢰도 표시가 Fall 확률과 혼동될 수 있었다.

이를 다음과 같이 분리하였다.

```text
fall_probability = 모델의 Fall 클래스 확률
confidence = 최종 예측 클래스에 대한 신뢰도
```

예시:

```json
{
  "prediction": "Sitting",
  "fall_probability": 0.028,
  "confidence": 0.450
}
```

이 경우 Fall 확률은 낮지만, 자세 분류 자체는 Sitting/Lying 사이에서 애매할 수 있다.

---

## 6.3 lying_suppression 추가

누워 있는 상태 또는 정적 자세를 낙상으로 오탐하지 않기 위해 `lying_suppression` 로직을 추가하였다.

주요 기준은 다음과 같다.

```text
vertical_drop < FALL_VERTICAL_DROP_MIN
motion_score < FALL_MOTION_SCORE_MIN
```

위 조건을 만족하면 모델이 Fall로 예측하더라도 최종 상태를 `lying_suppressed`로 강등한다.

예시 디버그 출력:

```text
final: lying_suppressed
P(Fall): 2.8%
pred: Sitting
drop: 0.000
motion: 0.0000
suppression_reason: lying_no_dynamic_event
```

---

## 6.4 movement_pending 상태 추가

천천히 눕는 과정에서는 바로 낙상으로 표시하지 않고 `movement_pending` 또는 `checking_transition` 상태로 표시하도록 수정하였다.

기대 흐름:

```text
천천히 눕기
→ movement_pending
→ lying_suppressed
```

이 상태에서는 보호자 알림을 보내지 않는다.

```text
movement_pending
alert_triggered = false
```

---

## 6.5 alert_triggered 분리

낙상 의심 상태와 실제 보호자 알림 조건을 분리하였다.

| 상태 | 화면 표시 | 보호자 알림 |
|---|---|---|
| normal | 정상 | X |
| movement_pending | 판정 중 | X |
| lying_suppressed | 누워 있음 | X |
| fall_suspected | 낙상 의심 | X |
| fall_emergency | 낙상 발생 | O |

알림은 반드시 다음 조건에서만 발생하도록 제한하였다.

```text
alert_triggered == true
```

프론트에서도 `alert_triggered` 값을 기준으로 알림을 발생시키도록 수정하였다.

---

## 6.6 dynamic_fall_event gate 추가

정적 누움 상태에서 모델이 Fall로 튀는 문제를 방지하기 위해 `dynamic_fall_event` gate를 추가하였다.

`dynamic_fall_event`는 다음 신호를 조합하여 판단한다.

```text
vertical_drop >= FALL_VERTICAL_DROP_MIN
motion_score >= FALL_DYNAMIC_MOTION_MIN
torso_angle_change >= FALL_TORSO_CHANGE_MIN
```

위 조건이 모두 충족될 때만 `dynamic_fall_event = true`로 판단한다.

### 낙상 확정 조건

```text
fall_probability >= FALL_PROB_THRESHOLD
model_prediction == "Fall"
dynamic_fall_event == true
consecutive_fall_windows >= EMERGENCY_THRESHOLD
```

위 조건을 모두 만족할 때만 `fall_emergency` 상태로 전이한다.

---

## 6.7 dynamic_event=false일 때 fall 라벨 차단

최종적으로 `dynamic_fall_event = false`인 경우에는 `fall_suspected`, `fall_emergency` 라벨 자체가 화면에 표시되지 않도록 수정하였다.

백엔드 상태 머신과 프론트 화면 표시 양쪽에서 이중 방어를 적용하였다.

### 백엔드

```text
dynamic_fall_event == false
→ fall_suspected / fall_emergency 전이 차단
→ movement_pending 또는 lying_suppressed로 강등
```

### 프론트엔드

```text
backend가 fall_* 상태를 보내더라도
dynamic_fall_event == false이면
movement_pending으로 강제 표시
```

알림 역시 다음 조건을 모두 만족할 때만 발생한다.

```text
alert_triggered == true
dynamic_fall_event == true
```

---

## 7. 추가된 주요 API 응답 필드

QA 2차 과정에서 디버깅과 상태 확인을 위해 다음 필드들이 추가 또는 유지되었다.

```json
{
  "final_decision": "movement_pending | lying_suppressed | fall_suspected | fall_emergency | normal",
  "alert_triggered": false,
  "alert_reason": null,
  "fall_probability": 0.0,
  "confidence": 0.0,
  "model_prediction": "Sitting",
  "dynamic_fall_event": false,
  "dynamic_gate_reason": "static_state",
  "vertical_drop": 0.0,
  "motion_score": 0.0,
  "torso_angle_change": 0.0,
  "consecutive_fall_windows": 0,
  "pending_windows": 0,
  "suppression_reason": "lying_no_dynamic_event"
}
```

---

## 8. 환경변수

QA 2차 기준 주요 환경변수는 다음과 같다.

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `FALL_PROB_THRESHOLD` | `0.65` | Fall 클래스 확률 임계값 |
| `FALL_CONFIRM_WINDOWS` | `3` | 판정 중 상태 유지/확인 window 수 |
| `FALL_CONSECUTIVE_THRESHOLD` | `3` | 연속 fall window 기준 |
| `FALL_VERTICAL_DROP_MIN` | `0.10` | vertical drop 최소 기준 |
| `FALL_MOTION_SCORE_MIN` | `0.005` | lying suppression용 motion 기준 |
| `FALL_SLOW_MOTION_MAX` | `0.020` | 천천히 눕기 판단 motion 상한 |
| `FALL_DYNAMIC_MOTION_MIN` | `0.020` | dynamic event 판단 motion 기준 |
| `FALL_TORSO_CHANGE_MIN` | `30.0` | torso angle 변화 기준 |

---

## 9. 웹 테스트 시나리오 및 기대 결과

| 시나리오 | 기대 상태 흐름 | dynamic_event | alert |
|---|---|---|---|
| 서 있기 | normal | false | off |
| 걷기 | normal | false | off |
| 앉기 | normal 또는 movement_pending 후 normal | false | off |
| 천천히 눕기 | movement_pending → lying_suppressed | false | off |
| 누워서 가만히 있기 | lying_suppressed | false | off |
| 누운 상태에서 모델이 Fall로 튐 | lying_suppressed 또는 movement_pending | false | off |
| 실제 급격한 낙상 | movement_pending → fall_suspected → fall_emergency | true | ON |

---

## 10. 검증 결과 요약

### 10.1 정상 눕기 동작

기존 문제:

```text
천천히 눕는 중 화면에 낙상 의심 또는 낙상 발생 표시
```

개선 후 기대 결과:

```text
천천히 눕는 중 → 판정 중
안정 후 → 누워 있음
alert_triggered = false
```

### 10.2 정적 누움 상태

기존 문제:

```text
누워 있는 상태에서 모델이 Fall로 튀면 낙상으로 표시
```

개선 후 기대 결과:

```text
dynamic_fall_event = false
final_decision = lying_suppressed 또는 movement_pending
alert_triggered = false
```

### 10.3 실제 급격한 낙상

기대 결과:

```text
dynamic_fall_event = true
consecutive_fall_windows 증가
fall_suspected → fall_emergency
alert_triggered = true
```

---

## 11. 테스트 실행 방법

### 11.1 pytest 실행

```cmd
cd C:\carevision
.venv311\Scripts\activate
py -m pytest -q qa/test_fall_api.py
```

기대 결과:

```text
6 passed
```

---

### 11.2 AI 서버 실행

```cmd
cd C:\carevision
.venv311\Scripts\activate
uvicorn ai.main:app --reload --port 8000
```

경로 오류가 발생하면 다음 방식으로 실행한다.

```cmd
cd C:\carevision\ai
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

---

### 11.3 모델 정보 확인

브라우저에서 다음 주소를 확인한다.

```text
http://127.0.0.1:8000/detect/model_info
```

정상 기준:

```json
{
  "model_backend": "pytorch_vB",
  "sequence_len": 30,
  "num_classes": 5,
  "fall_class": 0,
  "qa_validated": true
}
```

---

## 12. QA 2차 결론

QA 2차에서는 실제 웹 시연 중 발생한 정상 눕기 동작의 낙상 오탐 문제를 개선하였다.

기존에는 천천히 눕거나 누운 상태에서 모델이 일시적으로 Fall을 예측하면 화면에 낙상 상태가 표시되거나 알림 조건에 가까워지는 문제가 있었다.

이를 해결하기 위해 다음 개선을 적용하였다.

```text
1. lying_suppression으로 정적 누움 상태 억제
2. movement_pending으로 눕는 중간 상태 분리
3. alert_triggered를 fall_emergency와 분리
4. dynamic_fall_event gate로 실제 동적 낙상 사건만 낙상 후보로 인정
5. dynamic_event=false일 때 fall_suspected/fall_emergency 화면 표시 차단
6. 프론트에서도 dynamic_event=false일 때 fall 라벨을 movement_pending으로 강제 변환
```

최종적으로 정상 눕기 동작은 `movement_pending → lying_suppressed`로 처리하고, 실제 급격한 낙상만 `fall_suspected → fall_emergency`로 전이하도록 개선하였다.

---

## 13. 남은 이슈

현재 QA 2차 기준으로 정상 눕기 오탐 문제는 개선되었으나, 추가로 확인된 이슈가 있다.

```text
실제 낙상을 fall_emergency로 판정한 이후,
사람이 오래 누워 있으면 다시 lying_suppressed 또는 누워 있음으로 돌아갈 수 있음.
```

이 문제는 모델 판정 문제가 아니라 **낙상 사건 상태 유지 로직**의 문제이다.

향후 개선 방향:

```text
fall_incident_active 상태 추가
fall_emergency 발생 후 명시적 reset 전까지 사건 상태 유지
알림은 최초 1회만 발송
이후 화면은 낙상 발생 / 확인 필요 상태 유지
```

---

## 14. 향후 보완 방향

### 14.1 fall_incident_active 상태 추가

낙상이 한 번 확정되면 사람이 이후 가만히 누워 있어도 자동으로 정상 또는 누워 있음 상태로 돌아가지 않도록 사건 상태를 유지해야 한다.

예상 상태 흐름:

```text
fall_emergency
→ fall_incident_active
→ 사용자 또는 보호자 reset
→ normal
```

### 14.2 데이터셋 보강

현재 문제의 1차 원인은 백엔드 상태 전이 로직이지만, 장기적으로는 데이터셋 보강도 필요하다.

보강이 필요한 hard negative 데이터:

```text
천천히 눕기
침대에 눕기
바닥에 눕기
누워서 자기
앉았다가 눕기
빠르게 침대에 눕기
낙상 후 누워 있음과 정상 누움 구분 데이터
```

데이터셋 보강은 현재 후처리 로직 안정화 이후 진행하는 것이 적절하다.

---

---

## 15. 한 줄 요약

```text
QA 2차에서는 정상 눕기와 정적 누움 상태를 낙상으로 오탐하지 않도록, 
movement_pending, lying_suppression, alert_triggered 분리, dynamic_fall_event gate를 적용하였다.
실제 낙상 이후 상태 유지 문제는 fall_incident_active 추가가 필요한 다음 개선 항목으로 남겨두었다.
```

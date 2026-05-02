# Fall Detection 최소 QA 테스트

이 문서는 낙상감지 FastAPI 엔드포인트의 **최소 동작 확인** 테스트 방법을 설명합니다.

## 파일
- `qa/test_fall_api.py`

## 테스트 범위
1. `/health` 헬스체크
2. `/detect/reset` 상태 초기화 응답 검증
3. `/detect/fall` 기본 응답 구조 검증
4. `/detect/fall` 잘못된 payload(422) 검증

## 왜 mock을 쓰나요?
테스트 환경에서 `mediapipe`, `tensorflow`, `torch`, 모델 파일 등이 없어도
엔드포인트 레벨 계약(상태코드/JSON 구조)을 검증하기 위해,
테스트 코드에서만 무거운 파이프라인 모듈을 test double로 주입합니다.

> 핵심 앱 코드(`ai/` 내부)는 수정하지 않습니다.

## 실행 방법
프로젝트 루트에서:

```bash
pytest -q qa/test_fall_api.py
```

또는 상세 로그:

```bash
pytest -vv qa/test_fall_api.py -rs
```

## 최소 필요 패키지 (테스트 실행용)
- `pytest`
- `fastapi`
- `starlette`

AI 추론용 패키지(`mediapipe`, `tensorflow`, `torch`, `ultralytics`)는
이 테스트 파일에서 mock 처리하므로 필수 아님.

## 참고: requirements 누락 후보
현재 AI 코드 기준 `ai/services/backend_client.py`는 `httpx`를 import하지만,
루트 `requirements.txt`에는 `httpx`가 명시되어 있지 않습니다.

# 검증 상태

이 문서는 구현 존재와 실행 검증을 구분한다.

- Python 3.12.14 문법 검사: 전체 Python 파일 통과.
- 근접 규칙 단위 테스트: 8개 통과 (2026-09-17).
- 전체 테스트 실행: 9개 중 8개 통과, MP4 통합 테스트 1개는 의존성 미설치로 skip.
- 실제 YOLO26s 추론, ByteTrack 연동, MP4 출력: 아직 실행 검증하지 않음.
- MP4 통합 테스트: 작성됨. OpenCV/PyYAML/NumPy 설치가 필요함.
- 학습: 데이터 미제공으로 미실행.
- 현장 성능: 산업현장 MP4와 person/forklift 가중치 미제공으로 미측정.

검증 환경에는 Python만 준비되어 있고 Ultralytics, PyTorch, OpenCV, PyYAML은 설치되지 않았다. 의존성을 설치한 환경에서 모델 기반 파이프라인을 추가 검증해야 한다.

## GitHub Actions 검증

2026-09-17, 초기 코드 커밋 `650d54a`의 [자동 테스트 실행](https://github.com/k-j-hero/FactoryGuard/actions/runs/35177293320)이 성공했다. Linux/Python 3.12 환경에 OpenCV, PyYAML, NumPy를 설치한 뒤 규칙 테스트와 MP4 통합 테스트를 실행했다.

MP4 통합 테스트는 합성 입력 영상과 스크립트 탐지 결과로 annotated MP4, 이벤트 CSV, 이벤트 클립의 생성·재생 및 프레임 수를 검사한다. YOLO26s나 실제 ByteTrack 추적을 실행하는 검증은 포함하지 않는다.

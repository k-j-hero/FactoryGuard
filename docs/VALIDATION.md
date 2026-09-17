# 검증 상태

이 문서는 구현 존재와 실행 검증을 구분한다.

- Python 3.12.14 문법 검사: 전체 Python 파일 통과.
- 근접 규칙 단위 테스트: 8개 통과 (2026-09-17).
- 전체 테스트 실행: 로컬 Windows/GPU 환경과 GitHub Actions에서 9개 모두 통과.
- 실제 YOLO26s 추론, ByteTrack 연동, MP4 출력: person detection-only 범위에서 실행 검증 완료.
- MP4 통합 테스트: 작성됨. OpenCV/PyYAML/NumPy 설치가 필요함.
- 학습: 데이터 미제공으로 미실행.
- 현장 성능: 산업현장 MP4와 person/forklift 가중치 미제공으로 미측정.

## 로컬 GPU 실제 영상 검증

2026-09-17에 Pexels video 4294434의 첫 300프레임을 1280×720으로 축소해 실제 추론했다.

| 항목 | 결과 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| Python / PyTorch / CUDA | 3.12.14 / 2.12.1+cu130 / 13.0 |
| Ultralytics / OpenCV | 8.4.154 / 4.14.0 |
| 모델 / tracker | yolo26s.pt / ByteTrack |
| 영상 | 1280×720, 25 FPS, 300프레임, 12초 |
| 처리 | 6.715초, 약 44.7 FPS (파일 I/O와 annotation 포함) |
| 추적 | person 1명, 300/300프레임에서 track ID 1 유지 |
| 출력 | annotated.mp4 15.58MB, tracks.csv, summary.json |
| 상태 | status=complete, mode=detection_only |

프레임 0, 200, 299를 육안 확인해 person 박스와 ID 1이 유지되는 것을 확인했다. 이 수치는 한 샘플 구간의 실행 결과이며 일반화된 정확도나 처리 성능 벤치마크가 아니다.

COCO 사전학습 모델에는 forklift 클래스가 없어서 지게차는 탐지하지 않았고 이벤트도 생성하지 않았다. 따라서 이 검증은 실제 GPU에서 YOLO26s → ByteTrack → annotated MP4/CSV 파이프라인이 동작함을 보인다. forklift 탐지, normalized proximity, potential near-miss candidate의 실제 영상 검증에는 person/forklift 데이터로 학습한 best.pt가 필요하다.

## GitHub Actions 검증

2026-09-17, 초기 코드 커밋 `650d54a`의 [자동 테스트 실행](https://github.com/k-j-hero/FactoryGuard/actions/runs/35177293320)이 성공했다. Linux/Python 3.12 환경에 OpenCV, PyYAML, NumPy를 설치한 뒤 규칙 테스트와 MP4 통합 테스트를 실행했다.

MP4 통합 테스트는 합성 입력 영상과 스크립트 탐지 결과로 annotated MP4, 이벤트 CSV, 이벤트 클립의 생성·재생 및 프레임 수를 검사한다. YOLO26s나 실제 ByteTrack 추적을 실행하는 검증은 포함하지 않는다.

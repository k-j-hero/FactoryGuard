# 검증 상태

이 문서는 코드 동작, 데이터셋 탐지 성능, 외부 영상 결과를 구분해 기록한다.

## 자동 테스트

- Python 3.12.14 환경에서 전체 Python 파일 문법 검사 통과.
- 근접 규칙과 MP4 저장 통합 테스트 9개가 로컬 Windows에서 통과.
- 초기 코드 커밋 `650d54a`의 [GitHub Actions 실행](https://github.com/k-j-hero/FactoryGuard/actions/runs/35177293320)도 같은 9개 테스트 통과.
- MP4 통합 테스트는 합성 입력과 스크립트 탐지 결과로 annotated MP4, 이벤트 CSV, 이벤트 클립의 생성·재생·프레임 수를 검사한다. 실제 YOLO 정확도 테스트는 아니다.

## 학습 데이터

2026-09-17에 [Warehouse Safety v7](https://universe.roboflow.com/s-workspace-zi5d1/warehouse-safety-rhspm-3u50l/dataset/7)의 YOLO26 export를 사용했다. 데이터셋 제공 파일은 CC BY 4.0으로 표시하며, 640×640 이미지 906장으로 구성된다.

| split | 이미지 | forklift 박스 | person 박스 | 빈 라벨 이미지 |
| --- | ---: | ---: | ---: | ---: |
| train | 788 | 273 | 1,716 | 17 |
| validation | 79 | 20 | 163 | 2 |
| test | 39 | 7 | 77 | 4 |

이미지와 라벨 파일은 각 split에서 모두 대응했고, 유효 범위를 벗어난 라벨은 발견되지 않았다. 파일명의 원본 frame ID 기준으로 split 간 중복도 발견되지 않았다. person 박스가 forklift보다 약 6.5배 많고 test의 forklift 정답이 7개뿐이라는 제한이 있다.

## YOLO26s 학습 결과

| 항목 | 값 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| Python / PyTorch / CUDA | 3.12.14 / 2.12.1+cu130 / 13.0 |
| Ultralytics / OpenCV | 8.4.154 / 4.14.0 |
| 입력 / batch / seed | 640 / 8 / 42 |
| 예정 epoch / 실제 종료 | 80 / 53 (patience 15 early stopping) |
| 최고 checkpoint | epoch 38 `best.pt` |
| 학습 시간 | 약 0.216시간 (약 13분) |

최고 checkpoint의 validation 결과:

| 클래스 | 정답 수 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 183 | 0.748 | 0.969 | 0.892 | 0.814 |
| forklift | 20 | 0.591 | 0.950 | 0.802 | 0.732 |
| person | 163 | 0.904 | 0.988 | 0.982 | 0.897 |

학습 중 모델 선택에 사용하지 않은 test split 결과:

| 클래스 | 정답 수 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 84 | 0.824 | 0.968 | 0.852 | 0.782 |
| forklift | 7 | 0.680 | 1.000 | 0.722 | 0.685 |
| person | 77 | 0.969 | 0.935 | 0.981 | 0.879 |

test의 forklift recall 1.000은 정답 7개에 대한 결과다. 표본이 작고 촬영 시점도 같은 데이터 계열이므로 현장 영상의 일반 성능을 뜻하지 않는다.

## 외부 영상 검증

Pexels video 4294434의 첫 300프레임(3840×2160, 25 FPS, 12초)을 `best.pt`와 ByteTrack으로 분석했다.

| 항목 | 결과 |
| --- | --- |
| 처리 | 19.88초, 약 15.1 FPS (파일 I/O와 4K annotation 포함) |
| 추적 로그 | person 39행, forklift 0행 |
| 이벤트 / 클립 | 0 / 0 |
| 상태 | 실행 완료, proximity 모드 활성화 |

영상에는 지게차가 보이지만 모델은 confidence 0.01까지 낮춘 점검에서도 forklift를 탐지하지 못했다. 사람도 전체 몸 대신 일부만 잡히거나 ID가 바뀌었다. 학습 데이터가 bird's-eye view이고 외부 영상은 사람 뒤에서 지게차로 접근하는 지상 시점이라 생긴 도메인 차이가 주요 원인으로 보인다. 따라서 이벤트 0건은 안전 판정이 아니라 탐지 누락 결과다.

## 실제 모델을 사용한 출력 파이프라인 증명

test의 `frame_00065`를 25 FPS, 3초 길이로 반복해 `best.pt` 탐지와 ByteTrack, 이벤트 기록, 클립 추출을 함께 실행했다. 예측 박스 기준 normalized proximity가 약 0.085여서 이 증명에서는 `enter_threshold=0.09`, `exit_threshold=0.12`를 사용했다.

- 75프레임에서 forklift 1개와 person 3개를 지속 추적.
- person ID 4와 forklift ID 2의 potential near-miss candidate 1건 기록.
- `events.csv`, `events.jsonl`, 75프레임 event clip 1개 생성.
- 최소 normalized proximity 0.0852, 관측 구간 3.0초.

이는 동일한 정지 이미지를 반복한 저장 파이프라인 증명이다. 실제 접근 동작의 이벤트 탐지 성능을 측정한 결과가 아니다.

## 사전학습 모델 smoke test

같은 Pexels 영상의 첫 300프레임을 COCO 사전학습 `yolo26s.pt`로 먼저 실행했다. 1280×720 버전에서 person 1명을 300/300프레임 동안 ID 1로 유지했고, 6.715초(약 44.7 FPS)에 처리했다. COCO 모델에 forklift 클래스가 없어 이 실행은 detection-only 모드였다.

현재 결과는 MVP가 실제 GPU에서 탐지·추적·로그·클립을 생성함을 보여준다. 외부 영상의 forklift 누락을 줄이려면 지상 시점 forklift 데이터를 추가하고, 촬영 단위로 분리한 새 test set에서 다시 평가해야 한다.

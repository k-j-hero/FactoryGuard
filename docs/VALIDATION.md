# 검증 상태

이 문서는 코드 동작, 데이터셋 탐지 성능, 외부 영상 결과를 구분해 기록한다.

## 자동 테스트

- Python 3.12.14 환경에서 전체 Python 파일 문법 검사 통과.
- 근접 규칙, MP4 저장, 데이터 감사·정제, 실행 진단, 통합 데이터 경로, W&B import 변환 테스트 20개가 로컬 Windows에서 통과.
- 커밋 `0b2fd3c`의 [GitHub Actions 실행](https://github.com/k-j-hero/FactoryGuard/actions/runs/35287626070)에서 최신 15개 테스트가 모두 통과했다.
- MP4 통합 테스트는 합성 입력과 스크립트 탐지 결과로 annotated MP4, 이벤트 CSV, 이벤트 클립의 생성·재생·프레임 수를 검사한다. 실제 YOLO 정확도 테스트는 아니다.

## 학습 데이터

2026-09-17에 [Warehouse Safety v7](https://universe.roboflow.com/s-workspace-zi5d1/warehouse-safety-rhspm-3u50l/dataset/7)의 YOLO26 export를 사용했다. 데이터셋 제공 파일은 CC BY 4.0으로 표시하며, 640×640 이미지 906장으로 구성된다.

| split | 이미지 | forklift 박스 | person 박스 | 빈 라벨 이미지 |
| --- | ---: | ---: | ---: | ---: |
| train | 788 | 273 | 1,716 | 17 |
| validation | 79 | 20 | 163 | 2 |
| test | 39 | 7 | 77 | 4 |

이미지와 라벨 파일은 각 split에서 모두 대응했고, 유효 범위를 벗어난 라벨은 발견되지 않았다. 파일명의 원본 frame ID 기준으로 split 간 중복도 발견되지 않았다. person 박스가 forklift보다 약 6.5배 많고 test의 forklift 정답이 7개뿐이라는 제한이 있다.

외부 지상 시점 영상에서 forklift가 누락된 뒤 [Forklift v1](https://universe.roboflow.com/helmetaiworkspace/forklift-dsitv-yhncw/dataset/1)을 보강 데이터로 추가했다. 이 데이터도 CC BY 4.0으로 표시되어 있으며 원본 421장, 전처리·증강 없음으로 게시돼 있다.

| 보강 split | 이미지 | forklift 박스 | person 박스 | 빈 라벨 이미지 |
| --- | ---: | ---: | ---: | ---: |
| train | 295 | 319 | 296 | 1 |
| validation | 84 | 96 | 75 | 0 |
| test | 42 | 43 | 44 | 0 |

통합 데이터는 train 1,083장, validation 163장, test 81장이다. `audit_dataset.py` 검사에서 이미지-라벨 누락, 잘못된 정규화 좌표, 파일 내용이 완전히 같은 split 간 중복, Roboflow 파일명 기반 원본 후보 중복은 발견되지 않았다. 검사 결과는 `reports/`의 JSON에 남겼다. 이 자동 검사는 잘못된 박스 의미나 같은 영상의 서로 다른 프레임 누출까지 보장하지 않는다.

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

## 지상 시점 보강 통합 학습

동일한 YOLO26s, 640 입력, batch 8, seed 42로 통합 데이터 1,083장을 80 epoch 학습했다. 전체 학습 시간은 1,590초(약 26.5분)였고 75번째 epoch의 `best.pt`가 선택됐다.

통합 validation 결과:

| 클래스 | 정답 수 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 354 | 0.869 | 0.873 | 0.910 | 0.719 |
| forklift | 116 | 0.853 | 0.851 | 0.899 | 0.675 |
| person | 238 | 0.884 | 0.895 | 0.921 | 0.762 |

통합 test 결과:

| 클래스 | 정답 수 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 전체 | 171 | 0.923 | 0.856 | 0.893 | 0.664 |
| forklift | 50 | 0.887 | 0.940 | 0.924 | 0.628 |
| person | 121 | 0.959 | 0.772 | 0.863 | 0.699 |

지상 시점 보강 test 42장만 동일하게 평가하면 baseline 모델의 forklift Recall/mAP50-95는 0.140/0.008이었고 통합 모델은 0.953/0.637이었다. 원래 Warehouse test 39장에서도 통합 모델의 forklift mAP50-95는 0.786으로 baseline 0.685보다 낮아지지 않았다. 이 비교는 각 제공 split 안에서의 결과이며 실제 현장 일반화를 보장하지 않는다. 원시 수치는 `reports/model-evaluation.json`에 기록했다.

## W&B 기록 검증

`log_wandb.py`로 통합 학습의 `results.csv` 80개 epoch, 학습 설정, 곡선 이미지, 평가 비교표를 W&B 0.30.0 오프라인 run으로 가져왔다. 로컬 run은 `runs/train/combined_yolo26s/wandb/offline-run-20260918_083036-wxl58ot9`에 생성됐고 최고 epoch 75와 mAP50-95 0.71861을 동일하게 기록했다.

새 학습의 자동 callback도 지상 시점 데이터 1 epoch smoke run으로 검증했다. `runs/train/wandb/offline-run-20260918_083410-1fscae27`에 train/validation metrics, 곡선과 20.3MB `best.pt` artifact가 생성됐다. 이는 W&B 연결 검증용 실행이며 모델 성능 비교에는 사용하지 않는다. 계정 대시보드 공유 링크는 로그인 후 run을 sync한 뒤 추가한다.

## 외부 영상 검증

Pexels video 4294434의 첫 300프레임(3840×2160, 25 FPS, 12초)을 baseline과 통합 `best.pt`, ByteTrack으로 분석했다.

| 모델 / 입력 | 처리 | 추적 로그 | 이벤트 / 클립 | 육안 검토 |
| --- | ---: | --- | ---: | --- |
| baseline / 640 | 19.88초 | person 39행, forklift 0행 | 0 / 0 | 명백한 forklift 누락 |
| 통합 / 640 | 35.52초 | person 331행, forklift 340행 | 2 / 2 | 실제 지게차 존재는 회복했으나 박스가 과도하게 크고 추가 오탐 존재 |
| 통합 / 640, 안정형 추적 | 300프레임 | person 235행, forklift 306행 | 2 / 2 | 고유·단기 ID와 겹침은 감소, 과대 forklift 박스는 유지 |
| 통합 / 1280 | 46.14초 | person 263행, forklift 674행 | 4 / 4 | 중복·오탐과 ID 분할이 늘어 640보다 나쁨 |

통합 모델은 영상 중앙의 실제 지게차 존재를 탐지하고 접근 구간 후보를 생성했다. 그러나 forklift 박스가 화면의 큰 영역까지 확장되고 사람 박스도 중복되는 프레임이 있어 2건을 정확한 이벤트 탐지 성공으로 계산하지 않았다. 1280 입력은 작은 객체 정보를 늘렸지만 중복 박스와 ID 분할도 늘었다. 현재 결과는 도메인 누락을 줄인 증거이면서, 외부 영상에 대한 박스 품질과 이벤트 precision을 추가로 개선해야 한다는 실패 사례다.

동일한 통합 모델에서 NMS IoU를 0.50으로 낮추고 새 track 생성 기준을 0.40으로 높인 설정은 forklift 고유 ID를 15개에서 10개, person 고유 ID를 16개에서 6개로 줄였다. 같은 클래스 박스가 IoU 0.5 이상 겹치는 프레임도 forklift 6개와 person 81개에서 모두 0개가 됐다. 반면 forklift 박스 면적 90백분위는 52.1%에서 51.8%로 거의 변하지 않았다. 이 결과는 추적 설정이 ID 요동을 완화하지만 과대 박스의 학습 원인은 해결하지 못함을 보여준다.

## 실제 모델을 사용한 출력 파이프라인 증명

test의 `frame_00065`를 25 FPS, 3초 길이로 반복해 `best.pt` 탐지와 ByteTrack, 이벤트 기록, 클립 추출을 함께 실행했다. 예측 박스 기준 normalized proximity가 약 0.085여서 이 증명에서는 `enter_threshold=0.09`, `exit_threshold=0.12`를 사용했다.

- 75프레임에서 forklift 1개와 person 3개를 지속 추적.
- person ID 4와 forklift ID 2의 potential near-miss candidate 1건 기록.
- `events.csv`, `events.jsonl`, 75프레임 event clip 1개 생성.
- 최소 normalized proximity 0.0852, 관측 구간 3.0초.

이는 동일한 정지 이미지를 반복한 저장 파이프라인 증명이다. 실제 접근 동작의 이벤트 탐지 성능을 측정한 결과가 아니다.

## 사전학습 모델 smoke test

같은 Pexels 영상의 첫 300프레임을 COCO 사전학습 `yolo26s.pt`로 먼저 실행했다. 1280×720 버전에서 person 1명을 300/300프레임 동안 ID 1로 유지했고, 6.715초(약 44.7 FPS)에 처리했다. COCO 모델에 forklift 클래스가 없어 이 실행은 detection-only 모드였다.

현재 결과는 MVP가 실제 GPU에서 탐지·추적·로그·클립을 생성하고 지상 시점 보강으로 forklift 누락을 줄일 수 있음을 보여준다. 다음 개선은 이 외부 영상과 유사한 원거리·부분 가림 장면을 별도 validation으로 만들고, 잘못 커지는 박스와 중복 person 탐지를 줄인 뒤 사람이 표시한 이벤트 구간과 비교하는 것이다.

## 큰 박스 필터 ablation

보강 데이터 421장 중 객체 박스가 화면 면적의 70% 이상인 이미지 152장을 제외한 파생본으로 같은 YOLO26s를 80 epoch 학습했다. 자세한 선별 근거와 contact sheet는 [데이터 품질 검토](DATA_QUALITY.md)에 기록했다.

외부 영상 300프레임에서 거대한 forklift 오탐과 ID 분할은 크게 감소했다. forklift 추적 행은 340에서 49, 고유 ID는 15에서 4, 프레임당 최대 박스는 3에서 1로 줄었다. person 고유 ID도 16에서 9로 줄었다. 그러나 실제 지게차를 탐지하는 프레임도 함께 줄어, 이벤트 후보는 2건에서 0건이 됐다. 정답 라벨이 없으므로 외부 영상 Recall 수치는 주장하지 않는다.

이 결과는 데이터 구성 변경이 문제의 원인에 직접 영향을 준다는 증거이지만, 큰 박스 일괄 제외가 최종 해법이라는 증거는 아니다. 다음 모델은 목표 시점에서 직접 라벨링한 지게차 양성 프레임과 선반 hard negative를 함께 사용해야 한다.

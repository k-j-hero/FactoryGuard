# FactoryGuard

**산업현장 영상에서 작업자와 지게차를 추적하고, 화면상 close approach를 검토할 수 있는 클립으로 남기는 MVP.**

YOLO26s → ByteTrack → normalized proximity → potential near-miss candidate → annotated MP4 / CSV / event clips.

화면상 근접 후보를 검토하는 도구입니다. 실제 거리(m), 사고 확률, 충돌 여부를 추정하지 않습니다. PPE, PLC, DB, 웹 대시보드는 이 버전의 범위에 포함하지 않습니다.

## 구현 상태

- [x] 로컬 MP4 입력, YOLO 탐지와 ByteTrack 추적 연결
- [x] 클래스 이름에 따른 person/forklift 선택 (클래스 ID 순서 독립)
- [x] 근접 규칙, 최소 관측 시간, 진입/해제 임계값, 추적 누락 처리
- [x] annotated MP4, tracks.csv, events.csv, events.jsonl, summary.json
- [x] 이벤트 전후 annotated MP4 클립 추출 및 별도 재추출 명령
- [x] YOLO 형식 데이터 경로 확인 및 YOLO26s 학습 명령
- [x] VS Code 작업 공간/디버그 설정, 단위 테스트, GitHub Actions 설정
- [ ] 실제 YOLO26s 가중치를 이용한 실행 검증
- [ ] 사용자 산업현장 영상과 학습 데이터로 성능 평가

현재 검증 범위는 [docs/VALIDATION.md](docs/VALIDATION.md)를 확인하세요. 실제 학습 결과나 탐지 성능 수치를 아직 보고하지 않습니다.

**처음 보는 경우:** [결과물을 어디서 어떻게 보는지](docs/VIEWING.md)부터 확인하세요.

**테스트할 영상이 없다면:** [공개 샘플로 첫 실제 추론 실행](docs/FIRST_DEMO.md). 전용 가상환경에서 설치부터 짧은 결과 영상 생성까지 진행하는 명령을 제공합니다.

## 프로젝트 구조

```text
FactoryGuard/
├── infer.py                    # 영상 탐지·추적·이벤트 파이프라인
├── train.py                    # 데이터 경로 확인 및 fine-tuning
├── extract_events.py           # CSV 기반 클립 재추출
├── factoryguard/
│   ├── config.py               # 설정 검증, 모델 클래스 매핑
│   ├── proximity.py            # 화면상 근접도와 pair별 이벤트 상태
│   ├── pipeline.py             # 순차 프레임 처리 및 결과 기록
│   └── clips.py                # annotated MP4 클립 추출
├── configs/                    # 추론, 학습, ByteTrack, 데이터 예시
├── tests/                      # 규칙 및 영상 저장 통합 테스트
├── docs/                       # 검증 상태, 포트폴리오 기록 양식
├── .vscode/launch.json
├── FactoryGuard.code-workspace
├── requirements.txt
└── .github/workflows/tests.yml
```

## 1. VS Code에서 같이 작업하기

이 폴더를 **File → Open Folder**로 열거나 `FactoryGuard.code-workspace`를 여세요. Codex에도 동일한 `FactoryGuard` 폴더를 로컬 프로젝트로 추가하면 같은 파일을 기준으로 작업할 수 있습니다. 함께 변경을 보려면 동일한 로컬 폴더에서 작업하세요. 별도 worktree는 다른 작업 사본이므로 그 폴더도 따로 열어야 합니다.

아래 명령은 모두 **FactoryGuard 폴더의 터미널**에서 실행합니다. Python 3.10 이상을 사용하세요 (작성 시 문법 검증: Python 3.12).

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

VS Code의 `Python: Select Interpreter`에서 이 프로젝트의 `.venv`를 선택합니다. PowerShell 실행 정책을 바꿀 필요 없이 위처럼 가상환경 Python 경로를 직접 사용할 수 있습니다. 이하 `python`은 선택한 가상환경의 Python을 뜻합니다. 터미널에서 자동 활성화되지 않았다면 `.\.venv\Scripts\python.exe`로 바꾸세요.

기본 장치는 CPU입니다. GPU 학습/추론을 하려면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 맞는 CUDA 지원 PyTorch를 설치하고 `--device 0`을 지정하세요. CPU는 실행 확인용이며 하루 안의 학습에는 GPU 사용을 전제로 시간을 잡는 것이 좋습니다.

## 2. 먼저 pretrained YOLO26s로 영상 저장 확인

```bash
python infer.py --source "data/videos/site.mp4" --output runs/pretrained01 --model yolo26s.pt --max-frames 100
```

첫 실행은 Ultralytics가 모델 가중치를 다운로드하므로 네트워크가 필요합니다. 오프라인 환경은 미리 받은 로컬 `.pt` 경로를 지정하세요.

**COCO 사전학습 YOLO26s에는 forklift 클래스가 없습니다.** 이 명령은 person 탐지·추적과 영상 저장을 확인하는 `detection_only` 실행입니다. 콘솔과 영상에 해당 상태를 표시하고 이벤트 CSV는 헤더만 남깁니다. car/truck을 forklift로 바꾸지 않습니다. 실제 이벤트 기능은 person/forklift로 학습한 모델에서 활성화됩니다.

VS Code에서 F5를 누르고 `FactoryGuard: pretrained smoke test`를 선택해도 됩니다. 영상 경로와 결과 폴더를 입력합니다. 결과 덮어쓰기를 방지하므로 실행마다 새 출력 폴더를 지정하세요.

## 3. 학습한 best.pt로 전체 분석

```bash
python infer.py --source "data/videos/site.mp4" --output runs/site01 --model "runs/train/warehouse_yolo26s/weights/best.pt" --require-forklift --device 0
```

CPU에서는 `--device cpu`로 바꾸세요. `--require-forklift`는 모델에 forklift 클래스가 없을 때 즉시 오류를 냅니다. `model.names`에서 클래스 이름을 확인하므로 person/forklift의 ID 순서가 바뀌어도 동작합니다. 커스텀 이름은 설정의 `person_names`/`forklift_names`를 실제 의미에 맞게 지정하세요.

출력:

| 파일 | 내용 |
| --- | --- |
| `annotated.mp4` | 박스, 추적 ID, 근접선, normalized proximity, 후보 상태 |
| `tracks.csv` | 프레임, 영상 시간, 추적 ID, 클래스, confidence, 박스 좌표 |
| `events.csv` | 종료된 후보 이벤트 한 건당 한 행 |
| `events.jsonl` | 이벤트 종료 시 즉시 flush하는 동일 내용 로그 |
| `clips/event_00001.mp4` | 이벤트 전후 구간을 포함한 annotated clip |
| `clips/clips.csv` | 클립 파일과 실제 추출 구간 |
| `summary.json` | 완료 여부, 모델 클래스, 처리 프레임, 이벤트 수 |
| `config.resolved.json` | 실행에 사용한 설정 |

오류 시 `FAILED.txt`를 남깁니다. `summary.json`이 없거나 `FAILED.txt`가 있으면 완성된 실행 결과로 취급하지 마세요. 처리 도중 중단되면 이미 종료된 이벤트만 로그에 있고, 열린 이벤트와 클립은 완성되지 않을 수 있습니다.

## 4. normalized proximity 규칙

각 박스의 하단 중심을 화면상 기준점으로 사용합니다.

```text
anchor = ((x1 + x2) / 2, y2)
normalized_proximity = ||person_anchor - forklift_anchor|| / sqrt(width² + height²)
```

값이 작을수록 영상 좌표상 가깝습니다. `0.08`은 화면 대각선의 8%에 해당하는 **영상 좌표 거리**입니다. 미터 단위가 아닙니다. 카메라 원근을 보정하지 않습니다.

- `enter_threshold: 0.08`: 이 값 이하에서 pair의 close approach 구간 시작.
- `exit_threshold: 0.11`: 이 값 초과에서 구간 종료. 그 사이 값은 현재 상태 유지.
- `min_duration_sec: 0.5`: 가까운 상태로 관측된 프레임 수가 0.5초 분량에 도달하면 후보 확정.
- `lost_tolerance_sec: 0.3`: 같은 ID가 잠시 누락되면 상태 보존. 누락 프레임은 최소 관측 시간에 포함하지 않음.

후보 확정 시간은 `confirmed_frame`, 구간 시작은 최초 close 프레임입니다. 영상에는 확정 시점부터 후보 상태가 보이고 클립에는 시작 전후 문맥도 들어갑니다. 별도 사람-지게차 ID 쌍마다 상태를 관리합니다. ID가 바뀌면 다른 쌍으로 취급합니다.

`start_frame`/`end_frame`은 0부터 시작하고 둘 다 구간에 포함합니다. `start_sec = start_frame / fps`, `end_sec = (end_frame + 1) / fps`이며 `end_sec`은 구간에 포함하지 않습니다. `observed_close_sec`은 실제로 가까운 상태가 관측된 프레임 수/FPS입니다. 추적 누락 유예 때문에 구간 길이와 다를 수 있습니다. 프레임 제한/영상 종료에서도 확정된 이벤트를 마감합니다.

기본 임계값은 튜닝 시작점입니다. 현장별 정답으로 검증된 값이 아닙니다. 정지한 두 객체도 가까우면 후보가 됩니다. 상대속도·접근 방향·time-to-collision은 구현하지 않았습니다.

## 5. Roboflow YOLO 데이터 준비와 학습

Warehouse Safety 계열의 **person/forklift 두 클래스, YOLO detection 형식**으로 내려받은 데이터를 로컬에 풀어 놓습니다. 접근 키나 다운로드 자동화는 필요하지 않습니다. 라벨은 `class_id x_center y_center width height` 형식의 정규화 좌표여야 합니다.

```text
data/warehouse/
├── data.yaml
├── train/images/   + train/labels/
├── valid/images/   + valid/labels/
└── test/images/    + test/labels/  (선택)
```

`configs/data.example.yaml`을 참고하되 **실제 라벨의 클래스 ID 순서를 유지**하세요. 이 프로젝트는 `path`를 `data.yaml` 기준으로, split 경로를 `path` 기준으로 해석합니다. 내보낸 YAML에 `../train/images`가 있는데 실제 폴더는 YAML 옆이라면 `train/images`로 수정하세요. 절대 경로도 가능합니다. 자동으로 다른 폴더를 추측하지 않습니다.

```bash
python train.py --data data/warehouse/data.yaml --check-only
python train.py --data data/warehouse/data.yaml --device 0 --epochs 80 --batch 8
```

`--check-only`는 두 클래스, split 이미지 폴더와 라벨 폴더 존재를 확인합니다. 각 라벨의 품질·누락·좌표 유효성에 대한 전체 검사는 아니며 Ultralytics의 학습 검사와 별도 수동 검토가 필요합니다. `configs/train.yaml`에서 seed, patience, batch 등을 조정합니다. VRAM 부족 시 batch를 줄이세요. 같은 영상의 인접 프레임이 train/val에 섞이지 않도록 촬영 단위로 분리하고, 증강 이미지는 원본과 같은 split에 두세요.

가중치는 기본적으로 `runs/train/warehouse_yolo26s/weights/best.pt`에 생성됩니다. 동일 이름의 기존 학습 결과가 있으면 Ultralytics가 새 이름을 사용할 수 있으므로 마지막 콘솔의 실제 경로를 확인하세요.

## 6. 클립 재추출

```bash
python extract_events.py --video runs/site01/annotated.mp4 --events runs/site01/events.csv --output runs/site01/clips_long --pre-sec 3 --post-sec 3
```

영상 길이를 넘는 구간은 잘라냅니다. 영상과 CSV는 반드시 같은 실행 결과를 사용하세요. 한 이벤트씩 읽어 메모리를 제한하므로 이벤트 수가 많으면 추출 시간이 늘어납니다. 중첩 이벤트는 별도 클립으로 남깁니다.

## 7. 테스트와 포트폴리오 완성 순서

```bash
python -m unittest discover -s tests -v
```

근접 규칙 테스트는 추가 패키지가 필요 없습니다. MP4 통합 테스트는 `requirements.txt` 설치 후 실행되며, **스크립트로 제공한 탐지 결과**로 이벤트·저장·클립을 확인합니다. 실제 모델 성능 테스트가 아닙니다. GitHub Actions도 이 범위만 검증합니다.

하루 MVP의 작업 순서:

1. 사용자 MP4로 pretrained 실행 → 사람 추적과 MP4 재생 확인.
2. YOLO 데이터 경로 및 클래스 점검 → GPU fine-tuning 시작.
3. `best.pt`로 별도 테스트 영상 분석 → 지게차 누락과 ID 변경 확인.
4. 검토한 후보 클립을 기준으로 임계값 조정 → 오탐·누락 사례 기록.
5. 공개 가능한 데모 장면, 실행 환경, 실제 측정치를 [포트폴리오 기록](docs/PORTFOLIO.md)에 추가.

정확도는 학습 validation 수치와 별도 영상에서의 후보 검토 결과를 구분해서 기록하세요. 구현 완료, 실행 검증, 현장 성능 검증은 각각 별개의 단계입니다.

## 제약과 재현성

- 가림, 원근, 카메라 이동, 탑승자, 박스 흔들림으로 오탐이 발생할 수 있습니다. 화면에서 가까워도 실제로는 다른 깊이에 있을 수 있습니다.
- ByteTrack ID는 영구 신원이 아닙니다. ID switch/재등장은 이벤트를 나눌 수 있습니다.
- OpenCV의 FPS로 시간을 계산합니다. 일정 프레임률(CFR) MP4를 사용하세요. 가변 프레임률 영상은 시간과 클립 경계가 부정확할 수 있습니다.
- 출력은 `mp4v`, 원래 해상도/FPS, **오디오 없음**입니다. 웹 재생용 H.264가 필요하면 별도 변환하세요. 홀수 해상도 영상은 짝수 해상도로 변환 후 입력하세요.
- 모델 confidence는 탐지 점수이며 사고 확률이 아닙니다. 후보 없음도 현장의 안전을 보장하지 않습니다.
- 실제 실행 환경에서 `python -m pip freeze > requirements.lock.txt`로 검증된 버전을 기록하세요. 현재 requirements는 호환 범위이며 lock 파일은 아닙니다.
- `.gitignore`는 데이터, `.pt`, MP4, 실행 결과를 제외합니다. GitHub에는 코드·문서부터 올리고, 공개 권한이 있는 데모만 별도 선별하세요. `.env`와 인증 정보도 커밋하지 않습니다.

## 근거 문서

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/): YOLO26s 모델 및 학습 API.
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/): `model.track`, ByteTrack, `persist=True`.
- [YOLO detection dataset format](https://docs.ultralytics.com/datasets/detect/): 데이터와 라벨 형식.
- [Ultralytics licensing](https://www.ultralytics.com/license): 의존 모델/소프트웨어 사용 조건.

데이터셋은 선택한 버전의 원본 페이지, 라이선스, attribution을 별도로 기록하세요. 이 저장소에는 데이터셋과 가중치를 포함하지 않습니다.

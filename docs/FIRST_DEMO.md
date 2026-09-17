# 첫 실제 영상 실행

이 단계에서는 사전학습 YOLO26s로 사람 탐지와 ByteTrack 추적을 확인한다. forklift 탐지와 근접 이벤트 검증에는 학습한 best.pt가 필요하다.

## 이 컴퓨터에서 시작

FactoryGuard 폴더에 `.venv`가 생성되어 있다. 아직 패키지는 설치되지 않았다. VS Code에서 **Terminal → New Terminal**을 누르고, 터미널 위치가 FactoryGuard인지 확인한 뒤 다음 한 줄을 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts/first_demo.py --setup
```

전용 가상환경에 패키지를 설치하고 GPU 연산을 확인한다. Pexels 샘플을 다운로드하고, 첫 300프레임을 최대 1280×720으로 축소한 뒤 YOLO26s와 ByteTrack을 실행한다. 최초 설치는 수 GB와 인터넷 접속이 필요하다. 모델 가중치도 첫 실행 때 다운로드된다.

다른 컴퓨터에서 새로 받은 저장소라면 먼저 Python 3.12로 `python -m venv .venv`를 실행한다. 전역 Python 패키지는 변경하지 않는다. GPU가 없다면 명령 뒤에 `--device cpu`를 추가한다.

## 어디서 무엇을 보는가

터미널은 `[1/4]` 설치 → `[2/4]` 환경 점검 → `[3/4]` 샘플 준비 → `[4/4]` 실제 추론 순서로 진행된다.

완료 시 `DONE. Play this video:` 아래에 **실제 파일의 전체 경로**가 출력된다.

| 파일 | 확인할 것 |
| --- | --- |
| `runs/first_demo_날짜_고유값/input_preview.mp4` | 분석한 원본 구간 |
| `runs/first_demo_날짜_고유값/result/annotated.mp4` | 플레이어로 열어 person 박스와 추적 ID 확인 |
| `runs/first_demo_날짜_고유값/result/tracks.csv` | 같은 사람의 ID가 유지되는지, 탐지가 누락되는지 |
| `runs/first_demo_날짜_고유값/result/summary.json` | status=complete, mode=detection_only, 처리 프레임 수 |
| `runs/first_demo_날짜_고유값/environment.json` | 실제 패키지 버전과 GPU |
| `runs/first_demo_날짜_고유값/SETUP_OR_DEMO_FAILED.txt` | 실패한 경우 원인. 완료 결과로 취급하지 않음 |

VS Code에서 MP4 재생이 안 되면 파일 탐색기에서 외부 영상 플레이어로 연다. 이 단계에서는 이벤트 CSV가 비어 있는 것이 정상이다. 지게차가 사람·차량 등으로 오인식되더라도 forklift로 이름을 바꾸지 않는다.

## 다시 실행하거나 다른 영상 사용

설치 완료 후에는 `--setup`을 빼면 된다. 실행마다 새 결과 폴더를 만든다.

```powershell
.\.venv\Scripts\python.exe scripts/first_demo.py
.\.venv\Scripts\python.exe scripts/first_demo.py --source "C:\videos\site.mp4" --frames 500
```

Pexels 다운로드가 실패하면 아래 원본 페이지의 Free download에서 MP4를 내려받고 `--source`로 지정한다. 브라우저에서 받은 실제 영상 FPS와 코드에서 읽은 FPS를 기준으로 시간을 해석한다.

## 출처와 환경 선택

- 샘플: Tiger Lily, [Man Walking towards the Forklift inside the Warehouse](https://www.pexels.com/video/man-walking-towards-the-forklift-inside-the-warehouse-4294434/), Pexels, video 4294434.
- 원본: [Pexels MP4](https://videos.pexels.com/video-files/4294434/4294434-uhd_3840_2160_25fps.mp4). 원본과 결과 영상은 Git에 포함하지 않는다.
- 패키지: [PyTorch 공식 설치 조합](https://pytorch.org/get-started/previous-versions/)의 torch 2.12.1 / torchvision 0.27.1, CUDA 13.0. [PyTorch의 Blackwell 안내](https://pytorch.org/blog/pytorch-2-12-release-blog/)와 설치된 RTX 5060 Ti, NVIDIA 드라이버 610.88을 기준으로 선택했다. 실제 실행 성공 여부는 GPU 연산 검사로 확인한다.

현재 문서는 실행 절차이며 실제 영상 분석이 완료됐다는 증거가 아니다. `summary.json`과 재생 확인 후 검증 기록을 갱신한다.

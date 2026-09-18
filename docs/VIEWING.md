# 결과물 보는 방법

작업 중 새 결과가 생기면 `완료한 것 → 여는 곳 → 확인할 내용 → 아직 검증하지 않은 것` 순서로 안내한다.

## 지금 볼 수 있는 것

| 결과물 | 여는 방법 | 확인할 내용 |
| --- | --- | --- |
| 프로젝트 설명 | VS Code에서 README.md 선택 후 Ctrl+Shift+V | 구조, 설치, 실행 순서, 한계 |
| 공개 저장소 | https://github.com/k-j-hero/FactoryGuard | 업로드된 코드와 변경 이력 |
| 자동 테스트 | GitHub의 Actions → 실행 제목 → tests | 초록 체크와 테스트 로그; 실제 탐지 성능은 별도 |
| 핵심 규칙 | factoryguard/proximity.py | 하단 중심 좌표, 정규화, 진입/종료 조건 |
| 실행 설정 | configs/default.yaml | 장치, 임계값, 최소 관측 시간, 클립 앞뒤 길이 |
| 학습 곡선 | runs/train/warehouse_yolo26s/results.png | loss와 validation 지표의 epoch별 변화 |
| 통합 학습 곡선 | runs/train/combined_yolo26s/results.png | 지상 시점 보강 후 loss와 validation 지표 변화 |
| 정제 ablation 곡선 | runs/train/curated_yolo26s/results.png | 큰 객체 중심 이미지 제외 후 80 epoch 학습 변화 |
| 세 설정 비교 이미지 | docs/assets/external-three-way.jpg | 기존, 안정형 추적, 데이터 필터 모델의 같은 프레임 비교 |
| 데이터 감사 보고서 | reports/combined-dataset-audit.json | split별 이미지·객체 수, 누락·좌표 오류·중복 여부 |
| 라벨 검토 contact sheet | runs/label-review/forklift-ground/ | 큰 박스와 명백한 중복 라벨을 표시한 19개 이미지 |
| 데이터 품질 결론 | docs/DATA_QUALITY.md | 선별 기준, 재학습 결과, precision/recall tradeoff |
| 모델 비교 원시 수치 | reports/model-evaluation.json | baseline/통합 모델의 지상 시점·원래 test 성능 비교 |
| 외부 영상 실행 진단 | reports/external-run-diagnostics.json | 박스 면적, 겹침 프레임, 고유·단기 ID 비교 |
| W&B 오프라인 run | runs/train/combined_yolo26s/wandb/offline-run-20260918_083036-wxl58ot9 | 80 epoch metrics, 설정, 곡선 이미지, 평가표. `wandb sync` 후 웹에서 확인 |
| W&B 자동 callback 증명 | runs/train/wandb/offline-run-20260918_083410-1fscae27 | 새 학습의 metrics·곡선·best.pt artifact 기록 여부 |
| 정제 ablation W&B run | runs/train/wandb/offline-run-20260918_091204-x6u8lzle | 큰 객체 중심 이미지 제외 학습의 80 epoch metrics와 곡선 |
| test 평가 그림 | runs/eval/warehouse_test/ | confusion matrix, PR/F1 곡선, 예측 샘플 |
| baseline 외부 영상 | runs/trained_demo_20260917/annotated.mp4 | 사람 탐지와 지게차 누락 실패 사례 |
| 통합 모델 외부 영상 | runs/combined_demo_20260917/annotated.mp4 | 지게차 존재 회복, 후보 2건, 과대 박스·오탐도 함께 확인 |
| 안정형 추적 외부 영상 | runs/combined_stable_20260918/annotated.mp4 | ID·겹침 감소와 남아 있는 과대 forklift 박스를 함께 확인 |
| 정제 모델 외부 영상 | runs/curated_demo_20260918/annotated.mp4 | 과대 박스 감소와 지게차 탐지 프레임 감소를 함께 확인 |
| 이벤트 파이프라인 증명 | runs/pipeline_proof_relaxed_20260917/ | annotated MP4, 이벤트 CSV, 실제 추출 clip |

## 첫 영상 실행 후 볼 것

아래 파일은 실제 `infer.py` 실행이 성공한 뒤 선택한 출력 폴더에 생긴다. 위 표의 두 실행 폴더에서 현재 결과를 바로 확인할 수 있다.

1. `summary.json`: `status`가 `complete`인지, `mode`가 `detection_only`인지 `proximity`인지 확인.
2. `annotated.mp4`: 파일 탐색기에서 영상 플레이어로 재생. 객체 박스, 추적 ID, 근접선, NP 값 확인. VS Code 안에서 재생이 안 되면 외부 플레이어 사용.
3. `events.csv`: VS Code 또는 Excel에서 열기. 시작/종료 시각과 person/forklift ID로 해당 구간 찾기.
4. `clips/`: `event_00001.mp4` 등의 짧은 영상을 재생. 후보 앞뒤 상황을 보고 화면상 close approach가 적절히 선택됐는지 검토.
5. `tracks.csv`: ID가 갑자기 바뀌거나 지게차가 누락되는 구간을 확인할 때 사용.

VS Code 탐색기에서 파일을 우클릭해 **Reveal in File Explorer**를 선택하면 Windows 영상 플레이어로 MP4를 열기 쉽다. 학습 곡선과 confusion matrix PNG는 VS Code에서 클릭하면 바로 보인다.

`FAILED.txt`가 있거나 `summary.json`이 없으면 부분 결과다. CSV에 이벤트가 없으면 모델의 forklift 클래스 유무부터 확인한다. 이벤트 없음은 현장이 안전하다는 의미가 아니다.

## 함께 작업하는 방식

VS Code와 Codex에서 같은 FactoryGuard 폴더를 사용한다. Codex의 프로젝트 추가에서 이 폴더를 선택하면 이후 작업을 한 프로젝트로 모을 수 있다. 결과를 설명할 때는 실제 생성된 파일 경로와 실행 폴더 이름을 전달한다.

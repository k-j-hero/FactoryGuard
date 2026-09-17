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

## 첫 영상 실행 후 볼 것

아래 파일은 실제 `infer.py` 실행이 성공한 뒤 선택한 출력 폴더에 생긴다. 아직 생성된 데모로 간주하지 않는다.

1. `summary.json`: `status`가 `complete`인지, `mode`가 `detection_only`인지 `proximity`인지 확인.
2. `annotated.mp4`: 파일 탐색기에서 영상 플레이어로 재생. 객체 박스, 추적 ID, 근접선, NP 값 확인. VS Code 안에서 재생이 안 되면 외부 플레이어 사용.
3. `events.csv`: VS Code 또는 Excel에서 열기. 시작/종료 시각과 person/forklift ID로 해당 구간 찾기.
4. `clips/`: `event_00001.mp4` 등의 짧은 영상을 재생. 후보 앞뒤 상황을 보고 화면상 close approach가 적절히 선택됐는지 검토.
5. `tracks.csv`: ID가 갑자기 바뀌거나 지게차가 누락되는 구간을 확인할 때 사용.

`FAILED.txt`가 있거나 `summary.json`이 없으면 부분 결과다. CSV에 이벤트가 없으면 모델의 forklift 클래스 유무부터 확인한다. 이벤트 없음은 현장이 안전하다는 의미가 아니다.

## 함께 작업하는 방식

VS Code와 Codex에서 같은 FactoryGuard 폴더를 사용한다. Codex의 프로젝트 추가에서 이 폴더를 선택하면 이후 작업을 한 프로젝트로 모을 수 있다. 결과를 설명할 때는 실제 생성된 파일 경로와 실행 폴더 이름을 전달한다.

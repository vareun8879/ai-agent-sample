# 생성 입력 비교 실험

서비스를 변경하거나 검색·임베딩 모델을 적재하지 않는다. 실행 중인 vLLM에
Generator.generate()로 순차 요청하며 재생성은 하지 않는다. 골든 측정과 동시에 실행하지 않는다.

기존 trace에는 history/mode가 없으므로 history=[]와 명시한 --mode로 비교한다.
현재 코드의 프롬프트·모델 설정을 사용하므로 과거 요청의 완전 재현이라고 해석하지 않는다.
각 문항은 지정한 trace 발생 순번(기본 첫 번째)을 사용한다.

변형은 control(원본), fact-first(사실 행 먼저), drop(지정 ID 제외),
paren-space(숫자와 숫자로 시작하는 괄호 사이 공백)이다. 모든 실행에 control이 포함된다.
출력에는 원문 근거와 표시 근거, 생성 원문, 항목, 원문 기준 검증 오류, 시간이 남는다.
인용 검증은 완화하지 않는다. 검증 통과를 정답으로 자동 채점하지 않는다.

## 클라우드 실행 예

TRACE는 261103_5 측정에 해당하는 실제 파일 경로로 지정한다.
아래 명령은 서브셸에서 환경변수를 읽으므로 SSH 세션 환경을 변경하지 않는다.

```bash
(
cd /root/ai-agent-v2
set -a
source deploy/gpu_server/.env
set +a
cd rag_service
PY=/root/ai-agent/app-env/bin/python
TRACE=/실제/261103_5/trace.jsonl
OUT="../reports/replay_$(date '+%Y%m%d_%H%M%S')"

"$PY" -c "import ast; from pathlib import Path; ast.parse(Path('evaluation/replay_generation.py').read_text(encoding='utf-8')); print('syntax OK')" &&
"$PY" -m evaluation.replay_generation "$TRACE" --cases b01 --variants fact-first drop --drop-ids E2 --mode voice --output "$OUT/b01.jsonl" --dry-run &&
"$PY" -m evaluation.replay_generation "$TRACE" --cases m17 m32 --variants paren-space --mode voice --output "$OUT/m17_m32.jsonl" --dry-run &&
"$PY" -m evaluation.replay_generation "$TRACE" --cases b01 --variants fact-first drop --drop-ids E2 --mode voice --output "$OUT/b01.jsonl" &&
"$PY" -m evaluation.replay_generation "$TRACE" --cases m17 m32 --variants paren-space --mode voice --output "$OUT/m17_m32.jsonl"
)
```
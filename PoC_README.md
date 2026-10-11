# PoC: smolagents Agent 호출 이력 대시보드

smolagents agent가 **LLM에 무엇을 물었고, 어떤 코드나 tool을 어떤 파라미터로 실행했는지** step 단위로 뽑아서 웹에서 보여주는 PoC입니다. browser-use용으로 만든 Agent Observatory를 같은 디자인으로 옮겼습니다.

- **왼쪽:** Archify로 그린 smolagents agent loop 구조도. LLM 호출과 tool 실행이 일어나는 위치가 표시돼 있고, 노드의 소스 링크를 누르면 해당 코드가 열립니다.
- **오른쪽:** 실행 이력. step마다 Plan → LLM 호출(생성한 코드나 tool call) → tool 실행(파라미터, 출력/에러, 시간) → Observation 순서로 쌓입니다.

`CodeAgent`와 `ToolCallingAgent` 둘 다 지원합니다. browser-use와 같은 방법으로, **agent loop 안의 LLM 호출 지점과 tool 실행 지점에 로그를 심고** 대시보드가 그 로그를 읽습니다.

## 1. 실행 방법

browser-use 대시보드와 같은 방법으로 실행합니다.

준비물: Python 3.10+, [uv](https://docs.astral.sh/uv/), Anthropic API 키

```bash
git clone https://github.com/NiceGuy1313/smolagents.git
cd smolagents

# 1. API 키 설정 (처음 한 번만)
cp .env.example .env
# .env를 열어서 ANTHROPIC_API_KEY= 뒤에 실제 키를 넣으세요. .env는 git에 안 올라갑니다.

# 2. 의존성 설치
uv sync

# 3. 서버 실행
uv run uvicorn examples.agent_dashboard.app:app --port 8000
```

브라우저에서 http://127.0.0.1:8000 을 엽니다.

- `uv sync`는 `pyproject.toml`의 기본 의존성 그룹 `dashboard`(LiteLLM, 웹 검색/페이지 읽기 tool, FastAPI, uvicorn, pytest)까지 한 번에 설치합니다.
- `.env`는 서버가 시작할 때만 읽습니다. 키를 바꿨으면 서버를 재시작하세요.
- browser-use 대시보드와 동시에 띄울 때는 포트를 나눠 쓰세요(예: browser-use `8000`, smolagents `8001`).

## 2. 사용 방법

1. 오른쪽 위 입력칸에 **task**, **최대 step 수**, **agent 종류**(`CodeAgent` / `ToolCallingAgent`), **모델**(기본 `claude-sonnet-5`)을 넣고 **Run**을 누릅니다.
   - 모델 칸에는 browser-use처럼 Claude 모델 이름을 그대로 쓰면 됩니다. 다른 제공자는 LiteLLM 형식(`openai/gpt-4o` 등)으로 쓰고, 해당 키를 `.env`에 넣으세요.
2. 오른쪽 **Call history**에 step별 이력이 실시간으로 쌓입니다.
   - `Plan`: planning step의 계획 (planning을 켰을 때만)
   - `LLM` 행: LLM 호출 1회. `CodeAgent`면 생각과 생성한 코드, `ToolCallingAgent`면 고른 tool call 목록. 입력→출력 토큰 수와 소요 시간
   - `TOOL` 행: 실제 실행된 tool, 파라미터, 출력. 실패하면 빨간색으로 에러 표시
   - `STEP` 행: tool 밖에서 난 에러 (코드 파싱 실패, 코드 실행 에러 등)
   - `Observation`: 다음 step에 LLM이 받는 관찰 결과 (펼쳐서 보기)
   - `FINAL ANSWER` 배지: 최종 답을 낸 step
3. 상단 카운터: steps / LLM 호출 수 / tool 실행 수 / 토큰 합계 / 에러 수
4. **Raw log** 탭에서 가공하지 않은 이벤트 전체를 볼 수 있습니다.
5. **Export JSON**으로 전체 이력을 받습니다. 실행이 끝날 때마다 `examples/agent_dashboard/runs/<run_id>.json`에도 자동 저장됩니다(git에는 안 올라감).

실행 도중이나 끝난 뒤에 페이지를 새로 열어도 이력이 처음부터 다시 표시됩니다. 한 번에 하나의 실행만 할 수 있습니다.

## 3. 이력 JSON 형식

```jsonc
{
  "id": "…", "task": "…", "model": "claude-sonnet-5", "agent_type": "code", "status": "finished",
  "events": [
    { "seq": 1, "kind": "llm_query",    "step": 1, "data": { "model": "…", "messages": 4, "purpose": "action" } },
    { "seq": 2, "kind": "llm_response", "step": 1, "data": { "purpose": "action", "actions": [{ "python_interpreter": "results = web_search(query=…)" }], "input_tokens": 1830, "output_tokens": 95, "duration_seconds": 2.1 } },
    { "seq": 3, "kind": "tool_start",   "step": 1, "data": { "action": "web_search", "params": "{'query': '…'}" } },
    { "seq": 4, "kind": "tool_end",     "step": 1, "data": { "action": "web_search", "result_type": "str", "duration_seconds": 0.8 } },
    { "seq": 5, "kind": "step",         "step": 1, "data": { "type": "action", "model_output": "Thought: …", "code": "…", "observations": "…", "error": null, "is_final_answer": false } }
  ]
}
```

`kind`는 `run`, `llm_query`, `llm_response`, `llm_error`, `tool_start`, `tool_end`, `tool_error`, `step` 중 하나입니다. browser-use 버전과 같은 형식이라 두 agent의 이력을 같은 도구로 분석할 수 있습니다.

## 4. 동작 원리와 코드 위치

browser-use와 같은 방식으로 smolagents 코어에 경계 로그(DEBUG)를 심었습니다. 태그와 형식도 browser-use와 같습니다.

| 로그 | 심은 위치 | 남는 정보 |
|---|---|---|
| `[LLM_QUERY]` | `agents.py`: 모델을 부르기 직전. planning(`_generate_planning_step`), action(`CodeAgent._step_stream`, `ToolCallingAgent._step_stream`), max steps 후 강제 답변(`provide_final_answer`) | step, 모델, 메시지 수, 용도 |
| `[LLM_RESPONSE]` | 응답을 액션으로 파싱한 직후. CodeAgent는 코드(`python_interpreter`), ToolCallingAgent는 tool call 목록 | step, 용도, 액션, 토큰 수 |
| `[LLM_ERROR]` | 모델 호출이 실패했을 때 | step, 용도, 에러 종류 |
| `[TOOL_EXECUTION_START/END/ERROR]` | `tools.py`: `Tool.__call__`에서 `forward` 앞뒤. CodeAgent(코드 안에서 호출)와 ToolCallingAgent(`execute_tool_call`) 모두 여기를 지납니다 | tool 이름, 파라미터, 결과 타입/에러 종류, 소요 시간 |

LLM 쪽 로그는 `MultiStepAgent`의 `_log_llm_query`, `_log_llm_response`, `_log_llm_error` 세 메서드로 모아서 모든 호출 지점이 같은 형식을 씁니다. step 결과(코드, observation, 에러, 최종 답 여부, plan)는 browser-use가 `on_step_end`를 쓴 것처럼 `step_callbacks`로 받습니다.

| 파일 | 역할 |
|---|---|
| `src/smolagents/agents.py` | LLM 경계 로그 (`[LLM_QUERY]`, `[LLM_RESPONSE]`, `[LLM_ERROR]`) |
| `src/smolagents/tools.py` | tool 경계 로그 (`[TOOL_EXECUTION_*]`) |
| `examples/agent_dashboard/app.py` | FastAPI 서버. 로그 → 이력 변환(`DashboardLogHandler`), agent 생성(`build_agent`), SSE, export |
| `examples/agent_dashboard/static/index.html` | 화면 |
| `examples/agent_dashboard/agent-architecture.json` | 구조도 원본 spec (Archify 형식) |
| `examples/agent_dashboard/static/agent-architecture.html` | Archify 3.0.1로 생성한 구조도 (생성 결과) |
| `tests/test_boundary_logging.py` | 코어 경계 로그 테스트 |
| `examples/agent_dashboard/test_app.py` | 로그 → 이력 변환 테스트 |

테스트(네트워크/키 불필요): `uv run pytest tests/test_boundary_logging.py examples/agent_dashboard/test_app.py`

## 5. 아직 안 되는 것 / 다음 작업

- **구조도를 고치려면 Archify가 필요합니다.** spec(`agent-architecture.json`)을 수정한 뒤 Archify로 검증하고 HTML을 다시 생성하세요. 명령은 `examples/agent_dashboard/README.md`에 있습니다.
- **원격 실행기(E2B, Docker, Modal)에서 도는 tool은 다른 프로세스에서 로그를 남겨서 안 잡힙니다.** 로컬 실행기를 쓰세요.
- **managed agent(하위 agent)도 같은 로거에 로그를 남겨서**, 부모 실행 이력에 하위 agent의 step 번호로 섞여 들어갑니다.
- **LLM 호출이 실패하면 smolagents는 다음 step으로 넘어가지 않고 실행을 끝냅니다.** 이력에는 `llm_error`와 `Run failed`가 남습니다.

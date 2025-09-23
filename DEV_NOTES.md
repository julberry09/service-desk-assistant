# 🛠️ DEV_NOTES: 개발 과정에서의 어려움과 해결 가이드

본 문서는 **Service Desk Assistant** 프로젝트를 개발하며 겪었던 시행착오와 그 해결 과정을 정리한 가이드입니다.  
향후 비슷한 구조/환경에서 개발할 때 참고할 수 있도록 작성되었습니다.

---

## 1. `pyproject.toml` vs `requirements.txt`

### 📌 차이점
- **requirements.txt**
  - `pip install -r requirements.txt` 로 단순 의존성 설치
  - 빠르고 직관적이지만, 버전 충돌/메타데이터 관리가 부족
- **pyproject.toml**
  - [PEP 621](https://peps.python.org/pep-0621/) 기반의 현대적 빌드 표준
  - 의존성 + 빌드 시스템 + 프로젝트 메타데이터를 통합 관리
  - Poetry, Hatch 같은 도구와 호환성 높음
  - `pip install -e .` 로 개발 모드 설치 가능

### 📌 교훈
- 초반에는 `requirements.txt`만 사용했으나, 나중에 **pyproject.toml 권장** 흐름을 따라갔음.
- 그러나 협업 시 일부 환경에서 `requirements.txt`를 요구하기도 하므로 **둘 다 유지**하는 것이 안전.
- → 최종적으로는 `pyproject.toml`을 **메인**, `requirements.txt`는 **보조** 용도로 관리.

### 📦 requirements.txt의 필요성 (공유/협업 관점)
- **협업/공유**: 팀원이 `pip install -r requirements.txt` 한 줄로 동일 환경을 바로 재현 가능  
- **배포 환경**: Dockerfile, CI/CD, 클라우드 배포에서 가장 흔히 사용되는 방식  
  ```dockerfile
  RUN pip install -r requirements.txt
  ```
- **호환성**: Poetry/Hatch 같은 최신 툴을 쓰지 않는 환경에서도 문제없이 설치 가능  
- **버전 고정**: `pip freeze > requirements.txt` 로 실제 설치된 버전을 기록해 두면, 재현성 있는 실행 환경 확보 가능

📌 따라서,  
- **pyproject.toml** = 현대적 관리, 패키징/배포 표준  
- **requirements.txt** = 여전히 협업·배포·호환성을 위해 필요  

---

## 2. 프로그램 구조 (표준구조 vs 서비스-API 분리형)

### 1) 파이썬 표준구조
```bash
src/service_desk_assistant/
├── ui.py
└── core.py
```
- 교과서, 튜토리얼에서 자주 등장
- 작은 규모 프로젝트에는 적합
- UI보다는 백엔드 스크립트/모듈 중심

### 2) 서비스-API 분리형 아키텍처
```bash
service-desk-assistant/
├── platform_assistant/  # UI
└── platform_service/    # API + Core
```
- 마이크로서비스 유사 구조
- UI(Core 직접 호출 ❌ → API 통신만 ✅)
- Streamlit ↔ FastAPI ↔ LLM/VectorStore
- 확장성과 유지보수성에 강점

### 📌 시행착오
- Jupyter Notebook 환경에서는 **서비스-API 분리형 구조만 정상 작동**  
  (버튼 클릭으로 Streamlit 실행하려면 분리형 필요)
- 초기에 표준구조로 시도 → **UI 실행 실패** → 구조를 분리형으로 리팩터링

---

## 3. Jupyter vs Local 개발 환경

### 📓 Jupyter Notebook (실습환경)
- 가상환경 별도 생성 ❌ (이미 환경 제공됨)
- 설치:  
  ```bash
  pip install -e ".[test]"
  ```
  → 단, **실습환경에서는 필수 아님**. 그냥 `python -m ...` 방식으로 직접 실행 가능.
- 실행:
  ```bash
  python -m platform_service.api --port 8001
  streamlit run platform_assistant/ui.py
  ```

### 💻 Local 개발환경
- 가상환경 직접 생성:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # (윈도우는 .venv\Scripts\activate)
  pip install -e ".[test]"
  ```
- 실행:
  ```bash
  python -m platform_service.api --port 8001
  streamlit run platform_assistant/ui.py
  ```

### 📦 패키징의 의미
- `pip install -e .` 는 단순 실행을 위한 명령어가 아니라,  
  **프로젝트를 패키지처럼 site-packages에 등록**하는 과정임.
- 이를 통해:
  - `import platform_service` 형태로 어디서든 사용 가능
  - 협업자/다른 서버에서 쉽게 설치 가능 (`pip install myproject`)
  - wheel(.whl) 파일이나 PyPI 업로드로 **배포/공유** 준비 완료
- 따라서,
  - **실습환경(Jupyter)** → 패키징 생략해도 무방 (직접 실행 권장)  
  - **Local/배포 환경** → 패키징 필수 (재사용성, 이식성 확보)

### 📌 교훈
- Jupyter 환경 = **제한적**, 여러 명이 동시에 `.venv` 만들면 충돌/속도 저하로 만들지 않는걸 추천
- Local 환경 = **자유도 높음**, 가상환경 + 패키징 권장
- 따라서, **실습/테스트는 Jupyter**, **실제 개발/공유는 Local 패키징**으로 진행하는 것이 최적

---

## 4. Pytest 전략

- 테스트 폴더: `tests/`
- 예시: `tests/test_api.py`

### 전략
1. **단위 테스트**: DB 초기화, 헬스체크 API
2. **통합 테스트**: `/chat` 요청 → 파이프라인 → 답변/DB 저장 여부 검증
3. **예외 상황**: API 서버 다운, 잘못된 입력 처리
4. **자동화**: CI 환경에서 `pytest -q` 실행

→ 작은 모듈 단위부터 API 전체까지 **점진적 보장**

---

## 5. DB 연동

- **SQLite 권장** → 별도 설치 불필요, 파이썬 표준 라이브러리로 사용 가능
- `platform_service/db/history.py`에서 구현
- 채팅 기록 저장/조회 기능 추가 → **대화 연속성 확보**
- 로깅/DB 결합으로 **문제 추적 및 복기 용이**

📌 교훈:  
외부 DB(Mongo, Postgres)보다, **PoC/내부 프로젝트에는 SQLite가 간단하고 충분**

---

## 6. 프로그램 적용 체크리스트 ✅

### 프롬프트 최적화
- 역할 부여 (Role Prompting) + Chain-of-Thought - checklist: 1
- Few-shot Prompting - checklist: 2
- 재사용성 고려 - checklist: 3

### 멀티에이전트 / LangGraph
- Supervisor → Planner/Researcher/Coder - checklist: 4
- ReAct 스타일 Tool Agent - checklist: 5
- 멀티턴 대화 (memory) - checklist: 6

### RAG (Retrieval Augmented Generation)
- 데이터 수집/전처리 - checklist: 7
- FAISS 기반 Vector DB - checklist: 8
- FAQ/문서 검색 + 답변 강화 - checklist: 9

### 서비스 개발/배포
- UI: Streamlit - checklist: 10
- API: FastAPI - checklist: 11
- 배포: Docker, docker-compose - checklist: 12

---

# 📌 결론

이 프로젝트는 **단순 헬프데스크 챗봇을 넘어, API/DB/RAG/멀티에이전트까지 통합**한 구조 실험이었음.  
개발 과정에서:
- 구조 전환 (표준형 → 분리형)
- Jupyter/Local 차이
- 의존성 관리 (pyproject.toml vs requirements.txt)
- DB 연동
- pytest 전략 수립
등을 겪으며 실무적으로 성장할 수 있었음.

앞으로 유사 프로젝트 진행 시 본 문서를 참고하여 시행착오를 줄일 수 있을 것임.

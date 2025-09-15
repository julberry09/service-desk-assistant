# 🌞 사내 헬프데스크 챗봇 (Quick Start)

LangChain, FastAPI, Streamlit을 활용한 **사내 헬프데스크 챗봇**입니다.  
RAG(Retrieval-Augmented Generation) 기반 답변과 ID 발급, 비밀번호 초기화, 담당자 조회 같은 자동화 툴을 제공합니다.

---

## 🚀 빠른 시작 (Quick Start)

### 1. 환경 변수 설정
`.env` 파일을 생성하고 Azure OpenAI 환경 변수를 입력하세요.

```bash
# .env 예시
AOAI_ENDPOINT=https://your-aoai-endpoint.openai.azure.com/
AOAI_API_KEY=your-aoai-api-key
AOAI_API_VERSION=2024-10-21

# Deployments
AOAI_DEPLOY_GPT4O_MINI=gpt-4o-mini
AOAI_DEPLOY_GPT4O=gpt-4o
AOAI_DEPLOY_EMBED_3_SMALL=text-embedding-3-small

# API 설정
API_SERVER_HOST=0.0.0.0
API_CLIENT_HOST=localhost
API_PORT=8001
```

---

### 2. 가상환경 & 의존성 설치
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows는 .venv\Scripts\activate)

# 필수 라이브러리 설치
pip install -e .

# 테스트 포함 개발환경 설치
pip install -e ".[test]"
```

---

### 3. 실행
#### 백엔드 (FastAPI)
```bash
python -m platform_service.api --port 8001
# or
uvicorn platform_service.api:api --reload --port 8001
```

#### 프론트엔드 (Streamlit)
```bash
streamlit run platform_assistant/ui.py --server.port 8507
```

👉 브라우저에서 `http://localhost:8507` 접속

---

### 4. Docker 실행
```bash
docker compose up --build -d
# UI: http://localhost:8507
# API: http://localhost:8001/health
```

---

## 📂 프로젝트 구조 (요약)
```
service-desk-assistant/
├── platform_assistant/   # 🌞 Streamlit UI
│   └── ui.py
├── platform_service/     # ⚙️ FastAPI & 워크플로우
│   ├── api.py
│   ├── core.py
│   ├── constants.py
│   ├── logging_config.py
│   └── db/
│       └── history.py
├── kb_default/           # 📚 기본 FAQ
├── kb_data/              # 🗂️ 업로드 문서
├── indexes/              # 🗂️ FAISS 인덱스
├── logs/                 # 🪵 로그
├── docker/               # 🐳 Dockerfile
└── tests/                # 🧪 테스트 코드
```

---

## ✅ 기능 요약
- **대화형 헬프데스크**: 세션별 대화 관리, 히스토리 조회
- **RAG 응답**: 문서 기반 검색+생성
- **자동화 툴**: ID 발급, 비밀번호 초기화, 담당자 조회
- **확장성**: 문서 업로드 및 인덱스 재생성 지원

---

## 🐛 트러블슈팅 (자주 발생하는 오류)
- `address already in use`: 기존 서버 프로세스 종료 후 재실행 → `pkill -f platform_service.api`
- `ModuleNotFoundError`: `pip install -e .` 재실행 필요
- `.venv/bin/activate: not found`: 가상환경 미생성 → `python -m venv .venv`

---

## 📜 라이선스
이 프로젝트는 내부 사용 목적으로 작성되었습니다.

# platform_service/api.py

import os
import uvicorn
import time as _time
import contextlib
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, Body, Request, UploadFile, File
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from platform_service import pipeline, build_or_load_vectorstore, AZURE_AVAILABLE, constants
from platform_service.db.history import init_db, save_message, load_messages 

logger = logging.getLogger(__name__)


# =============================================================
# 1. FastAPI 앱 설정 - server/config.py (Config & 초기화)
# - FastAPI 앱 생성 및 lifespan 이벤트 정의
# =============================================================
# 서비스 개발 및 패키징 - FastAPI를 활용하여 백엔드 API 구성 [checklist: 11]
# 애플리케이션 시작/종료 시 이벤트 처리
@contextlib.asynccontextmanager
async def lifespan(api: FastAPI):
    logger.info("Application starting up...")
    try:
        init_db()  # ✅ DB 초기화
        # 서버 시작 시 기본 KB 문서를 로드하여 벡터스토어 생성
        if AZURE_AVAILABLE:
            build_or_load_vectorstore()
            logger.info("기본 KB 문서를 벡터스토어에 로드 완료")
        else:
            logger.warning("Azure OpenAI 설정이 없어 기본 KB 로드는 생략")
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {e}")
    yield
    logger.info("Application shutting down.")

api = FastAPI(
    title="Service Desk Assistant API",
    version="0.2.0",
    lifespan=lifespan
)


# =============================================================
# 2. server/middleware.py (Middleware 정의)
# - 요청/응답/에러 로깅
# =============================================================
class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = _time.time()
        logger.info("api_request", extra={"extra_data": {"path": request.url.path}})
        try:
            response = await call_next(request)
            dur = round((_time.time() - start) * 1000)
            logger.info(
                "api_response",
                extra={"extra_data": {"status": response.status_code, "ms": dur}}
            )
            return response
        except Exception as e:
            logger.exception("api_error", extra={"extra_data": {"error": str(e)}})
            raise

api.add_middleware(AuditMiddleware)


# =============================================================
# 3. API 엔드포인트
# server/routers/chat.py (API 엔드포인트)
# =============================================================
class ChatIn(BaseModel):
    message: str
    session_id: str

class ChatOut(BaseModel):
    reply: str
    intent: str
    sources: list[dict] = Field(default_factory=list)

@api.get("/health")
def health():
    return {"ok": True}

@api.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn = Body(...)):
    out = pipeline(payload.message, payload.session_id)
    save_message(payload.session_id, "user", payload.message)
    save_message(payload.session_id, "assistant", out.get("reply", ""))
    return ChatOut(
        reply=out.get("reply", ""),
        intent=out.get("intent", ""),
        sources=out.get("sources", [])
    )

@api.get("/history")
def get_history(session_id: str, limit: int = 20):
    msgs = load_messages(session_id, limit)
    return {"ok": True, "messages": msgs}

@api.post("/sync")
def sync_index():
    if not AZURE_AVAILABLE:
        return {"ok": False, "message": "Azure 설정 없음"}
    try:
        build_or_load_vectorstore()
        return {"ok": True, "message": "인덱스 재생성 완료"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@api.get("/status")
def status():
    return {
        "ok": True,
        "azure_available": AZURE_AVAILABLE,
    }

@api.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """업로드된 문서를 kb_data 디렉토리에 저장"""
    saved = []
    constants.KB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        path = constants.KB_DATA_DIR / f.filename
        with open(path, "wb") as w:
            w.write(await f.read())
        saved.append(f.filename)
    return {"ok": True, "saved": saved}

# =============================================================
# 4. 실행 엔트리포인트 - main.py 
# =============================================================
if __name__ == "__main__":
    import argparse
    default_host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    default_port = int(os.getenv("API_PORT", 8000))

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", default=default_port, type=int)
    args = parser.parse_args()

    uvicorn.run(api, host=args.host, port=args.port)

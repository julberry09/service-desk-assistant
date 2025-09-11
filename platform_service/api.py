# api.py

import os
import uvicorn
import time as _time
from typing import List, Dict, Any
import contextlib # 💡 추가

from fastapi import FastAPI, Body, Request
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from helpdesk_bot.core import pipeline, logger

# =============================================================
# 1. FastAPI 앱 설정
# =============================================================
# 서비스 개발 및 패키징 - FastAPI를 활용하여 백엔드 API 구성 [checklist: 11] 
# 💡 추가: 애플리케이션 시작/종료 시 이벤트 처리
@contextlib.asynccontextmanager
async def lifespan(api: FastAPI):
    logger.info("Application starting up...")
    try:
        # Okt 초기화 로직을 제거
        # _ = get_okt()
        # logger.info("Okt has been initialized successfully.")
        pass # 빈 블록 유지
    except Exception as e:
        logger.error(f"Failed to initialize Okt/JVM: {e}")
    yield
    logger.info("Application shutting down.")

api = FastAPI(title="Helpdesk RAG API", version="0.1.0", lifespan=lifespan) # 💡 lifespan 추가

class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = _time.time()
        logger.info("api_request", extra={"extra_data": {"path": request.url.path}})
        try:
            response = await call_next(request)
            dur = round((_time.time() - start)*1000)
            logger.info("api_response", extra={"extra_data": {"status": response.status_code, "ms": dur}})
            return response
        except Exception as e:
            logger.exception("api_error", extra={"extra_data": {"error": str(e)}})
            raise

api.add_middleware(AuditMiddleware)

# =============================================================
# 2. API 엔드포인트
# =============================================================
class ChatIn(BaseModel): message: str; session_id: str

class ChatOut(BaseModel): reply: str; intent: str; sources: list[dict] = Field(default_factory=list)

@api.get("/health")
def health(): return {"ok":True}

@api.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn = Body(...)):
    out = pipeline(payload.message, payload.session_id)
    return ChatOut(reply=out.get("reply",""), intent=out.get("intent",""), sources=out.get("sources", []))

# =============================================================
# 3. 엔트리포인트
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
# tests/conftest.py
import pytest
import os
from platform_service.db import history

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """테스트 시작 전에 SQLite DB(chat_history) 테이블을 생성"""
    print(f"[pytest] DB 초기화 경로: {history.DB_PATH}")
    history.init_db()
    yield
    # 테스트 끝나고 DB 파일 삭제
    if os.path.exists(history.DB_PATH):
        os.remove(history.DB_PATH)
        print(f"[pytest] 테스트 완료 후 DB 삭제: {history.DB_PATH}")

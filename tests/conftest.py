# tests/conftest.py
import pytest
from platform_service.db import history

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """테스트 시작 전에 SQLite DB(chat_history) 테이블을 생성"""
    print(f"[pytest] DB 초기화 경로: {history.DB_PATH}")
    history.init_db()
    yield
    # 정리 단계에서는 DB를 지우지 말고 그대로 둡니다.

# tests/__init__.py

import pytest
from platform_service.db import history

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """테스트 세션 시작 전에 SQLite DB 테이블 생성"""
    print("\n[pytest] history DB 초기화 시작")
    history.init_db()
    yield
    print("\n[pytest] history DB 테스트 종료")
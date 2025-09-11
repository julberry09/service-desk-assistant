# tests/test_api.py

import sys
import os
import json
import re
import pytest
from fastapi.testclient import TestClient
import uuid

from helpdesk_bot.api import api
from helpdesk_bot.core import AZURE_AVAILABLE, build_or_load_vectorstore, get_okt # get_okt 임포트 추가

# =============================================================
# Fixtures & Helpers
# =============================================================
# @pytest.fixture(scope="session", autouse=True)
# def okt_initialization():
#     """
#     테스트 세션 시작 시 Okt 인스턴스를 한 번만 초기화합니다.
#     """
#     print("\npytest-session: Okt 객체 초기화를 시작합니다.")
#     _ = get_okt()
#     print("pytest-session: Okt 객체가 성공적으로 초기화되었습니다.")
#     yield
#     print("\npytest-session: Okt 객체 테스트 세션이 종료됩니다.")
@pytest.fixture(scope="session", autouse=True)
def _disable_jvm_for_tests():
    # pytest 동안 JVM 사용 금지 (더미 OKT 사용)
    os.environ["TEST_DISABLE_JVM"] = "1"
    # 최초 접근에서 더미 OKT가 초기화되는지 확인(부수효과: lazy init 트리거)
    _ = get_okt()
    yield
    # 테스트 종료 시 원복(굳이 없어도 무방)
    os.environ.pop("TEST_DISABLE_JVM", None)

@pytest.fixture(scope="module")
def client():
    """
    FastAPI 테스트 클라이언트를 반환합니다.
    """
    return TestClient(api)

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """
    테스트에 필요한 환경을 설정합니다.
    (예: 테스트용 벡터스토어 재빌드)
    """
    if AZURE_AVAILABLE:
        print("\nAZURE_AVAILABLE: 테스트를 위해 벡터스토어를 재빌드합니다.")
        try:
            build_or_load_vectorstore()
        except RuntimeError as e:
            pytest.skip(f"Azure OpenAI 설정이 없어 테스트를 건너뜁니다: {e}")
    else:
        print("\nAZURE_AVAILABLE: False, 폴백 모드로 테스트합니다.")
    yield

def run_api_test(client, endpoint, payload, expected_status, expected_keys=None, additional_assertions=None):
    """
    API 테스트를 위한 유틸리티 함수
    
    Args:
        client: TestClient 인스턴스
        endpoint: 테스트할 API 엔드포인트
        payload: 요청에 사용할 JSON 페이로드
        expected_status: 예상하는 HTTP 상태 코드
        expected_keys: 응답 JSON에 포함되어야 하는 키 목록 (Optional)
        additional_assertions: 추가적인 커스텀 검증 함수 (Optional)
    """
    # payload에 session_id가 없으면 임의의 UUID를 추가
    if "session_id" not in payload:
        payload["session_id"] = str(uuid.uuid4())
        
    # 1. API 요청 실행
    response = client.post(endpoint, json=payload)
    
    # 2. 상태 코드 검증
    assert response.status_code == expected_status
    
    # 3. 응답 키 검증 (선택적)
    data = response.json()
    if expected_keys:
        for key in expected_keys:
            assert key in data, f"응답에 {key} 키가 누락되었습니다."

    # 4. 추가 검증이 있다면 실행
    if additional_assertions:
        additional_assertions(data, response)

    return data

# =============================================================
# 1. API 기본 동작 테스트
# =============================================================
def test_health_ok(client):
    """/health 엔드포인트가 정상적으로 200 OK와 {"ok": True}를 반환하는지 테스트합니다."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_chat_bad_request(client):
    """'message' 필드가 없는 잘못된 요청에 대해 422 Unprocessable Entity를 반환하는지 테스트합니다."""
    r = client.post("/chat", json={"session_id": "bad_request"})
    assert r.status_code == 422
    assert "message" in r.json()["detail"][0]["loc"]

# =============================================================
# 2. 통합 테스트: 전체 파이프라인 동작 검증
# =============================================================
def test_rag_flow_integration_with_faq(client):
    """
    FAQ에 대한 질문이 RAG 파이프라인을 통해 올바르게 답변되는지 테스트합니다.
    """
    def assert_faq_response(data, response):
        if AZURE_AVAILABLE:
            # Azure가 연결된 정상 모드에서는 FAQ를 통해 답변을 찾고 소스도 제공해야 함
            assert data["intent"] == "faq" or data["intent"] == "general_qa"
            assert "자주 묻는 질문" in data.get("reply", "") or len(data.get("sources", [])) > 0
        else:
            # 폴백 모드에서는 FAQ 검색 후 답변이 반환되어야 함
            assert data["intent"] == "faq"
            assert "문의하신 내용에 대한 답변입니다." in data["reply"]

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "식당 메뉴는 어디서 확인?", "session_id": str(uuid.uuid4())},
        expected_status=200,
        expected_keys=["reply", "intent", "sources"],
        additional_assertions=assert_faq_response
    )

def test_tool_owner_lookup_integration(client):
    """
    '담당자 조회' 도구 호출 흐름을 통합 테스트합니다.
    """
    def assert_owner_lookup_response(data, response):
        if AZURE_AVAILABLE:
            assert data["intent"] == "agent_action"
            assert "홍길동" in data["reply"]
        else:
            # 폴백 모드에서는 도구 함수를 직접 호출하므로 그에 맞춰 검증
            assert data["intent"] == "direct_tool"
            assert "담당자" in data["reply"]
    
    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "인사시스템 사용자관리 담당자 알려줘", "session_id": str(uuid.uuid4())},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_owner_lookup_response
    )

def test_tool_reset_password_integration(client):
    """
    '비밀번호 초기화' 도구 호출 흐름을 통합 테스트합니다.
    """
    def assert_reset_pw_response(data, response):
        if AZURE_AVAILABLE:
            assert data["intent"] == "agent_action"
            assert "비밀번호 초기화" in data["reply"]
        else:
            # 폴백 모드에서는 도구 함수를 직접 호출하므로 그에 맞춰 검증
            assert data["intent"] == "direct_tool"
            assert "비밀번호 초기화 안내" in data["reply"]

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "비밀번호 초기화", "session_id": str(uuid.uuid4())},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_reset_pw_response
    )

def test_tool_request_id_integration(client):
    """
    '아이디 발급 절차' 도구 호출 흐름을 통합 테스트합니다.
    """
    def assert_request_id_response(data, response):
        if AZURE_AVAILABLE:
            # Azure가 연결되어 있을 때의 테스트 로직
            assert data["intent"] == "direct_tool"
            assert "HR 포털" in data["reply"]
            assert "계정 신청" in data["reply"]
            assert len(data.get("sources", [])) == 0
        else:
            # 폴백 모드에서는 도구 함수를 직접 호출하므로 그에 맞춰 검증
            assert data["intent"] == "direct_tool"
            # 수정: node_finalize의 변경된 답변 내용에 맞춰 검증
            assert "ID 발급 신청 절차 안내" in data["reply"]
            assert "HR 포털 접속" in data["reply"]
            assert len(data.get("sources", [])) == 0

    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "계정 발급 신청", "session_id": str(uuid.uuid4())},
        expected_status=200,
        expected_keys=["reply", "intent", "sources"],
        additional_assertions=assert_request_id_response
    )

# =============================================================
# 3. 새로운 테스트: 담당자 조회 시나리오
# =============================================================

def test_owner_lookup_no_screen(client):
    """
    '담당자' 키워드만 입력했을 때 전체 담당자 목록을 안내하는지 테스트합니다.
    (폴백 모드에서만 해당)
    """
    # Azure가 사용 가능한 경우 이 테스트는 스킵
    if AZURE_AVAILABLE:
        pytest.skip("이 테스트는 폴백 모드(Azure 비활성화)에서만 실행됩니다.")
    
    def assert_owner_list_response(data, response):
        # 폴백 모드에서는 도구 함수를 직접 호출하므로 그에 맞춰 검증
        assert data["intent"] == "direct_tool"
        assert "담당자" in data["reply"]
    
    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "담당자 알려줘", "session_id": str(uuid.uuid4())},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_owner_list_response
    )

def test_owner_lookup_specific_screen(client):
    """
    '인사시스템 담당자'와 같이 특정 시스템명을 입력했을 때 해당 담당자 정보를 반환하는지 테스트합니다.
    """
    def assert_specific_owner_response(data, response):
        if AZURE_AVAILABLE:
            assert data["intent"] == "agent_action"
        else:
            # 폴백 모드에서는 도구 함수를 직접 호출하므로 그에 맞춰 검증
            assert data["intent"] == "direct_tool"
            assert "담당자" in data["reply"]
    
    run_api_test(
        client,
        endpoint="/chat",
        payload={"message": "인사시스템 담당자 누구야?", "session_id": str(uuid.uuid4())},
        expected_status=200,
        expected_keys=["reply", "intent"],
        additional_assertions=assert_specific_owner_response
    )
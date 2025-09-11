# src/helpdesk_bot/constants.py

from pathlib import Path

# 경로 변수
KB_DEFAULT_DIR = Path("./kb_default")
KB_DATA_DIR = Path("./kb_data")
INDEX_DIR = Path("./index")
INDEX_NAME = "faiss_index"

# 상수 항목 및 샘플데이터
OWNER_FALLBACK = {
    "인사시스템-사용자관리": {"owner": "홍길동", "email": "owner.hr@example.com", "phone": "010-1234-5678"},
    "재무시스템-정산화면": {"owner": "김재무", "email": "owner.fa@example.com", "phone": "010-2222-3333"},
    "포털-공지작성": {"owner": "박운영", "email": "owner.ops@example.com", "phone": "010-9999-0000"},
}
EMPLOYEE_DIR = {
    "kim.s": {"name": "김선니", "dept": "IT운영", "phone": "010-1111-2222", "status": "active"},
    "lee.a": {"name": "이알파", "dept": "보안", "phone": "010-3333-4444", "status": "active"},
}
# 답변 제목
PREFIX_MESSAGES = {
    "ok": "[안내] 문의하신 내용에 대한 답변입니다.\n\n---\n\n",
    "fail": "[안내] 현재는 '기본 모드'로 운영되며, 자주 묻는 질문(FAQ) 및 핵심 업무(비밀번호 초기화, ID 발급 신청, 담당자 조회)만 지원합니다.\n\n---\n\n"
}
# 인사말 키워드
GREETINGS = ["안녕", "안녕하세요", "하이", "반가워", "헬로우", "hi", "hello"]
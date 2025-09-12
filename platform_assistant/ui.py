# platform_assistant/ui.py

import os
# import sys
# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parents[1]  # platform_assistant의 상위 = 프로젝트 루트
# SRC_DIR = BASE_DIR / "platform_service"
# sys.path.insert(0, str(BASE_DIR))  # 루트를 sys.path에 추가해야 함
import streamlit as st
import httpx
import uuid
from dotenv import load_dotenv

# =============================================================
# 환경 변수 로드 & 기본 설정
# =============================================================
load_dotenv()
API_BASE_URL = f"http://{os.getenv('API_CLIENT_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"

# =============================================================
# 유틸 함수
# =============================================================
@st.cache_data(ttl=60)
def check_api_health() -> bool:
    """API 서버 헬스체크 (60초 캐시)"""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_BASE_URL}/health")
            return resp.status_code == 200 and resp.json().get("ok")
    except Exception:
        return False

@st.cache_data(ttl=30)
def check_api_status() -> dict:
    """시스템 상태 조회 (/status)"""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_BASE_URL}/status")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {"ok": False, "azure_available": False}

def format_source_name(source_name: str) -> str:
    """문서 경로에서 사용자 친화적인 소스 이름으로 변환"""
    base = os.path.basename(source_name or "")
    return os.path.splitext(base)[0] if base else "알 수 없음"

# [checklist: 10] 서비스 개발 및 패키징 (Streamlit을 활용한 UI 개발)
# Streamlit을 사용하여 사용자 친화적인 웹 인터페이스를 구축함
# # =============================================================
# 메인 앱
# =============================================================
def main():
    st.set_page_config(page_title="사내 헬프데스크 챗봇", page_icon="💡", layout="wide")
    st.title("🌞 사내 헬프데스크 챗봇")
    # [checklist: 5] LangChain & LangGraph - 멀티턴 대화 (memory) 활용
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "안녕하세요! 무엇을 도와드릴까요?")]

    # 상태 값(한 번만 계산해 재사용)
    with st.spinner("시스템 상태 확인 중..."):
        sys_status = check_api_status()
    api_is_healthy = bool(sys_status.get("ok"))
    azure_available = bool(sys_status.get("azure_available"))

    # # 상단 표시
    # if not api_is_healthy:
    #     st.error("API 서버가 실행 중이지 않습니다. FastAPI 서버를 먼저 실행하세요.")
    # else:
    #     st.success("API 서버 연결됨")

    # ---------------------------------------------------------
    # 글로벌 스타일 (버튼 왼쪽 정렬)
    # ---------------------------------------------------------
    st.markdown(
        """
        <style>
        /* stButton 클래스 바로 아래 button 요소의 정렬 방식을 강제로 왼쪽으로 변경 */
          .stButton>button { justify-content: flex-start !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # 사이드바
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("🎓 AI 학습시키기")

        # 업로드
        uploaded = st.file_uploader(
            "문서 업로드",
            type=["pdf","csv","txt","md","docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded and api_is_healthy:
            try:
                with httpx.Client(timeout=30.0) as client:
                    files = [("files", (f.name, f.getvalue())) for f in uploaded]
                    resp = client.post(f"{API_BASE_URL}/upload", files=files)
                data = resp.json()
                if data.get("ok"):
                    st.success(f"{len(data.get('saved', uploaded))}개 문서 저장됨. 'Sync Content'를 눌러 반영하세요.")
                else:
                    st.error(f"업로드 실패: {data}")
            except Exception as e:
                st.error(f"업로드 오류: {e}")

        # 인덱스 재생성
        if st.button("Sync Content", disabled=not api_is_healthy, use_container_width=True):
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(f"{API_BASE_URL}/sync")
                data = resp.json()
                if data.get("ok"):
                    st.success("인덱스 재생성 완료!")
                else:
                    st.error(f"실패: {data.get('message')}")
            except Exception as e:
                st.error(f"Sync 오류: {e}")

        st.divider()

        # 시스템 상태 표시 (기존 UX 유지)
        with st.status("시스템 상태 확인 중...", expanded=False) as status_box:
            status_box.update(label="서버 연결 상태", state="complete", expanded=False)
            if api_is_healthy and azure_available:
                st.markdown("✅ AI : 온라인")
            elif api_is_healthy and not azure_available:
                st.markdown("⚠️ AI 제한: 기본모드")
            else:
                st.markdown("🚨 API 서버: 오프라인")
            status_box.update(label="서버 연결 상태", state="complete", expanded=True)
        st.divider()
        
        #  대화 기록 조회
        st.header("📜 대화 기록")
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(
                    f"{API_BASE_URL}/history",
                    params={"session_id": st.session_state["thread_id"], "limit": 10},
                )
            if resp.status_code == 200 and resp.json().get("ok"):
                msgs = resp.json().get("messages", [])
                if not msgs:
                    st.info("저장된 대화가 없습니다.")
                else:
                    for m in msgs:
                        role = "🧑 사용자" if m["role"] == "user" else "🤖 어시스턴트"
                        st.write(f"{role}: {m['message']}  \n🕒 {m['created_at']}")
            else:
                st.warning("대화 기록을 불러오지 못했습니다.")
        except Exception as e:
            st.error(f"조회 오류: {e}")

        st.markdown("---")
        st.caption("Service Desk Assistant © 2025")
    

    # ---------------------------------------------------------
    # 채팅 인터페이스
    # ---------------------------------------------------------
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    if q := st.chat_input("궁금한 것을 입력해주세요..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("처리 중..."):
                if not api_is_healthy:
                    st.error("API 서버가 오프라인입니다.")
                else:
                    try:
                        with httpx.Client(timeout=30.0) as client:
                            resp = client.post(
                                f"{API_BASE_URL}/chat",
                                json={
                                    "message": q,
                                    "session_id": st.session_state["thread_id"],
                                },
                            )
                        if resp.status_code == 200:
                            data = resp.json()
                            reply = data.get("reply", "")
                            sources = data.get("sources", [])

                            st.markdown(reply)
                            st.session_state.chat.append(("assistant", reply))

                            # 참고 문서(중복 제거 + page 표시)
                            if sources:
                                seen = set()
                                with st.expander("🔎 참고 자료"):
                                    for s in sources:
                                        raw_name = s.get("name") or s.get("source") or ""
                                        key = (raw_name, s.get("page"))
                                        if key in seen:
                                            continue
                                        seen.add(key)

                                        display = format_source_name(raw_name)
                                        if s.get("page") is not None:
                                            st.write(f"- {display}, page {int(s['page']) + 1}")
                                        else:
                                            st.write(f"- {display}")
                        else:
                            st.error(f"API 호출 실패 (status={resp.status_code})")
                    except Exception as e:
                        st.error(f"API 요청 중 오류 발생: {e}")

# =============================================================
# 실행 엔트리포인트
# =============================================================
if __name__ == "__main__":
    main()

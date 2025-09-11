# platform_assistant/ui.py

import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]  # platform_assistant의 상위 = 프로젝트 루트
SRC_DIR = BASE_DIR / "platform_service"
sys.path.insert(0, str(BASE_DIR))  # 루트를 sys.path에 추가해야 함
import streamlit as st
import httpx
import uuid
import logging

from platform_service import pipeline, build_or_load_vectorstore, AZURE_AVAILABLE, constants


logger = logging.getLogger(__name__)

# API 상태를 확인하는 함수 (60초 동안 결과를 캐시하여 성능 저하 방지)
@st.cache_data(ttl=180)
def check_api_health(api_base_url):
    """API 서버의 /health 엔드포인트를 확인하여 상태를 반환합니다."""
    try:
        with httpx.Client(timeout=2) as client:
            resp = client.get(f"{api_base_url}/health")
            return resp.status_code == 200
    except httpx.ConnectError:
        return False

def format_source_name(source_name: str) -> str:
    """
    파일 이름을 사용자가 이해하기 쉬운 설명으로 변환합니다.
    """
    known_sources = {
        "faq_data.csv": "자주 묻는 질문 (FAQ)"
       #, "seed-faq.txt": "기본 내장 지식"
    }
    display_name = known_sources.get(source_name, "참고 문서")
    return f"{display_name} (파일명: {source_name})"

# [checklist: 10] 서비스 개발 및 패키징 (Streamlit을 활용한 UI 개발)
# Streamlit을 사용하여 사용자 친화적인 웹 인터페이스를 구축함
def main():
    st.set_page_config(page_title="사내 헬프데스크 챗봇", page_icon="💡", layout="wide")
    st.title("🌞 사내 헬프데스크 챗봇")
    # st.markdown("도움이 필요하신가요? 질문을 입력해 주세요!")
    # [checklist: 5] LangChain & LangGraph - 멀티턴 대화 (memory) 활용
        # 💡 추가: 세션 상태에 thread_id가 없으면 새로 생성
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    # 버튼 텍스트를 왼쪽 정렬하는 CSS
    st.markdown("""
    <style>
        /* stButton 클래스 바로 아래 button 요소의 정렬 방식을 강제로 왼쪽으로 변경 */
        .stButton>button {
            justify-content: flex-start !important;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🎓 AI 학습시키기")

        # 전체적인 들여쓰기를 제어할 컬럼 생성
        left_space, main_col = st.columns([0.01, 0.99])

        # 파일 업로더와 버튼을 모두 오른쪽 컬럼(main_col) 안에 배치하여 시작점 통일
        # 문서 업로드 (PDF/CSV/TXT/DOCX)
        with main_col:
            uploaded = st.file_uploader(
                "문서 업로드", 
                type=["pdf","csv","txt","md","docx"], 
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if uploaded:
                constants.KB_DATA_DIR.mkdir(parents=True, exist_ok=True)
                for f in uploaded:
                    with open(constants.KB_DATA_DIR / f.name, "wb") as w:
                        w.write(f.read())
                st.success(f"{len(uploaded)}개 문서 저장됨. 'Sync Content'를 눌러 반영하세요.")

            # # 이 부분에서 중첩된 st.columns를 제거합니다.
            # btn_left, btn_mid, btn_right = st.columns([0.05, 0.7, 0.25])
            # with btn_mid:
            #     if st.butwqlton("Sync Content", disabled=not AZURE_AVAILABLE, use_container_width=True):
            #         try:
            #             for ext in [".faiss", ".pkl"]:
            #                 p = constants.INDEX_DIR / f"{constants.INDEX_NAME}{ext}"
            #                 if p.exists(): p.unlink()
            #             with st.spinner("Index 재생성 중..."):
            #                 build_or_load_vectorstore()
            #             st.success("완료!")
            #         except Exception as e:
            #             st.error(f"실패: {e}")
            
            # 버튼을 main_col에 직접 배치하여 오류를 해결합니다.
            if st.button("Sync Content", disabled=not AZURE_AVAILABLE, use_container_width=True):
                try:
                    for ext in [".faiss", ".pkl"]:
                        p = constants.INDEX_DIR / f"{constants.INDEX_NAME}{ext}"
                        if p.exists(): p.unlink()
                    with st.spinner("Index 재생성 중..."):
                        build_or_load_vectorstore()
                    st.success("완료!")
                except Exception as e:
                    st.error(f"실패: {e}")
        st.divider()
        api_host = os.getenv("API_CLIENT_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", 8001))
        api_base_url = f"http://{api_host}:{api_port}"
        
        with st.status("시스템 상태 확인 중...", expanded=False) as status:
            api_is_healthy = check_api_health(api_base_url)
            # 확인이 끝나면 이 메시지는 자동으로 사라집니다.
            status.update(label="서버 연결 상태", state="complete", expanded=False)

            if api_is_healthy and AZURE_AVAILABLE:
                st.markdown("✅ AI : 온라인")
            elif api_is_healthy and not AZURE_AVAILABLE:
                st.markdown("⚠️ AI 제한: 기본모드")
            else:
                st.markdown("🚨 API 서버: 오프라인")
            status.update(label="서버 연결 상태", state="complete", expanded=True)

    # 채팅 기록 표시
    if "chat" not in st.session_state:
        st.session_state.chat = [
            ("assistant", "안녕하세요! 무엇을 도와드릴까요?")
        ]

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    # 사용자 입력 처리
    if q := st.chat_input("궁금한 것을 입력해주세요..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"): st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("처리 중..."):
                reply, sources = None, []
                try:
                    if api_is_healthy:
                        with httpx.Client(timeout=30.0) as client:
                            resp = client.post(
                                f"{api_base_url}/chat", 
                                # 💡 수정: payload에 session_id 추가
                                json={"message": q, "session_id": st.session_state["thread_id"]}
                            )
                            resp.raise_for_status()
                            data = resp.json()
                            reply = data.get("reply",""); sources = data.get("sources", [])
                    else:
                        # 💡 수정: 로컬 폴백 모드에서도 session_id 전달 (사용되지는 않지만 API와의 일관성을 위해)
                        out = pipeline(q, st.session_state["thread_id"])
                        reply = out.get("reply",""); sources = out.get("sources", [])              
                except httpx.ConnectError:
                    st.warning("API 서버에 연결할 수 없어 로컬 폴백 모드로 자동 전환하여 재시도합니다.")
                    # 💡 수정: 로컬 폴백 모드에서도 session_id 전달
                    out = pipeline(q, st.session_state["thread_id"])
                    reply = out.get("reply",""); sources = out.get("sources", [])
                
                except Exception as e:
                    reply = f"오류: {e}"
                    sources = []

                # if reply:
                #     st.markdown(reply)
                #     if sources:
                #         with st.expander("🔎 참고 자료"):
                #             for s in sources:
                #                 source_display = format_source_name(s.get('source', '알 수 없음'))
                #                 if s.get("page") is not None:
                #                     line = f"- {source_display}, page {int(s['page']) + 1}"
                #                 else:
                #                     line = f"- {source_display}"
                #                 st.write(line)
                #     st.session_state.chat.append(("assistant", reply))
                if reply:
                    st.markdown(reply)
                    st.session_state.chat.append(("assistant", reply))  # 항상 기록
                    if sources:
                        with st.expander("🔎 참고 자료"):
                            # 중복 소스를 제거하기 위한 리스트
                            seen_sources = set()
                            
                            for s in sources:
                                source_key = s.get('source', '알 수 없음')
                                
                                # 중복되지 않은 소스만 추가
                                if source_key not in seen_sources:
                                    source_display = format_source_name(source_key)
                                    
                                    # 페이지 정보가 없는 경우와 있는 경우를 분리하여 표시
                                    if s.get("page") is not None:
                                        line = f"- {source_display}, page {int(s['page']) + 1}"
                                    else:
                                        line = f"- {source_display}"
                                    
                                    st.write(line)
                                    seen_sources.add(source_key) # 중복을 방지하기 위해 추가
                        st.session_state.chat.append(("assistant", reply))
if __name__ == "__main__":
    main()
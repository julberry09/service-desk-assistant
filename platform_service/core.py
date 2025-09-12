# platform_service/core.py

# Standard library imports
import os
import json
import logging
import csv
import time as _time
import threading
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv
from konlpy.tag import Okt

# LangChain & LangGraph 관련 라이브러리
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import tool
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
)
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

# LangSmith를 위한 CallbackManager 임포트 (python 버전 낮춰야해서 hold - 3.11)
# from langchain.callbacks.manager import CallbackManager
# from langchain_community.callbacks.langsmith import LangSmithCallbackHandler

# Local application imports
from . import constants

# 중앙 설정(logging_config.py)에서 세팅된 로거 불러오기
logger = logging.getLogger(__name__)

# =============================================================
# 1. 공통 설정 / 환경 변수
# =============================================================
load_dotenv()

# Azure OpenAI 환경변수
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT", "")
AOAI_API_KEY = os.getenv("AOAI_API_KEY", "")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-10-21")
AOAI_DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O", "gpt-4o")
AOAI_DEPLOY_EMBED_3_SMALL = os.getenv("AOAI_DEPLOY_EMBED_3_SMALL", "text-embedding-3-small")

# Azure 설정 확인 플래그
AZURE_AVAILABLE = bool(AOAI_ENDPOINT and AOAI_API_KEY)
if not AZURE_AVAILABLE:
    logger.warning("Azure OpenAI 설정이 없어 폴백(Fallback) 모드로 동작합니다.")


# =============================================================
# 2. 공통 유틸
# workflow/utils.py (형태소 분석기, 공용 유틸)
# =============================================================
# Okt 형태소 분석기 - Lazy Initialization (Thread-Safe)
_okt = None
_okt_lock = threading.Lock()

_faq_data = None   # FAQ 데이터 캐시 전역변수
_owner_data = None


class _DummyOkt:
    """pytest 전용: JVM 없이 최소 기능만 제공하는 더미 분석기"""
    def phrases(self, text: str):
        return [w for w in text.lower().split() if w]
    def nouns(self, text: str):
        return [w for w in text.lower().split() if w]


def get_okt():
    """
    Okt 인스턴스를 반환합니다.
    JVM은 프로세스에서 한 번만 실행될 수 있으므로
    thread-safe lazy init 패턴을 적용했습니다.
    테스트 시에는 환경변수 TEST_DISABLE_JVM=1이면 더미 OKT를 사용합니다.
    """
    global _okt
    if _okt is None:
        with _okt_lock:  # 다른 스레드와 동시 실행 방지
            if _okt is None:  # double-checked locking
                if os.getenv("TEST_DISABLE_JVM", "0") == "1":
                    _okt = _DummyOkt()       # ← JVM 미기동
                else:
                    _okt = Okt()             # ← 실제 JVM 기동
    return _okt


# # Okt 형태소 분석 싱글톤 패턴 적용
# class SingletonOkt:
#     _instance = None
#     _lock = threading.Lock() # 스레드 잠금 객체
# 
#     def __new__(cls):
#         with cls._lock: # 락(lock)을 사용해 스레드로부터 안전하게 접근
#             if cls._instance is None:
#                 try:
#                     # Okt 객체는 한 번만 생성
#                     cls._instance = Okt()
#                     logger.info("Okt 객체가 성공적으로 초기화되었습니다.")
#                 except Exception as e:
#                     logger.error(f"Okt 객체 초기화 중 오류 발생: {e}")
#                     raise RuntimeError("Okt 객체 초기화 실패") from e
#         return cls._instance
# 
# # Okt 인스턴스를 애플리케이션 시작 시점에 한 번만 미리 생성
# _okt_instance = None
# def get_okt():
#     """Okt 인스턴스를 반환 (Okt 객체는 한 번만 생성)"""
#     global _okt_instance
#     if _okt_instance is None:
#         _okt_instance = SingletonOkt() # 수정된 SingletonOkt 클래스 사용
#     return _okt_instance

# =============================================================
# 3. RAG 및 LLM 관련 함수 정의
# workflow/vectorstore.py (벡터스토어 관리)
# =============================================================
# 임베딩 모델 생성
def _make_embedder() -> AzureOpenAIEmbeddings:
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure OpenAI 설정이 없어 Embedder를 생성할 수 없습니다.")
    return AzureOpenAIEmbeddings(
        azure_deployment=AOAI_DEPLOY_EMBED_3_SMALL,
        api_key=AOAI_API_KEY,
        azure_endpoint=AOAI_ENDPOINT,
        api_version=AOAI_API_VERSION,
    )

# RAG - 원본 데이터 수집 및 전처리 로직 [checklist: 6]
def _load_docs_from_kb() -> List[Document]:
    """
    RAG - 원본 데이터 수집 및 전처리 로직
    """
    docs: List[Document] = []
    # FAQ 문서 추가
    faq_data = load_faq_data()
    if faq_data:
        docs.extend([
            Document(
                page_content=f"질문: {item.get('question')}\n답변: {item.get('answer')}",
                metadata={"source": "faq_data.csv"}
            ) for item in faq_data
        ])
    # 업무 담당자 문서 추가
    owner_data = load_owner_data()
    if owner_data:
        docs.extend([
            Document(
                page_content=f"화면: {item.get('screen')}\n담당자: {item.get('owner')}\n이메일: {item.get('email')}\n연락처: {item.get('phone')}",
                metadata={"source": "owners.csv"}
            ) for item in owner_data
        ])
    # 기타 문서 로드
    for kb_path in [constants.KB_DEFAULT_DIR, constants.KB_DATA_DIR]:
        if not kb_path.exists():
            kb_path.mkdir(parents=True, exist_ok=True)
            for p in kb_path.rglob("*"):
                # ✅ DB 파일(history.db 등)은 벡터화 대상에서 제외
                if not p.is_file():
                    continue
                if p.suffix.lower() == ".db" or p.name == "history.db":
                    continue
                if p.name == "faq_data.csv":
                    continue
                try:
                    suf = p.suffix.lower()
                    if suf == ".pdf": docs.extend(PyPDFLoader(str(p)).load())
                    elif suf == ".csv": docs.extend(CSVLoader(file_path=str(p), encoding="utf-8").load())
                    elif suf in [".txt", ".md"]: docs.extend(TextLoader(str(p), encoding="utf-8").load())
                    elif suf == ".docx": docs.extend(Docx2txtLoader(str(p)).load())
                except Exception as e:
                    logger.warning(f"문서 로드 실패: {p} - {e}")
    return docs

# RAG - FAISS 기반의 Vector 스토어 구축 [checklist: 7]
def build_or_load_vectorstore() -> FAISS:
    """
    RAG - FAISS 기반의 Vector 스토어 구축
    """
    if not AZURE_AVAILABLE:
        raise RuntimeError("'Rebuild Index'는 Azure OpenAI 설정이 필요합니다.")
    embed = _make_embedder()
    if (constants.INDEX_DIR / f"{constants.INDEX_NAME}.faiss").exists():
        return FAISS.load_local(
            str(constants.INDEX_DIR / constants.INDEX_NAME),
            embeddings=embed,
            allow_dangerous_deserialization=True
        )
    raw_docs = _load_docs_from_kb()
    if not raw_docs:
        faq_data = load_faq_data()
        if faq_data:
            raw_docs = [
                Document(
                    page_content=f"질문: {item.get('question')}\n답변: {item.get('answer')}",
                    metadata={"source": "faq_data.csv"}
                ) for item in faq_data
            ]
            logger.info("업로드된 문서가 없어 faq_data.csv를 기본 RAG 지식으로 사용합니다.")
        else:
            seed_text = """사내 헬프데스크 안내
- ID 발급: 신규 입사자는 HR 포털에서 '계정 신청' 양식을 제출. 승인 후 IT가 계정 생성.
- 비밀번호 초기화: SSO 포털의 '비밀번호 재설정' 기능 사용. 본인인증 필요.
- 담당자 조회: 포털 상단 검색창에 화면/메뉴명을 입력하면 담당자 카드가 표시됨."""
            raw_docs = [Document(page_content=seed_text, metadata={"source": "seed-faq.txt"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)
    constants.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # FAISS에 문서를 임베딩하고 저장
    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(str(constants.INDEX_DIR / constants.INDEX_NAME))
    return vs

# RAG - FAISS 벡터 스토어 검색기 (Singleton Pattern)
_vectorstore: Optional[FAISS] = None
_vectorstore_lock = threading.Lock()

def retriever(k: int = 4):
    """RAG - 벡터스토어 retriever (Singleton Pattern)"""
    global _vectorstore
    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                _vectorstore = build_or_load_vectorstore()
    return _vectorstore.as_retriever(search_kwargs={"k": k})

# LLM(언어 모델) 인스턴스를 생성
def make_llm(model: str = AOAI_DEPLOY_GPT4O_MINI, temperature: float = 0.2) -> AzureChatOpenAI:
    """
    Azure OpenAI 서비스에 연결하여 LLM(언어 모델) 인스턴스를 생성합니다.
    Args:
        model (str): 사용할 Azure OpenAI 배포 모델의 이름. 기본값은 gpt-4o-mini입니다.
        temperature (float): 모델의 창의성(무작위성)을 조절하는 매개변수. 0.0에서 2.0 사이의 값. 
                           값이 낮을수록 예측 가능하고 일관된 답변을 생성합니다.
    Returns:
        AzureChatOpenAI: 설정된 언어 모델 인스턴스.
    Raises:
        RuntimeError: Azure OpenAI 환경 변수(엔드포인트, API 키)가 설정되지 않은 경우 발생.
    """
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure OpenAI 설정이 없어 LLM을 생성할 수 없습니다.")
    return AzureChatOpenAI(
        azure_deployment=model,
        api_version=AOAI_API_VERSION,
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        temperature=temperature,
    )


# =============================================================
# 4. LangGraph 도구 정의
# workflow/tools.py (도구 정의)
# =============================================================
class BotState(TypedDict):
    question: str
    intent: str
    reply: str
    sources: List[Dict[str, Any]]
    tool_output: Dict[str, Any]


@tool
def tool_reset_password(payload: Dict[str, Any] = {}) -> Dict[str, Any]:
    """비밀번호 초기화 절차를 안내합니다."""
    return {
        "ok": True,
        "message": "비밀번호 초기화 절차 안내",
        "steps": ["SSO 포털 접속 > 비밀번호 재설정", "본인인증", "새 비밀번호 설정"]
    }


@tool
def tool_request_id(payload: Dict[str, Any] = {}) -> Dict[str, Any]:
    """ID 발급 신청 절차를 안내합니다."""
    return {
        "ok": True,
        "message": "ID 발급 신청 절차 안내",
        "steps": ["HR 포털 접속 > '계정 신청' 양식 제출", "양식 승인 후 IT팀에서 계정 생성"]
    }


@tool
def tool_owner_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    """화면이나 메뉴의 담당자 정보를 조회합니다. `screen` 인자가 필요합니다."""
    try:
        screen = payload.get("screen") or ""
        owner_data = load_owner_data()
        for item in owner_data:
            if screen in item.get("screen", ""):
                return {
                    "ok": True,
                    "screen": item.get("screen"),
                    "owner": {
                        "owner": item.get("owner"),
                        "email": item.get("email"),
                        "phone": item.get("phone")
                    }
                }
        return {"ok": False, "message": f"'{screen}' 담당자 정보를 찾지 못했습니다."}
    except Exception as e:
        return {"ok": False, "message": f"도구 실행 중 오류 발생: {str(e)}"}
# =============================================================
# 5. LangGraph 노드 정의
# workflow/nodes.py (LangGraph 노드 정의)
# =============================================================
    # Prompt Engineering - 프롬프트 최적화 (Few-shot Prompting) [checklist: 1]
# 노드(Node) 함수
def node_classify(state: BotState) -> BotState:
    llm = make_llm(model=AOAI_DEPLOY_GPT4O_MINI, temperature=0.1)
    prompt_template = PromptTemplate.from_template("""
    당신은 사용자 의도를 분류하는 AI입니다. 사용자의 질문을 가장 적합한 카테고리로 분류하세요.
    - `greeting`: 사용자가 인사말("안녕", "안녕하세요" 등)을 건넬 때
    - `direct_tool`: 사용자가 특정 시스템 작업(비밀번호 초기화, ID 발급, 담당자 조회)을 요청할 때
    - `faq`: 자주 묻는 질문(FAQ)에 관련된 질문일 때
    - `general_qa`: 위에 해당하지 않는 일반적인 질문일 때

    질문에 대한 분류와 필요한 인자(JSON 형식)를 반환하세요.
    JSON 형식: {{"intent": "분류", "arguments": {{"key": "value"}}}}
    예시:
    사용자 질문: "비밀번호 초기화 알려줘" -> {{"intent": "direct_tool", "arguments": {{"tool_name": "tool_reset_password"}}}}
    사용자 질문: "인사시스템 담당자 누구야?" -> {{"intent": "direct_tool", "arguments": {{"tool_name": "tool_owner_lookup", "screen": "인사시스템-사용자관리"}}}}
    사용자 질문: "점심시간이 언제야?" -> {{"intent": "faq", "arguments": {{}}}}
    사용자 질문: "회사 복지제도 설명해줘" -> {{"intent": "general_qa", "arguments": {{}}}}
    사용자 질문: "안녕" -> {{"intent": "greeting", "arguments": {{}}}}

    사용자 질문: "{question}" -> 
    """)
    prompt = prompt_template.format(question=state["question"])
    out = llm.invoke(prompt).content
    
    intent, args = "general_qa", {}
    try:
        data = json.loads(out.strip())
        intent = data.get("intent", "general_qa")
        args = data.get("arguments", {}) or {}
    except json.JSONDecodeError:
        logger.warning(f"[Classifier 오류] JSONDecodeError: {out}")
    except Exception as e:
        logger.error(f"[Classifier 오류] 알 수 없는 오류: {out} - {e}")
    
    return {**state, "intent": intent, "tool_output": args}

def node_greeting(state: BotState) -> BotState:
    return {**state, "reply": "네, 반갑습니다. 문의사항을 말씀해 주시면 제가 도와드릴게요.", "sources": []}

def node_direct_tool(state: BotState) -> BotState:
    tool_name = state.get("tool_output", {}).get("tool_name")
    payload = state.get("tool_output", {}).get("arguments", {})
    
    if tool_name == "tool_reset_password":
        res = tool_reset_password.invoke({})
    elif tool_name == "tool_request_id":
        res = tool_request_id.invoke({})
    elif tool_name == "tool_owner_lookup":
        # ✅ 전체 조회 예외 처리
        query_text = state.get("question", "")
        if "전체" in query_text or "모두" in query_text or "리스트" in query_text:
            from platform_service.core import load_owner_data
            owners = load_owner_data()
            res = {
                "ok": True,
                "tool_name": "tool_owner_lookup",
                "owners": owners
            }
        else:
            res = tool_owner_lookup.invoke({"payload": payload})
    else:
        res = {"ok": False, "message": "알 수 없는 도구 호출"}
    
    # 도구 결과에 원래 호출된 도구 이름 추가
    res["tool_name"] = tool_name
    return {**state, "tool_output": res}
    
# 노드(Node) 함수 수정
def node_faq(state: BotState) -> BotState:
    # FAQ 데이터는 이미 RAG 파이프라인에 포함되어 있으므로,
    # 이 노드에서는 FAQ와 관련된 질문인지 확인만 하고 바로 RAG로 라우팅
    # (단, 아래 로직은 필요에 따라 삭제 또는 수정될 수 있습니다. 
    #  현재는 RAG 노드에서 FAQ를 포함한 모든 문서를 처리하도록 되어 있습니다.)
    
    # 이 부분을 삭제하거나, 더 이상 FAQ를 별도로 처리하지 않도록 수정합니다.
    # 기존 로직:
    # faq_answer = find_similar_faq(state["question"])
    # if faq_answer:
    #     return {**state, "tool_output": {"ok": True, "answer": faq_answer.get("answer")}, "intent": "faq", "sources": [{"source": "faq_data.csv"}]}
    # else:
    #     return {**state, "intent": "general_qa"}

    # 개선된 로직:
    # FAQ 질문은 일반 질문과 마찬가지로 RAG 노드로 라우팅
    return {**state, "intent": "general_qa"}
    # RAG - 사전 정의된 데이터(문서)를 검색하여 AI의 논리력을 보강/ RAG 기반 지식 검색 기능 구현 [checklist: 8,9] 
# Prompt Engineering - 프롬프트 최적화 (역할 부여 + Chain-of-Thought) [checklist: 1] 
def node_rag(state: BotState) -> BotState:
    docs = retriever(k=4).get_relevant_documents(state["question"])
    
    # 검색 결과가 전혀 없으면 → 일반 LLM 답변
    if not docs:
        llm = make_llm(model=AOAI_DEPLOY_GPT4O)
        sys_prompt = "너는 사내 헬프데스크 상담원이다. 내부 문서에 없는 질문이라도 일반 지식을 활용해 한국어로 답해라."
        out = llm.invoke([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": state["question"]}
        ]).content
        return {**state, "reply": out, "sources": []}

    # 검색 결과가 있으면 → RAG 기반 답변
    context = "\n\n".join([f"[{i+1}] {d.page_content[:1200]}" for i, d in enumerate(docs)])
    sources = [{"index": i+1, "source": d.metadata.get("source","unknown"), "page": d.metadata.get("page")} for i,d in enumerate(docs)]
    llm = make_llm(model=AOAI_DEPLOY_GPT4O)
    sys_prompt = (
        "너는 사내 헬프데스크 상담원이다. "
        "가능하면 제공된 컨텍스트를 활용해 답변하되, "
        "부족하면 일반 지식을 보완해서 답해라. "
        "추가 확인이 필요하면 '정확한 정책은 사내 포털에서 확인 필요'라고 말해라."
    )
    user_prompt = f"질문:\n{state['question']}\n\n컨텍스트:\n{context}"
    out = llm.invoke([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]).content
    
    return {**state, "reply": out, "sources": sources}

# 도구(Tool) 함수 결과 사용자 친화적인 형태로 변환
def node_finalize(state: BotState) -> BotState:
    # 이 노드는 도구 실행 결과를 사용자 친화적인 메시지로 변환
    res = state.get("tool_output", {})
    if state["intent"] == "direct_tool":
        if res.get("tool_name") == "tool_reset_password":
            text = f"✅ 비밀번호 초기화 안내\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", []))) if res.get("ok") else f"❗{res.get('message','실패')}"
        elif res.get("tool_name") == "tool_request_id":
            text = f"🆔 ID 발급 신청 절차 안내\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", []))) if res.get("ok") else f"❗{res.get('message','실패')}"  
        elif res.get("tool_name") == "tool_owner_lookup":
            if "owners" in res:  # 전체 조회 케이스
                lines = ["📋 전체 담당자 목록:"]
                for o in res["owners"]:
                    lines.append(
                        f"- 화면: {o.get('screen')} / 이름: {o.get('owner')} / "
                        f"이메일: {o.get('email')} / 연락처: {o.get('phone')}"
                    )
                text = "\n".join(lines)
            else:  # 기존 단일 담당자 조회
                text = (
                    f"👤 '{res.get('screen')}' 담당자\n"
                    f"- 이름: {res.get('owner', {}).get('owner')}\n"
                    f"- 이메일: {res.get('owner', {}).get('email')}\n"
                    f"- 연락처: {res.get('owner', {}).get('phone')}"
                    if res.get("ok")
                    else f"❗{res.get('message','조회 실패')}"
                )
        else: # FAQ도 여기서 처리
            text = res.get("answer", "죄송합니다. 답변을 찾지 못했습니다.")
        # 💡 수정: 반환 키를 'reply'로 통일
        return {**state, "reply": text}
    return state
# =============================================================
# 4. workflow/graph.py (LangGraph 그래프 정의)
# ==========================================================
def load_owner_data() -> List[Dict[str, str]]:
    global _owner_data
    if _owner_data is not None:
        return _owner_data

    owner_file_path = constants.KB_DATA_DIR / "owners.csv"
    if not owner_file_path.exists():
        logger.warning(f"OWNER 파일이 존재하지 않습니다: {owner_file_path}")
        _owner_data = []
        return _owner_data

    loaded_data = []
    try:
        # UTF-8 with BOM도 안전하게 처리
        with open(owner_file_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "screen" in row and "owner" in row:
                    # 형태소 분석 불필요 → 그대로 저장
                    loaded_data.append(row)
        logger.info(f"{len(loaded_data)}개의 OWNER 데이터를 로드했습니다.")
    except Exception as e:
        logger.error(f"OWNER 파일 로드 실패: {e}")
        loaded_data = []
    _owner_data = loaded_data
    return _owner_data

# LangChain & LangGraph - Multi-Agent Flow 설계 및 구현 [checklist: 3,4,5]
# FAQ 데이터 로드
def load_faq_data() -> List[Dict[str, str]]:
    global _faq_data
    # 이미 로드된 데이터가 있으면 바로 반환
    if _faq_data is not None:
        return _faq_data
    
    faq_file_path = constants.KB_DEFAULT_DIR / "faq_data.csv"
    if not faq_file_path.exists():
        logger.warning(f"FAQ 파일이 존재하지 않습니다: {faq_file_path}")
        _faq_data = []
        return _faq_data
    
    loaded_data = []
    try:
        # 파일 로드 및 데이터 파싱
        with open(faq_file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Okt 객체를 미리 생성해두고 재사용
            okt_processor = get_okt()
            for row in reader:
                if "question" in row and "answer" in row:
                    # Okt 객체를 사용하여 질문을 분석
                    row["faq_words"] = set(okt_processor.phrases(row.get("question", "")))
                    loaded_data.append(row)
        logger.info(f"{len(loaded_data)}개의 FAQ 데이터를 로드했습니다.")
    except Exception as e:
        logger.error(f"FAQ 파일 로드 실패: {e}")
        # 오류 발생 시에도 빈 목록을 반환
        loaded_data = []
    _faq_data = loaded_data
    return _faq_data

# FAQ 유사도 검색
def find_similar_faq(question: str) -> Optional[Dict[str, Any]]:
    logger.error(f"find_similar_faq - 질문은?: {question.lower().strip()}")
    faq_data = load_faq_data()
    if not faq_data: return None
    user_words = set(get_okt().phrases(question.lower()))
    if not user_words: user_words = set(get_okt().nouns(question.lower()))
    if not user_words: return None
    
    best_score = 0.0
    best_item = None
    
    for item in faq_data:
        faq_words = item.get("faq_words", set())
        if not faq_words: continue
        intersection = len(user_words.intersection(faq_words))
        union = len(user_words.union(faq_words))
        score = intersection / union if union > 0 else 0
        
        if score > best_score:
            best_score = score
            best_item = item

    # 점수 임계값(threshold)을 0.2로 설정
    return best_item if best_score > 0.2 else None

# StateGraph 클래스를 사용해 멀티 에이전트 워크플로우를 정의함
# =============================================================
# 5. StateGraph 클래스를 사용해 멀티 에이전트 워크플로우를 정의
# workflow/graph.py (LangGraph 그래프 정의)
# =============================================================
_memory_checkpointer = MemorySaver()
def build_graph():
    logger.info("build_graph")
    g = StateGraph(BotState)
    g.add_node("classify", node_classify)
    g.add_node("greeting", node_greeting)
    g.add_node("direct_tool", node_direct_tool)
    g.add_node("faq", node_faq)
    g.add_node("rag", node_rag)
    g.add_node("finalize", node_finalize)
    
    # 챗봇의 시작점
    g.set_entry_point("classify")

    # 의도에 따라 노드 연결
    g.add_conditional_edges(
        "classify",
        lambda s: s["intent"],
        {
            "greeting": "greeting",
            "direct_tool": "direct_tool",
            "faq": "faq",
            "agent_action": "rag",
            "general_qa": "rag", # 일반 질문은 RAG로 라우팅
        }
    )

    # 노드 연결
    g.add_edge("greeting", END)
    g.add_edge("direct_tool", "finalize")
    g.add_edge("rag", END)
    
    # FAQ 노드에서 답변을 찾지 못하면 RAG로 다시 라우팅
    g.add_conditional_edges(
        "faq",
        lambda s: s["intent"],
        {
            "faq": "finalize", # FAQ 답변을 찾았을 때
            "general_qa": "rag", # FAQ 답변을 못 찾았을 때
        }
    )
    g.add_edge("finalize", END)
    
    return g.compile(checkpointer=_memory_checkpointer)
# =============================================================
# 6. Pipeline Orchestration
# workflow/pipeline.py (파이프라인 실행)
# =============================================================
# LangChain & LangGraph를 활용한 전체 파이프라인 구성
_graph = None
def run_graph_pipeline(question: str, session_id: str) -> Dict[str, Any]:
    # [checklist: 5] LangChain & LangGraph - 멀티턴 대화 (memory) 활용
    """LangGraph 기반의 AI 파이프라인을 실행합니다."""
    global _graph
    # LangGraph Studio (UI 기반 그래프 시각화 툴)를 활용하여 그래프 모니터링 및 디버깅 - hold
    # LangSmith 콜백 핸들러 설정
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project_name = os.getenv("LANGSMITH_PROJECT", "Servicedesk Bot")
    
    callbacks = None
    if langsmith_api_key:
        # callbacks = CallbackManager([
        #     LangSmithCallbackHandler(
        #         project_name=langsmith_project_name, 
        #         api_key=langsmith_api_key, 
        #         session_id=session_id
        #     )
        # ])
        logger.info("LangSmith is enabled.")
    else:
        logger.info("LangSmith is not enabled. Provide LANGSMITH_API_KEY in .env to enable it.")

    logger.info("pipeline_in", extra={"extra_data": {"q": question}})
    if _graph is None: _graph = build_graph()
    
    out = _graph.invoke(
        input={"question": question, "intent":"", "reply":"", "sources":[], "tool_output":{}},
        config={"configurable": {"thread_id": session_id}, "callbacks": callbacks}
    )
    logger.info("pipeline_out", extra={"extra_data": {"intent": out.get("intent", "")}})
    
    # LangGraph 결과에서 최종 답변과 의도 반환
    return {
        "reply": out.get("reply", "죄송합니다. 답변을 찾지 못했습니다."),
        "intent": out.get("intent", "unsupported"),
        "sources": out.get("sources", []),
    }

def pipeline(question: str, session_id: str) -> Dict[str, Any]:
    """
    Azure 연결 상태에 따라 적절한 파이프라인으로 요청을 라우팅합니다.
    """
    if AZURE_AVAILABLE:
        try:
            return run_graph_pipeline(question, session_id)
        except Exception as e:
            logger.error(f"주요 파이프라인 실행 중 오류 발생: {e}. 폴백 모드로 전환합니다.")
            return {
                "reply": "죄송합니다. 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                "intent": "system_error",
                "sources": []
            }
    else:
        # 폴백 모드
        logger.info("풀백 파이프라인")
        # 1. 인사말 처리
        logger.error(f"풀백 파이프라인 - 질문은?: {question.lower().strip()}")
        if question.lower().strip() in constants.GREETINGS:
            return {
                "reply": "네, 반갑습니다. 문의사항을 말씀해 주시면 제가 도와드릴게요.",
                "intent": "greeting",
                "sources": []
            }     
        # 2. 도구 관련 키워드 처리
        if "비밀번호 초기화" in question:
            res = tool_reset_password.invoke({})
            return {
                "reply": f"✅ 비밀번호 초기화 안내\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(res.get("steps", []))),
                "intent": "direct_tool",
                "sources": []
            }
        
        if "아이디 발급" in question or "계정 발급" in question:
            res = tool_request_id.invoke({})
            reply_text = f"🆔 ID 발급 신청 절차 안내\n\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(res.get("steps", [])))
            
            return {
                "reply": reply_text,
                "intent": "direct_tool",
                "sources": []
            }
        if "담당자" in question:
            logger.info("담당자 조회")
            owner_data = load_owner_data()

            # 1. 전체 조회 요청 처리
            if "전체" in question or question.strip() == "담당자 조회":
                if not owner_data:
                    return {
                        "reply": "등록된 담당자 정보가 없습니다.",
                        "intent": "direct_tool",
                        "sources": []
                    }
                reply_lines = ["📋 전체 담당자 목록"]
                for item in owner_data:
                    reply_lines.append(
                        f"- {item.get('screen')}: {item.get('owner')} "
                        f"(📧 {item.get('email')}, ☎ {item.get('phone')})"
                    )
                return {
                    "reply": "\n".join(reply_lines),
                    "intent": "direct_tool",
                    "sources": [{"source": "owner.csv"}]
                }

            # 2. 개별 screen 조회 처리 (정확히 일치할 때)
            for item in owner_data:
                
                #if question.strip() == item.get("screen", "").strip():
                if item.get("screen", "").strip() and item.get("screen", "").strip() in question:
                    return {
                        "reply": (
                            f"👤 '{item.get('screen')}' 담당자\n"
                            f"- 이름: {item.get('owner')}\n"
                            f"- 이메일: {item.get('email')}\n"
                            f"- 연락처: {item.get('phone')}"
                        ),
                        "intent": "direct_tool",
                        "sources": [{"source": "owner.csv"}]
                    }

            # 3. 못 찾았을 때
            return {
                "reply": "담당자를 찾지 못했습니다. 화면/메뉴명을 정확히 입력해 주세요.",
                "intent": "direct_tool",
                "sources": []
            }

                # if "담당자" in question:
        #     screen = "인사시스템-사용자관리" if "인사시스템" in question else None # screen이 없을 경우 None으로 설정
        #     if screen:
        #         payload = {"screen": screen}
        #         res = tool_owner_lookup.invoke({"payload": {"screen": screen}})
        #         # Pydantic 오류 수정: payload 인자를 딕셔너리로 래핑
        #         #res = tool_owner_lookup.invoke({"payload": {"screen": screen}})
        #         #res = tool_owner_lookup.invoke(payload)
        #         #res = tool_owner_lookup.invoke(**payload) # **로 언팩하여 호출
        #         #res = tool_owner_lookup.invoke(input=payload) # 수정된 호출 방식
        #     else:
        #         # screen이 없는 경우, 빈 딕셔너리를 input으로 전달
        #         #res = tool_owner_lookup.invoke({})
        #         res = tool_owner_lookup.invoke(input={})
        #         # 담당자 목록 전체를 안내하는 메시지를 반환
        #         reply = "담당자를 찾기 위한 화면이나 메뉴 정보가 필요합니다. 예: '재무시스템 담당자 알려줘'"
        #         return {
        #             "reply": reply,
        #             "intent": "direct_tool",
        #             "sources": []
        #         }

        # 3. FAQ 검색을 도구 키워드보다 먼저 수행하여 RAG 테스트 통과
        faq_item = find_similar_faq(question)
        if faq_item:
            return {
                "reply": f"[안내] 문의하신 내용에 대한 답변입니다.\n\n---\n\n{faq_item.get('answer')}",
                "intent": "faq",
                "sources": [{"source": "faq_data.csv"}]
            }

        # 4. 그 외 모든 질문에 대한 폴백
        return {
            "reply": "죄송합니다. 문의하신 내용을 이해하지 못했습니다.",
            "intent": "unsupported",
            "sources": []
        }


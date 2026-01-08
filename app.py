import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import json
import requests
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Firebase 라이브러리
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-AI Agent", page_icon="🤖", layout="wide")

# 모바일 최적화 CSS (기존 유지)
st.markdown("""
    <style>
        footer { visibility: hidden; }
        @media only screen and (max-width: 600px) {
            .main .block-container {
                padding-left: 0.2rem !important;
                padding-right: 0.2rem !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
            div[data-testid="stMarkdownContainer"] table {
                width: 100% !important;
                table-layout: fixed !important;
                display: table !important;
                font-size: 10px !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stMarkdownContainer"] th, 
            div[data-testid="stMarkdownContainer"] td {
                padding: 1px 1px !important;
                word-wrap: break-word !important;
                word-break: break-all !important;
                white-space: normal !important;
                line-height: 1.1 !important;
                vertical-align: middle !important;
            }
            div[data-testid="stMarkdownContainer"] th:first-child,
            div[data-testid="stMarkdownContainer"] td:first-child {
                width: 35px !important;
                font-size: 8px !important;
                text-align: center !important;
                letter-spacing: -0.5px !important;
            }
            button { min-height: 45px !important; }
            input { font-size: 16px !important; }
        }
    </style>
""", unsafe_allow_html=True)

# API Key 로드
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("🚨 Google API Key 설정이 필요합니다.")
    st.stop()

# 세션 초기화 (에이전트용)
if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []  # 통합 채팅 기록
if "timetable_data" not in st.session_state:
    st.session_state.timetable_data = ""      # 생성된 시간표 데이터
if "graduation_data" not in st.session_state:
    st.session_state.graduation_data = ""     # 졸업 진단 결과
if "user" not in st.session_state:
    st.session_state.user = None

# HTML 정제 함수
def clean_html_output(text):
    cleaned = text.strip()
    if cleaned.startswith("```html"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    return cleaned.replace("```html", "").replace("```", "").strip()

# 재시도 로직
def run_with_retry(func, *args, **kwargs):
    max_retries = 3
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 ** i)
                continue
            raise e

# -----------------------------------------------------------------------------
# [Firebase Manager] (기존 코드 유지)
# -----------------------------------------------------------------------------
class FirebaseManager:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.init_firestore()

    def init_firestore(self):
        if "firebase_service_account" in st.secrets:
            try:
                if not firebase_admin._apps:
                    cred_info = dict(st.secrets["firebase_service_account"])
                    cred = credentials.Certificate(cred_info)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_initialized = True
            except: pass

    def auth_user(self, email, password, mode="login"):
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            return None, "API Key Error"
        api_key = st.secrets["FIREBASE_WEB_API_KEY"].strip()
        endpoint = "signInWithPassword" if mode == "login" else "signUp"
        # URL 수정 완료 (마크다운 제거)
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key}"
        try:
            res = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True})
            data = res.json()
            if "error" in data: return None, data["error"]["message"]
            return data, None
        except Exception as e: return None, str(e)

    def save_chat(self, history):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            user_id = st.session_state.user['localId']
            # 채팅 내역은 너무 길 수 있으므로 최근 10개만 저장 예시
            save_data = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
            self.db.collection('users').document(user_id).collection('agent_chats').add({
                "history": save_data, "created_at": firestore.SERVER_TIMESTAMP
            })
            return True
        except: return False

fb_manager = FirebaseManager()

# PDF 데이터 로드
@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("data"): return ""
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files: return ""
    all_content = ""
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            for page in pages: all_content += page.page_content
        except: continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [AI Tools] 에이전트가 사용할 도구들
# -----------------------------------------------------------------------------
def get_llm(model_name="gemini-2.5-flash-preview-09-2025"):
    if not api_key: return None
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

# 1. 일반 질의응답 (RAG)
def tool_qa(query):
    llm = get_llm()
    prompt = f"""
    [문서 내용] {PRE_LEARNED_DATA[:50000]}... (생략)
    [질문] {query}
    문서 내용을 바탕으로 답변해. 근거가 되는 문장은 " "로 인용해.
    """
    return llm.invoke(prompt).content

# 2. 시간표 생성
def tool_generate_timetable(major, grade, semester, credit, requirements, blocked_times):
    llm = get_llm()
    
    # (기존 generate_timetable_ai 프롬프트 로직 재사용)
    common_instruction = """
    [엄격한 제약사항]
    1. 요람의 '{major} {grade} {semester}' 필수 과목을 반드시 포함하라.
    2. 시간표 데이터와 학년/이름이 정확히 일치하는 과목만 넣어라.
    3. 출력은 반드시 HTML Table 형식으로 하라. (가로폭 100%, 파스텔톤 배경)
    4. 온라인 강의는 표 맨 아래 행에 포함시켜라.
    """
    
    prompt = f"""
    전문가로서 시간표를 생성해.
    정보: {major} {grade} {semester}, 목표 {credit}학점.
    공강 요청: {blocked_times}. 추가요구: {requirements}.
    
    {common_instruction}
    
    [문서 데이터] {PRE_LEARNED_DATA}
    """
    res = llm.invoke(prompt).content
    return clean_html_output(res)

# 3. 졸업 진단
def tool_audit_graduation(images):
    llm = get_llm() # 멀티모달 지원
    
    img_content = []
    for img_file in images:
        img_file.seek(0)
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
        img_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    prompt_text = f"""
    업로드된 성적표 이미지를 분석하여 졸업 요건을 진단해.
    [학습된 요람 데이터] {PRE_LEARNED_DATA}
    종합 판정, 이수 현황(표), 미이수 과목, 조언 순으로 마크다운 리포트를 작성해.
    """
    
    msg = HumanMessage(content=[{"type": "text", "text": prompt_text}] + img_content)
    return llm.invoke([msg]).content

# -----------------------------------------------------------------------------
# [Agent Router] 사용자의 의도를 분류하는 두뇌
# -----------------------------------------------------------------------------
def route_intent(user_input):
    llm = get_llm()
    prompt = f"""
    사용자의 입력: "{user_input}"
    
    이 입력이 다음 중 어떤 작업에 해당하는지 분류하여 단어 하나만 출력하세요.
    1. TIMETABLE: 시간표 생성, 추천, 수정 요청 (예: "시간표 짜줘", "1교시 빼줘")
    2. GRADUATION: 졸업 요건, 학점 확인, 성적표 분석 (예: "졸업 가능해?", "이거 학점 인정돼?")
    3. QA: 학교 규정, 장학금, 일반적인 질문 (예: "재수강 학점 제한이 뭐야?")
    4. CHAT: 단순 인사나 잡담
    
    출력 예시: TIMETABLE
    """
    return llm.invoke(prompt).content.strip().upper()

# -----------------------------------------------------------------------------
# [UI] 메인 인터페이스
# -----------------------------------------------------------------------------

# 1. 사이드바: 에이전트에게 정보를 주는 곳 (Context Provider)
with st.sidebar:
    st.title("🤖 AI 에이전트 설정")
    
    # 로그인
    if st.session_state.user:
        st.info(f"🔑 {st.session_state.user['email']}")
        if st.button("로그아웃"):
            st.session_state.user = None
            st.rerun()
    else:
        with st.expander("로그인 / 회원가입"):
            email = st.text_input("이메일")
            pw = st.text_input("비밀번호", type="password")
            if st.button("로그인"):
                user, err = fb_manager.auth_user(email, pw, "login")
                if user: 
                    st.session_state.user = user
                    st.rerun()
                else: st.error(err)
            if st.button("회원가입"):
                user, err = fb_manager.auth_user(email, pw, "signup")
                if user:
                    st.session_state.user = user
                    st.rerun()
                else: st.error(err)

    st.divider()
    
    # 내 정보 (시간표 생성용)
    st.caption("📅 시간표 생성 설정")
    
    kw_departments = [
        "전자융합공학과", "전자공학과", "전자통신공학과", "전기공학과", 
        "전자재료공학과", "로봇학부", "컴퓨터정보공학부", "소프트웨어학부", 
        "정보융합학부", "건축학과", "건축공학과", "화학공학과", "환경공학과"
    ]
    
    major = st.selectbox("학과", kw_departments, key="agent_major")
    col1, col2 = st.columns(2)
    grade = col1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="agent_grade")
    semester = col2.selectbox("학기", ["1학기", "2학기"], key="agent_sem")
    credit = st.number_input("목표 학점", 9, 24, 18)
    reqs = st.text_area("추가 요구사항 (예: 오전 수업 X)")
    
    # 공강 설정
    with st.popover("공강 요일/시간 설정"):
        st.info("체크 해제 = 공강")
        # 간단하게 요일별 오전/오후 체크박스로 구현 (실제론 더 디테일하게 가능)
        days = ["월", "화", "수", "목", "금"]
        blocked_desc = []
        for d in days:
            if not st.checkbox(f"{d}요일 수업 가능", value=True, key=f"chk_{d}"):
                blocked_desc.append(d)
    
    st.divider()
    
    # 자료 제출 (졸업 진단용)
    st.caption("🎓 졸업 진단용 성적표")
    uploaded_imgs = st.file_uploader("성적표 캡처 업로드", type=['png', 'jpg'], accept_multiple_files=True)

# 2. 메인 채팅 인터페이스
st.title("🎓 KW-강의마스터 AI")
st.caption("무엇이든 말씀하세요. AI가 알아서 시간표를 짜거나 졸업 요건을 봐드립니다.")

# 대화 기록 출력
for msg in st.session_state.agent_chat_history:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "html":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("예: 1학년 시간표 짜줘, 졸업 가능한지 봐줘"):
    # 1. 사용자 메시지 표시
    st.session_state.agent_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 에이전트 생각 (Router)
    with st.chat_message("assistant"):
        with st.spinner("AI가 작업을 판단하고 있습니다..."):
            intent = route_intent(prompt)
            response_content = ""
            response_type = "text"
            
            # 3. 도구 실행 (Action)
            if intent == "TIMETABLE":
                st.info(f"📅 [{major} {grade}] 시간표 생성을 시작합니다...")
                blocked_str = ", ".join(blocked_desc) + "요일 공강" if blocked_desc else "공강 없음"
                
                # 기존에 생성된 시간표가 있고 '수정' 요청인 경우 context 유지
                if st.session_state.timetable_data and ("수정" in prompt or "빼줘" in prompt):
                     # 수정 로직 (약식 구현: 새로 생성하되 요구사항에 프롬프트 추가)
                     reqs += f" (수정 요청: {prompt})"
                
                html_table = tool_generate_timetable(major, grade, semester, credit, reqs, blocked_str)
                st.session_state.timetable_data = html_table
                response_content = html_table
                response_type = "html"
                st.markdown(response_content, unsafe_allow_html=True)
                
            elif intent == "GRADUATION":
                if not uploaded_imgs:
                    response_content = "🎓 졸업 요건을 진단하려면 먼저 왼쪽 사이드바에 **성적표 이미지를 업로드**해주세요!"
                    st.warning(response_content)
                else:
                    st.info("🎓 성적표를 분석하여 졸업 요건을 진단합니다...")
                    report = tool_audit_graduation(uploaded_imgs)
                    st.session_state.graduation_data = report
                    response_content = report
                    st.markdown(response_content)
            
            elif intent == "QA":
                response_content = tool_qa(prompt)
                st.markdown(response_content)
                
            else: # CHAT
                # 가벼운 대화 모델 호출
                llm = get_llm()
                response_content = llm.invoke(f"사용자: {prompt}\n친절한 학사 조교처럼 답변해.").content
                st.markdown(response_content)
            
            # 4. 결과 저장
            st.session_state.agent_chat_history.append({"role": "assistant", "content": response_content, "type": response_type})
            
            # 로그인 시 자동 클라우드 백업
            if st.session_state.user:
                fb_manager.save_chat(st.session_state.agent_chat_history)


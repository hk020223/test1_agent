import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import json
import uuid
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

# 모바일 최적화 및 UI 개선 CSS
st.markdown("""
    <style>
        footer { visibility: hidden; }
        
        /* 모바일 최적화 */
        @media only screen and (max-width: 600px) {
            .main .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
            
            /* 시간표 테이블 모바일 스타일 */
            div[data-testid="stMarkdownContainer"] table {
                width: 100% !important;
                table-layout: fixed !important;
                display: table !important;
                font-size: 11px !important;
                margin-bottom: 0px !important;
            }
            
            div[data-testid="stMarkdownContainer"] th, 
            div[data-testid="stMarkdownContainer"] td {
                padding: 2px !important;
                word-wrap: break-word !important;
                word-break: break-all !important;
                white-space: normal !important;
                line-height: 1.2 !important;
                vertical-align: middle !important;
            }
            
            /* 교시 열 너비 고정 */
            div[data-testid="stMarkdownContainer"] th:first-child,
            div[data-testid="stMarkdownContainer"] td:first-child {
                width: 40px !important;
                font-size: 9px !important;
                text-align: center !important;
                background-color: #f8f9fa;
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

# -----------------------------------------------------------------------------
# [Firebase Manager] 로그인, 저장, 불러오기
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
        # URL 형식 수정 완료
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key}"
        try:
            res = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True})
            data = res.json()
            if "error" in data: return None, data["error"]["message"]
            return data, None
        except Exception as e: return None, str(e)

    # 사용자 프로필(학과, 학년 등) 저장
    def save_profile(self, profile_data):
        if not self.is_initialized or not st.session_state.user: return
        try:
            uid = st.session_state.user['localId']
            self.db.collection('users').document(uid).collection('profile').document('info').set(profile_data)
        except: pass

    # 사용자 프로필 불러오기
    def load_profile(self):
        if not self.is_initialized or not st.session_state.user: return None
        try:
            uid = st.session_state.user['localId']
            doc = self.db.collection('users').document(uid).collection('profile').document('info').get()
            return doc.to_dict() if doc.exists else None
        except: return None

    # 채팅 세션 저장 (히스토리용)
    def save_chat_session(self, session_id, messages, summary):
        if not self.is_initialized or not st.session_state.user: return
        try:
            uid = st.session_state.user['localId']
            # 최근 20개 대화만 저장 (용량 최적화)
            save_data = [{"role": m["role"], "content": m["content"], "type": m.get("type", "text")} for m in messages[-20:]]
            self.db.collection('users').document(uid).collection('chat_sessions').document(session_id).set({
                "messages": save_data,
                "summary": summary,
                "updated_at": firestore.SERVER_TIMESTAMP
            }, merge=True)
        except: pass

    # 채팅 히스토리 목록 로드
    def load_chat_history_list(self):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            uid = st.session_state.user['localId']
            docs = self.db.collection('users').document(uid).collection('chat_sessions')\
                .order_by('updated_at', direction=firestore.Query.DESCENDING).limit(10).stream()
            return [{"id": d.id, **d.to_dict()} for d in docs]
        except: return []

    # 보관함(Bookmark) 저장
    def add_bookmark(self, type, content, note=""):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            uid = st.session_state.user['localId']
            self.db.collection('users').document(uid).collection('bookmarks').add({
                "type": type, "content": content, "note": note,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            return True
        except: return False
    
    # 보관함 로드
    def load_bookmarks(self):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            uid = st.session_state.user['localId']
            docs = self.db.collection('users').document(uid).collection('bookmarks')\
                .order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": d.id, **d.to_dict()} for d in docs]
        except: return []

fb_manager = FirebaseManager()

# -----------------------------------------------------------------------------
# [Session State] 초기화
# -----------------------------------------------------------------------------
if "user" not in st.session_state: st.session_state.user = None
if "current_chat" not in st.session_state: st.session_state.current_chat = []
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "major": "전자융합공학과", "grade": "1학년", "semester": "1학기", 
        "credit": 18, "requirements": "", "blocked_days": []
    }
if "grade_card_img" not in st.session_state: st.session_state.grade_card_img = []

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
# [AI Tools] 에이전트 도구
# -----------------------------------------------------------------------------
def get_llm(model_name="gemini-2.5-flash-preview-09-2025"):
    if not api_key: return None
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

# 1. 일반 질의응답 (RAG)
def tool_qa(query, profile):
    llm = get_llm()
    prompt = f"""
    [학생 정보] {profile['major']} {profile['grade']}
    [문서 내용] {PRE_LEARNED_DATA[:50000]}... (생략)
    [질문] {query}
    문서 내용을 바탕으로 답변해. 근거가 되는 문장은 " "로 인용해.
    """
    return llm.invoke(prompt).content

# 2. 시간표 생성 (검증 로직 이식 완료)
def tool_generate_timetable(major, grade, semester, credit, requirements, blocked_times):
    llm = get_llm()
    
    common_instruction = """
    [★★★ 핵심 알고리즘: 3단계 검증 및 필터링 (Strict Verification) ★★★]
    1. **Step 1: 요람(Curriculum) 기반 '수강 대상' 리스트 확정**:
       - 먼저 PDF 요람 문서에서 **'{major} {grade} {semester}'**에 배정된 **'표준 이수 과목' 목록**을 추출하세요.
    2. **Step 2: 학년 정합성 검사 (Grade Validation)**:
       - 추출된 과목이 실제 시간표 데이터에서 몇 학년 대상으로 개설되었는지 확인하세요.
       - **사용자가 선택한 학년({grade})과 시간표의 대상 학년이 일치하지 않으면 과감히 제외하세요.**
    3. **Step 3: 시간표 데이터와 정밀 대조 (Exact Match)**:
       - 과목명 완전 일치 필수 (예: '대학물리학1' vs '대학물리및실험1' 구분).
    
    [출력 형식: HTML Table]
    - 행: 1교시(09:00~) ~ 9교시
    - 열: 월~일 (7일)
    - 같은 과목 같은 배경색, 빈 시간 흰색.
    - **온라인/원격/시간미지정 과목은 표의 맨 아래 행에 포함하라.**
      (예: `<tr style='background-color:#eee;'><td colspan='8'><b>💻 온라인:</b> 과목명...</td></tr>`)
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

# 3. 졸업 진단 (멀티모달)
def tool_audit_graduation(profile, images_b64):
    if not images_b64:
        return "🎓 졸업 요건을 진단하려면 먼저 사이드바에서 **성적표 이미지를 업로드**해주세요!"
    
    llm = get_llm()
    image_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images_b64]
    
    prompt_text = f"""
    학생 정보: {profile['major']} {profile['grade']}
    업로드된 성적표를 분석하고 [학습된 요람]과 대조하여 졸업 요건을 진단해.
    종합 판정, 영역별 이수 현황(표), 미이수 과목, 조언 순으로 작성해.
    [학습된 요람] {PRE_LEARNED_DATA}
    """
    
    msg = HumanMessage(content=[{"type": "text", "text": prompt_text}] + image_content)
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
    
    예외: "다전공 설명하고 시간표 짜줘" 처럼 두 가지가 섞여 있으면, 
    논리적 순서에 따라 ["QA", "TIMETABLE"] 처럼 리스트로 반환하세요.
    
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
            st.session_state.clear() # 세션 초기화
            st.rerun()
    else:
        with st.expander("로그인 / 회원가입"):
            email = st.text_input("이메일")
            pw = st.text_input("비밀번호", type="password")
            col_l1, col_l2 = st.columns(2)
            if col_l1.button("로그인"):
                user, err = fb_manager.auth_user(email, pw, "login")
                if user: 
                    st.session_state.user = user
                    # 로그인 시 프로필 자동 로드
                    saved_profile = fb_manager.load_profile()
                    if saved_profile: st.session_state.user_profile.update(saved_profile)
                    st.rerun()
                else: st.error(err)
            if col_l2.button("회원가입"):
                user, err = fb_manager.auth_user(email, pw, "signup")
                if user:
                    st.session_state.user = user
                    st.rerun()
                else: st.error(err)

    st.divider()
    
    # 내 정보 (시간표 생성용)
    st.caption("📅 시간표 생성 설정")
    kw_depts = [
        "전자융합공학과", "전자공학과", "전자통신공학과", "전기공학과", "전자재료공학과", "로봇학부",
        "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "건축학과", "건축공학과", "화학공학과", "환경공학과",
        "국어국문학과", "영어영문학과", "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부",
        "행정학과", "법학부", "국제학부", "경영학부", "국제통상학부"
    ]
    
    # 세션 값으로 초기값 설정
    p = st.session_state.user_profile
    major = st.selectbox("학과", kw_depts, index=kw_depts.index(p["major"]) if p["major"] in kw_depts else 0, key="agent_major")
    col1, col2 = st.columns(2)
    grade = col1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], index=["1학년", "2학년", "3학년", "4학년"].index(p["grade"]), key="agent_grade")
    semester = col2.selectbox("학기", ["1학기", "2학기"], index=["1학기", "2학기"].index(p["semester"]), key="agent_sem")
    credit = st.number_input("목표 학점", 9, 24, p["credit"], key="agent_credit")
    reqs = st.text_area("추가 요구사항 (예: 오전 수업 X)", value=p["requirements"], key="agent_reqs")
    
    # 공강 설정
    with st.popover("공강 요일/시간 설정"):
        st.info("체크 해제 = 공강")
        days = ["월", "화", "수", "목", "금"]
        new_blocked = []
        cols = st.columns(5)
        for i, d in enumerate(days):
            is_checked = d not in p["blocked_days"]
            if not cols[i].checkbox(d, value=is_checked, key=f"chk_{d}"):
                new_blocked.append(d)
                
    # 정보 변경 시 자동 저장
    if st.button("설정 저장"):
        st.session_state.user_profile = {
            "major": major, "grade": grade, "semester": semester,
            "credit": credit, "requirements": reqs, "blocked_days": new_blocked
        }
        if st.session_state.user:
            fb_manager.save_profile(st.session_state.user_profile)
        st.success("저장됨!")
    
    st.divider()
    
    # 자료 제출 (졸업 진단용)
    st.caption("🎓 졸업 진단용 성적표")
    uploaded_imgs = st.file_uploader("성적표 캡처 업로드", type=['png', 'jpg'], accept_multiple_files=True)
    if uploaded_imgs:
        imgs_b64 = []
        for img in uploaded_imgs:
            img_bytes = img.read()
            imgs_b64.append(base64.b64encode(img_bytes).decode('utf-8'))
        st.session_state.grade_card_img = imgs_b64
        st.success(f"{len(imgs_b64)}장 업로드됨")

    st.divider()

    # 히스토리 & 보관함 탭
    tab1, tab2 = st.tabs(["🗂️ 히스토리", "⭐ 보관함"])
    
    with tab1:
        if st.session_state.user:
            history_list = fb_manager.load_chat_history_list()
            for h in history_list:
                date_str = h['updated_at'].strftime('%m/%d %H:%M') if h.get('updated_at') else ""
                if st.button(f"💬 {h.get('summary', '대화')} ({date_str})", key=h['id']):
                    st.session_state.current_chat = h['messages']
                    st.rerun()
        else:
            st.caption("로그인 시 기록됨")

    with tab2:
        if st.session_state.user:
            bookmarks = fb_manager.load_bookmarks()
            for b in bookmarks:
                with st.expander(f"📌 {b.get('note', '보관된 항목')}"):
                    if b['type'] == 'html':
                        st.markdown(b['content'], unsafe_allow_html=True)
                    else:
                        st.markdown(b['content'])
        else:
            st.caption("로그인 시 사용 가능")

# 2. 메인 채팅 인터페이스
st.title("🎓 KW-강의마스터 AI")
st.caption("무엇이든 말씀하세요. AI가 알아서 시간표를 짜거나 졸업 요건을 봐드립니다.")

# 대화 내용 출력
for msg in st.session_state.current_chat:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "html":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])
        
        # 보관함 저장 버튼
        if msg["role"] == "assistant" and st.session_state.user:
            btn_key = f"save_{hash(str(msg['content']))}" 
            if st.button("💾 저장", key=btn_key):
                note = "시간표" if msg.get("type") == "html" else "답변 내용"
                fb_manager.add_bookmark(msg.get("type", "text"), msg["content"], note)
                st.toast("보관함에 저장되었습니다!")

# 사용자 입력 처리
if prompt := st.chat_input("예: 1학년 시간표 짜줘, 졸업 가능한지 봐줘"):
    st.session_state.current_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI가 작업을 판단하고 있습니다..."):
            # 프로필 업데이트
            profile = st.session_state.user_profile
            
            # 의도 분류
            intent_str = route_intent(prompt)
            intents = []
            if "TIMETABLE" in intent_str: intents.append("TIMETABLE")
            if "GRADUATION" in intent_str: intents.append("GRADUATION")
            if "QA" in intent_str and "TIMETABLE" not in intent_str: intents.append("QA")
            if not intents: intents = ["CHAT"]
            
            # 순차 실행
            for intent in intents:
                response_content = ""
                response_type = "text"
                
                if intent == "QA":
                    response_content = tool_qa(prompt, profile)
                    st.markdown(response_content)
                    
                elif intent == "TIMETABLE":
                    st.info(f"📅 [{profile['major']} {profile['grade']}] 시간표 생성 중...")
                    blocked_str = ", ".join(profile['blocked_days']) + "요일 공강" if profile['blocked_days'] else "공강 없음"
                    
                    if st.session_state.timetable_data and ("수정" in prompt or "빼줘" in prompt):
                        reqs = profile['requirements'] + f" (수정 요청: {prompt})"
                    else:
                        reqs = profile['requirements']

                    html_table = tool_generate_timetable(
                        profile['major'], profile['grade'], profile['semester'],
                        profile['credit'], reqs, blocked_str
                    )
                    st.session_state.timetable_data = html_table
                    response_content = html_table
                    response_type = "html"
                    st.markdown(response_content, unsafe_allow_html=True)
                    
                elif intent == "GRADUATION":
                    if not st.session_state.grade_card_img:
                        response_content = "🎓 졸업 요건을 진단하려면 먼저 왼쪽 사이드바에 **성적표 이미지를 업로드**해주세요!"
                        st.warning(response_content)
                    else:
                        st.info("🎓 성적표를 분석하여 졸업 요건을 진단합니다...")
                        report = tool_audit_graduation(profile, st.session_state.grade_card_img)
                        st.session_state.graduation_data = report
                        response_content = report
                        st.markdown(response_content)
                
                elif intent == "CHAT":
                    llm = get_llm()
                    response_content = llm.invoke(f"사용자: {prompt}\n친절한 학사 조교로서 답변해.").content
                    st.markdown(response_content)
                
                st.session_state.current_chat.append({"role": "assistant", "content": response_content, "type": response_type})
    
    # 자동 저장
    if st.session_state.user:
        fb_manager.save_chat_session(st.session_state.session_id, st.session_state.current_chat, summary=prompt[:15]+"...")
    
    st.rerun()

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
import ast
import re
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Firebase 라이브러리
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 유틸리티
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-AI Agent", page_icon="🤖", layout="wide")

# CSS: 모바일 최적화 및 UI 개선
st.markdown("""
    <style>
        footer { visibility: hidden; }
        @media only screen and (max-width: 600px) {
            .main .block-container {
                padding: 2rem 0.5rem !important;
                max-width: 100% !important;
            }
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

# API Key 검증
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🚨 Google API Key 설정이 필요합니다.")
    st.stop()

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

# 재시도 로직 (429 에러 대응 - 즉시 알림)
def run_with_retry(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 현재 AI 요청량이 많아 처리가 불가능합니다. 잠시 후(약 1분 뒤) 다시 질문해 주세요."
        raise e

# -----------------------------------------------------------------------------
# [Firebase Manager] (Identity Toolkit 제거 -> Firestore 직접 인증)
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

    # [수정됨] Firestore를 이용한 자체 간편 인증 (API 키 필요 없음)
    def auth_user(self, email, password, mode="login"):
        if not self.is_initialized:
            return None, "Firebase DB가 연결되지 않았습니다."
        
        # 이메일을 문서 ID로 사용하기 위해 특수문자 처리
        user_id = email.replace("@", "_at_").replace(".", "_dot_")
        doc_ref = self.db.collection('users').document(user_id)

        try:
            doc = doc_ref.get()
            
            if mode == "signup":
                if doc.exists:
                    return None, "이미 존재하는 이메일입니다."
                # 회원가입: 비밀번호 저장 (실제 서비스에선 해싱 필요하지만 여기선 평문 저장)
                doc_ref.set({
                    "password": password, 
                    "email": email, 
                    "created_at": firestore.SERVER_TIMESTAMP
                })
                return {"localId": user_id, "email": email}, None
            
            elif mode == "login":
                if not doc.exists:
                    return None, "존재하지 않는 사용자입니다."
                user_data = doc.to_dict()
                if user_data.get("password") == password:
                    return {"localId": user_id, "email": email}, None
                else:
                    return None, "비밀번호가 틀렸습니다."
        except Exception as e:
            return None, str(e)

    def save_profile(self, profile_data, imgs_b64):
        if self.is_initialized and st.session_state.user:
            try:
                uid = st.session_state.user['localId']
                data = profile_data.copy()
                if imgs_b64:
                    data['grade_card_img'] = imgs_b64 
                self.db.collection('users').document(uid).collection('profile').document('info').set(data)
                return True
            except: return False
        return False

    def load_profile(self):
        if self.is_initialized and st.session_state.user:
            try:
                uid = st.session_state.user['localId']
                doc = self.db.collection('users').document(uid).collection('profile').document('info').get()
                return doc.to_dict() if doc.exists else None
            except: return None
        return None

    def save_chat_session(self, session_id, messages, summary):
        if self.is_initialized and st.session_state.user:
            try:
                uid = st.session_state.user['localId']
                save_data = [{"role": m["role"], "content": m["content"], "type": m.get("type", "text")} for m in messages[-20:]]
                self.db.collection('users').document(uid).collection('chat_sessions').document(session_id).set({
                    "messages": save_data, "summary": summary, "updated_at": firestore.SERVER_TIMESTAMP
                }, merge=True)
            except: pass

    def load_chat_history_list(self):
        if self.is_initialized and st.session_state.user:
            try:
                uid = st.session_state.user['localId']
                docs = self.db.collection('users').document(uid).collection('chat_sessions')\
                    .order_by('updated_at', direction=firestore.Query.DESCENDING).limit(10).stream()
                return [{"id": d.id, **d.to_dict()} for d in docs]
            except: return []
        return []

    def add_bookmark(self, type, content, note=""):
        if self.is_initialized and st.session_state.user:
            try:
                uid = st.session_state.user['localId']
                self.db.collection('users').document(uid).collection('bookmarks').add({
                    "type": type, "content": content, "note": note, "created_at": firestore.SERVER_TIMESTAMP
                })
                return True
            except: return False
        return False

    def load_bookmarks(self):
        if self.is_initialized and st.session_state.user:
            try:
                uid = st.session_state.user['localId']
                docs = self.db.collection('users').document(uid).collection('bookmarks')\
                    .order_by('created_at', direction=firestore.Query.DESCENDING).stream()
                return [{"id": d.id, **d.to_dict()} for d in docs]
            except: return []
        return []

fb_manager = FirebaseManager()

# -----------------------------------------------------------------------------
# [Session & Data]
# -----------------------------------------------------------------------------
if "user" not in st.session_state: st.session_state.user = None
if "current_chat" not in st.session_state: st.session_state.current_chat = []
if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())

# 초기값을 빈 값으로 설정
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "major": "선택해주세요", "grade": "선택해주세요", "semester": "선택해주세요", 
        "credit": 19, "requirements": "", "blocked_days": []
    }
if "grade_card_img" not in st.session_state: st.session_state.grade_card_img = []
if "timetable_data" not in st.session_state: st.session_state.timetable_data = ""
if "graduation_data" not in st.session_state: st.session_state.graduation_data = ""

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
# [AI Tools] 에이전트 도구 (Throttling 적용)
# -----------------------------------------------------------------------------
def get_llm(model_name="gemini-2.5-flash-preview-09-2025"):
    if not api_key: return None
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

# 1. QA
def tool_qa(query, profile):
    time.sleep(2) # [Throttling] 강제 휴식
    llm = get_llm()
    prompt = f"""
    [학생 정보] {profile['major']} {profile['grade']}
    [문서] {PRE_LEARNED_DATA[:50000]}...
    [질문] {query}
    문서 내용을 바탕으로 답변해. 근거 문장은 " "로 인용해.
    """
    return run_with_retry(lambda: llm.invoke(prompt).content)

# 2. 시간표 생성
def tool_generate_timetable(profile, extra_req=""):
    time.sleep(2) # [Throttling] 강제 휴식
    llm = get_llm()
    blocked = ", ".join(profile['blocked_days']) + "요일" if profile['blocked_days'] else "없음"
    
    instruction = """
    [★★★ 핵심 알고리즘: 3단계 검증 (Strict Verification) ★★★]
    1. **Step 1:** 요람에서 '{major} {grade} {semester}' 필수 과목 추출.
    2. **Step 2 (학년 검증):** 시간표 데이터에서 해당 과목의 대상 학년이 '{grade}'와 일치하는지 확인. 불일치 시 제외.
    3. **Step 3 (정밀 대조):** 과목명이 정확히 일치하는 시간표만 사용.
    
    [출력 형식: HTML Table]
    - 행: 1교시(09:00~) ~ 9교시
    - 열: 월~일 (7일)
    - 같은 과목 같은 배경색.
    - **온라인/시간미지정 과목은 표의 맨 아래 행에 포함** (colspan 사용).
    """
    
    prompt = f"""
    전문가로서 시간표를 생성해.
    정보: {profile['major']} {profile['grade']} {profile['semester']}, 목표 {profile['credit']}학점.
    공강 요청: {blocked}. 추가요구: {profile['requirements']} {extra_req}.
    {instruction}
    [데이터] {PRE_LEARNED_DATA}
    """
    res = run_with_retry(lambda: llm.invoke(prompt).content)
    return clean_html_output(res)

# 3. 졸업 진단
def tool_audit_graduation(profile, images_b64):
    if not images_b64:
        return "⚠️ 저장된 성적표 이미지가 없습니다. 사이드바에서 업로드해주세요."
    
    time.sleep(2) # [Throttling] 강제 휴식
    llm = get_llm()
    image_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images_b64]
    
    prompt_text = f"""
    학생: {profile['major']} {profile['grade']}
    성적표 이미지를 분석해 [학습된 요람]과 대조하여 졸업 요건을 진단해.
    종합 판정, 이수 현황(표), 미이수 과목, 조언 순서로 작성.
    [요람] {PRE_LEARNED_DATA}
    """
    
    msg = HumanMessage(content=[{"type": "text", "text": prompt_text}] + image_content)
    return run_with_retry(lambda: llm.invoke([msg]).content)

# 4. [최적화] 키워드 기반 라우팅 (API 호출 0회)
def decide_intent_rule_based(user_input):
    intents = []
    text = user_input.replace(" ", "") # 띄어쓰기 무시
    
    # 키워드 사전
    kw_timetable = ["시간표", "짜줘", "만들어", "수정", "빼줘", "넣어줘"]
    kw_graduation = ["졸업", "학점", "이수", "요건", "진단", "성적"]
    kw_qa = ["규정", "장학", "재수강", "설명", "알려줘", "뭐야", "?", "기준"]
    
    # 1. 졸업 진단
    if any(k in text for k in kw_graduation):
        intents.append("GRADUATION")
    
    # 2. 시간표 (졸업 진단과 함께 요청 가능)
    if any(k in text for k in kw_timetable):
        intents.append("TIMETABLE")
        
    # 3. QA (설명 요청이 포함된 경우)
    if any(k in text for k in kw_qa):
        # 시간표나 졸업과 같이 묻는 경우 QA를 먼저 수행
        if "TIMETABLE" in intents or "GRADUATION" in intents:
            if "설명" in text or "규정" in text:
                intents.insert(0, "QA")
        else:
            intents.append("QA")
            
    # 아무것도 없으면 잡담
    if not intents:
        intents.append("CHAT")
        
    return list(dict.fromkeys(intents)) # 중복 제거

# -----------------------------------------------------------------------------
# [UI] 사이드바 및 메인
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 내 학사 프로필")
    
    # 로그인
    if st.session_state.user:
        st.success(f"**{st.session_state.user['email']}**님")
        # 로그아웃 시 확실한 초기화
        if st.button("로그아웃", use_container_width=True):
            st.session_state.user = None
            st.session_state.clear()
            st.rerun()
    else:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            email = st.text_input("이메일")
            pw = st.text_input("비밀번호", type="password")
            col_l1, col_l2 = st.columns(2)
            if col_l1.button("로그인"):
                user, err = fb_manager.auth_user(email, pw, "login")
                if user:
                    st.session_state.user = user
                    saved = fb_manager.load_profile()
                    if saved:
                        st.session_state.user_profile.update(saved)
                        if 'grade_card_img' in saved:
                            st.session_state.grade_card_img = saved['grade_card_img']
                        
                        # 위젯 키값 업데이트
                        if "major" in saved: st.session_state.agent_major = saved["major"]
                        if "grade" in saved: st.session_state.agent_grade = saved["grade"]
                        if "semester" in saved: st.session_state.agent_sem = saved["semester"]
                        if "credit" in saved: st.session_state.agent_credit = saved["credit"]
                        if "requirements" in saved: st.session_state.agent_reqs = saved["requirements"]
                        
                        blocked_days = saved.get("blocked_days", [])
                        for d in ["월", "화", "수", "목", "금"]:
                            st.session_state[f"chk_{d}"] = (d not in blocked_days)
                            
                    st.rerun()
                else: st.error(err)
            if col_l2.button("가입"):
                user, err = fb_manager.auth_user(email, pw, "signup")
                if user:
                    st.session_state.user = user
                    st.rerun()
                else: st.error(err)

    st.divider()
    
    # 내 정보 설정
    st.subheader("📝 내 학사 정보 설정")
    st.caption("이 정보는 시간표, 졸업진단, 질문 답변 시 AI가 참고합니다.")
    
    kw_depts = ["선택해주세요", "전자융합공학과", "전자공학과", "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "경영학부"]
    
    p = st.session_state.user_profile
    
    major_idx = kw_depts.index(p["major"]) if p["major"] in kw_depts else 0
    major = st.selectbox("학과", kw_depts, index=major_idx, key="agent_major")
    
    c1, c2 = st.columns(2)
    grades = ["선택해주세요", "1학년", "2학년", "3학년", "4학년"]
    semesters = ["선택해주세요", "1학기", "2학기"]
    
    grade_idx = grades.index(p["grade"]) if p["grade"] in grades else 0
    sem_idx = semesters.index(p["semester"]) if p["semester"] in semesters else 0
    
    grade = st.selectbox("학년", grades, index=grade_idx, key="agent_grade")
    semester = st.selectbox("학기", semesters, index=sem_idx, key="agent_sem")
    
    credit = st.number_input("목표 학점", 0, 24, p["credit"], key="agent_credit")
    reqs = st.text_area("요구사항", value=p["requirements"], key="agent_reqs")
    
    with st.popover("공강 요일 설정"):
        days = ["월", "화", "수", "목", "금"]
        new_blocked = []
        cols = st.columns(5)
        for i, d in enumerate(days):
            is_checked = d not in p["blocked_days"]
            if not cols[i].checkbox(d, value=is_checked, key=f"chk_{d}"):
                new_blocked.append(d)
                
    if st.button("설정 저장"):
        st.session_state.user_profile = {
            "major": major, "grade": grade, "semester": semester,
            "credit": credit, "requirements": reqs, "blocked_days": new_blocked
        }
        if st.session_state.user:
            fb_manager.save_profile(st.session_state.user_profile, st.session_state.grade_card_img)
        st.success("저장됨!")
    
    st.divider()
    
    # 성적표
    st.subheader("📄 성적표")
    if st.session_state.grade_card_img:
        st.info(f"✅ {len(st.session_state.grade_card_img)}장의 성적표가 저장되어 있습니다.")
    else:
        st.caption("저장된 성적표가 없습니다.")

    uploaded_imgs = st.file_uploader("새로 업로드 (기존 파일 덮어씀)", type=['png', 'jpg'], accept_multiple_files=True)
    if uploaded_imgs:
        imgs_b64 = []
        for img in uploaded_imgs:
            img_bytes = img.read()
            imgs_b64.append(base64.b64encode(img_bytes).decode('utf-8'))
        st.session_state.grade_card_img = imgs_b64
        st.success("업로드 완료! (설정 저장을 눌러주세요)")

    st.divider()

    # 히스토리 & 보관함
    tab1, tab2 = st.tabs(["🗂️ 히스토리", "⭐ 보관함"])
    with tab1:
        if st.session_state.user:
            for h in fb_manager.load_chat_history_list():
                dt = h['updated_at'].strftime('%m/%d %H:%M') if h.get('updated_at') else ""
                if st.button(f"💬 {h.get('summary', '대화')} ({dt})", key=h['id']):
                    st.session_state.current_chat = h['messages']
                    st.rerun()
        else: st.caption("로그인 필요")
        
    with tab2:
        if st.session_state.user:
            for b in fb_manager.load_bookmarks():
                with st.expander(f"📌 {b.get('note', '항목')}"):
                    if b['type'] == 'html': st.markdown(b['content'], unsafe_allow_html=True)
                    else: st.markdown(b['content'])
        else: st.caption("로그인 필요")

# -----------------------------------------------------------------------------
# [Main] 채팅 인터페이스
# -----------------------------------------------------------------------------
st.title("🎓 KW-강의마스터 AI")

# 초기값 미설정 시 블라인드 처리
profile = st.session_state.user_profile
if profile["major"] == "선택해주세요" or profile["grade"] == "선택해주세요":
    st.warning("👈 왼쪽 사이드바에서 **학과와 학년**을 먼저 설정해주세요.")
    st.info("로그인하시면 저장된 설정을 불러올 수 있습니다.")
else:
    st.caption(f"**{profile['major']} {profile['grade']}**님, 무엇을 도와드릴까요?")

    # 대화 내용 출력
    for msg in st.session_state.current_chat:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "html": st.markdown(msg["content"], unsafe_allow_html=True)
            else: st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and st.session_state.user:
                k = f"save_{hash(str(msg['content']))}"
                if st.button("💾 저장", key=k):
                    note = "시간표" if msg.get("type") == "html" else "답변"
                    fb_manager.add_bookmark(msg.get("type", "text"), msg["content"], note)
                    st.toast("저장됨!")

    # 사용자 입력 처리
    if prompt := st.chat_input("예: 1학년 시간표 짜줘, 졸업 요건 봐줘"):
        st.session_state.current_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            # 에이전트 사고 과정 시각화 (Status Container)
            with st.status("🤖 AI가 작업을 계획하고 있습니다...", expanded=True) as status:
                
                st.write("🔍 사용자의 의도를 분석 중입니다...")
                intents = decide_intent_rule_based(prompt)
                st.write(f"👉 작업 분류: {intents}")
                
                for intent in intents:
                    res_con, res_type = "", "text"
                    
                    if intent == "QA":
                        st.write("📚 문서를 검색하고 있습니다...")
                        res_con = tool_qa(prompt, profile)
                        
                    elif intent == "TIMETABLE":
                        st.write("📅 시간표를 생성하고 있습니다...")
                        extra = prompt if "수정" in prompt or "빼줘" in prompt else ""
                        res_con = tool_generate_timetable(profile, extra)
                        res_type = "html"
                        
                    elif intent == "GRADUATION":
                        st.write("🎓 성적표를 분석하고 있습니다...")
                        if not st.session_state.grade_card_img:
                             res_con = "⚠️ 성적표 이미지가 없습니다. 사이드바에서 업로드해주세요."
                        else:
                            res_con = tool_audit_graduation(profile, st.session_state.grade_card_img)
                            
                    else: # CHAT
                        st.write("💬 답변을 작성 중입니다...")
                        llm = get_llm()
                        res_con = run_with_retry(lambda: llm.invoke(f"사용자: {prompt}\n친절한 학사 조교로서 답변해.").content)
                    
                    # 상태창 업데이트 완료
                    status.update(label="완료!", state="complete", expanded=False)
                    
                    if res_type == "html": st.markdown(res_con, unsafe_allow_html=True)
                    else: st.markdown(res_con)
                    
                    st.session_state.current_chat.append({"role": "assistant", "content": res_con, "type": res_type})
        
        # 자동 저장
        if st.session_state.user:
            fb_manager.save_chat_session(st.session_state.session_id, st.session_state.current_chat, summary=prompt[:15])
        
        st.rerun()

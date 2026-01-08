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
from langchain_core.messages import HumanMessage

# Firebase 라이브러리
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-AI Agent", page_icon="🤖", layout="wide")

# 모바일 최적화 CSS
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
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key}"
        try:
            res = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True})
            data = res.json()
            if "error" in data: return None, data["error"]["message"]
            return data, None
        except Exception as e: return None, str(e)

    # 사용자 프로필(학과, 학년 등) 저장/로드
    def save_profile(self, profile_data):
        if not self.is_initialized or not st.session_state.user: return
        try:
            uid = st.session_state.user['localId']
            self.db.collection('users').document(uid).collection('profile').document('info').set(profile_data)
        except: pass

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
            self.db.collection('users').document(uid).collection('chat_sessions').document(session_id).set({
                "messages": messages,
                "summary": summary,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
        except: pass

    def load_chat_history_list(self):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            uid = st.session_state.user['localId']
            docs = self.db.collection('users').document(uid).collection('chat_sessions')\
                .order_by('updated_at', direction=firestore.Query.DESCENDING).limit(10).stream()
            return [{"id": d.id, **d.to_dict()} for d in docs]
        except: return []

    # 보관함(Bookmark) 저장/로드
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
# 졸업 진단용 성적표 이미지 (Base64)
if "grade_card_img" not in st.session_state: st.session_state.grade_card_img = []

# HTML 정제 함수
def clean_html_output(text):
    cleaned = text.strip()
    if cleaned.startswith("

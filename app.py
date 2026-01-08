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
# [0] 설정 및 데이터 로드 (기본 설정 유지)
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

# 세션 초기화 (에이전트용으로 단순화)
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
    if cleaned.startswith("

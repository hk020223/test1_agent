import streamlit as st
import pandas as pd
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [1] 서버 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# 지식 베이스 로딩 함수 (data 폴더의 모든 PDF 읽기)
@st.cache_resource(show_spinner="학교 정보를 학습하는 중입니다... (약 1분 소요)")
def load_knowledge_base():
    all_content = ""
    
    # 'data' 폴더가 없으면 생성 (에러 방지용)
    if not os.path.exists("data"):
        os.makedirs("data")
        return ""

    # data 폴더 안의 모든 .pdf 파일 찾기
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        return ""

    # 각 PDF 파일을 순서대로 읽어서 텍스트 합치기
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            
            # 파일명을 헤더로 추가해서 AI가 출처를 알게 함
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서 시작: {filename}] ---\n"
            
            for page in pages:
                all_content += page.page_content
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
            
    return all_content

# 앱 시작 시 한 번만 실행되어 모든 PDF를 메모리에 올림
PRE_LEARNED_DATA = load_knowledge_base()

# 강의 데이터베이스 (시간표용 - 이전과 동일)
@st.cache_data
def load_course_db():
    return pd.DataFrame([
        {"과목명": "인공지능기초", "교수": "김교수", "시간": "월1,2,3", "영역": "전공", "과제비중": 40, "시험비중": 60, "팀플": "유"},
        {"과목명": "전자회로1", "교수": "이교수", "시간": "화4,5,6", "영역": "전공", "과제비중": 20, "시험비중": 80, "팀플": "무"},
        {"과목명": "데이터베이스", "교수": "최교수", "시간": "목4,5,6", "영역": "전공", "과제비중": 30, "시험비중": 70, "팀플": "유"},
        {"과목명": "광운인성", "교수": "정교수", "시간": "금1,2", "영역": "교양", "과제비중": 10, "시험비중": 90, "팀플": "무"},
        {"과목명": "대학영어", "교수": "Brown", "시간": "월7,8", "영역": "교양", "과제비중": 30, "시험비중": 70, "팀플": "유"}
    ])

course_db = load_course_db()

# -----------------------------------------------------------------------------
# [2] AI 엔진
# -----------------------------------------------------------------------------
def ask_ai(question):
    if not api_key:
        return "⚠️ 서버에 API Key가 설정되지 않았습니다. (Render Settings 확인)"
    
    if not PRE_LEARNED_DATA: 
        return "⚠️ 학습된 데이터가 없습니다. VS Code의 'data' 폴더에 PDF 파일을 넣어주세요."

    try:
        # 정보가 많으므로 temperature를 0으로 설정하여 팩트 위주 답변
        # 수정: 모델명을 'gemini-1.5-flash-latest'로 변경하여 인식 오류 해결 시도
        # 만약 여전히 안 된다면 'gemini-pro'로 변경해보세요.
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        
        template = """
        너는 광운대학교 학사 전문 상담 비서 'KW-강의마스터'야.
        너는 아래 제공된 [학습된 PDF 문서들]의 내용을 완벽하게 숙지하고 있어.
        
        [지시사항]
        1. 질문에 대한 답변은 오직 제공된 문서 내용에 기반해서 작성해.
        2. 답변할 때 "참고한 문서의 이름(예: 장학금규정.pdf)"을 언급해주면 더 좋아.
        3. 문서에 없는 내용은 솔직하게 모른다고 답해.

        [학습된 PDF 문서들]
        {context}

        [학생의 질문]
        {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | llm
        response = chain.invoke({"context": PRE_LEARNED_DATA, "question": question})
        return response.content
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"

# -----------------------------------------------------------------------------
# [3] UI 구성
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 KW-강의마스터")
# glob 모듈이 없는 경우 대비
try:
    pdf_count = len(glob.glob("data/*.pdf"))
except:
    pdf_count = 0
st.sidebar.info(f"📚 현재 {pdf_count}개의 문서를 학습했습니다.")

menu = st.sidebar.radio("메뉴", ["AI 학사 지식인", "이수학점 진단", "스마트 시간표"])

if menu == "AI 학사 지식인":
    st.header("🤖 AI 학사 지식인")
    st.caption("업로드된 PDF 문서들을 기반으로 답변합니다.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("질문하세요 (예: 이번 학기 장학금 기준이 뭐야?)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("문서를 검색 중입니다..."):
                answer = ask_ai(user_input)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif menu == "이수학점 진단":
    st.header("📊 졸업 이수 현황")
    col1, col2 = st.columns(2)
    with col1:
        major = st.number_input("전공 이수 학점", 0, 130, 45)
        ge = st.number_input("교양 이수 학점", 0, 130, 20)
    with col2:
        total = major + ge
        st.metric("현재 총 이수", f"{total} / 130")
        st.progress(total/130)

elif menu == "스마트 시간표":
    st.header("📅 시간표 자동 생성")
    if st.button("공강 고려 시간표 추천받기"):
        res = course_db.sample(3)
        st.table(res[['과목명', '교수', '시간', '영역']])
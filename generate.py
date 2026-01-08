import os
import glob
from langchain_community.document_loaders import PyPDFLoader

def generate_cache():
    print("🔄 PDF 문서를 텍스트로 변환(학습) 중입니다...")
    
    # 데이터 폴더 확인
    if not os.path.exists("data"):
        print("❌ 'data' 폴더가 없습니다. PDF 파일을 data 폴더에 넣어주세요.")
        return

    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        print("❌ 'data' 폴더에 PDF 파일이 없습니다.")
        return

    all_content = ""
    # 모든 PDF 읽기
    for pdf_file in pdf_files:
        try:
            print(f"   - 읽는 중: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n"
            for page in pages:
                all_content += page.page_content
        except Exception as e:
            print(f"⚠️ 에러 발생 ({pdf_file}): {e}")
            continue

    # 결과 저장
    cache_path = "data/cached_knowledge.txt"
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(all_content)
    
    print(f"\n✅ 학습 완료! '{cache_path}' 파일이 생성되었습니다.")
    print("🚀 이제 이 파일(cached_knowledge.txt)을 GitHub에 함께 올리면, 웹사이트가 즉시 로딩됩니다.")

if __name__ == "__main__":
    generate_cache()
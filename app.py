import streamlit as st # 웹 화면을 만드는 라이브러리
import pandas as pd # 데이터(CSV)를 다루는 라이브러리
import os # 파일 경로 등을 관리하는 기본 기능
from dotenv import load_dotenv # 환경변수(.env 파일)를 불러오는 기능
# LangChain 라이브러리 - OpenAI 모델 및 벡터 검색 관련 기능
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma # 문서 저장소 (벡터 DB)
from langchain.text_splitter import RecursiveCharacterTextSplitter # 긴 글을 작은 조각으로 나누는 기능
from langchain.prompts import PromptTemplate # 질문 양식을 미리 정의하는 기능 (사용되지 않음)

# LCEL관련 기능
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # 파이프라인(순서)을 만드는 기능
from langchain_core.output_parsers import StrOutputParser # 모델 출력을 단순 텍스트로 변환
from langchain.chains import create_history_aware_retriever, create_retrieval_chain # 대화형 검색 체인 제작 도구
from langchain.prompts import ChatPromptTemplate # 대화형 질문 양식 정의
from langchain_community.chat_message_histories import StreamlitChatMessageHistory # 스트림릿 세션에 대화 기록 저장

# 💡 환경변수 API 키 불러오기 (OPENAI_API_KEY 설정 필수!)
load_dotenv() # .env 파일 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # API 키 가져오기

if not OPENAI_API_KEY:
    st.error("❗ OPENAI_API_KEY 환경변수가 설정되어 있지 않아요! .env 파일을 꼭 확인해주세요.")
    st.stop() # 키 없으면 앱 중단

st.set_page_config(page_title="심플 병원 상담 챗봇", page_icon="💬", layout="wide") # 웹페이지 기본 설정
st.markdown("<h2 style='text-align:center;'>🏥 병원 문의 도우미 (LCEL 버전)</h2>", unsafe_allow_html=True) # 제목 표시

# CSV 파일 경로 설정 
CSV_PATH = os.path.join("data", "병원_validation.csv")

# LCEL에서는 StreamlitChatMessageHistory를 사용하여 st.session_state와 메모리를 연결. 
msgs = StreamlitChatMessageHistory(key="messages") # 스트림릿에 대화 기록 저장

if len(msgs.messages) == 0: # 대화 기록이 없으면
    # 챗봇의 첫 인사말과 질문 예시 추가
    initial_greeting = """
안녕하세요! 저는 병원 상담 챗봇 래미입니다. 😊  
궁금한 점 있으면 편하게 말씀해주세요!  

**예를 들면, 이렇게 질문하시면 더 정확한 답변을 얻을 수 있을 거예요:** 
💊 "저는 고혈압 약을 복용 중인데, 약 복용 시 주의할 점이 있을까요?"  
🩺 "현재 당뇨 치료를 받고 있는데, 혈당 검사 결과를 어떻게 해석해야 하나요?"  
💬 "임신 중인데, 진료 예약은 어떻게 하면 되나요?"  
📝 "최근 건강검진 결과 종합검진을 좀 더 자세히 받고 싶어요."  
⚕️ "보험 청구를 하려고 하는데, 필요한 서류와 절차가 궁금합니다."  
📅 "병원 접수 시간이 어떻게 되어 있는지 알려주세요."  

이 외에도 병원이나 건강에 대해 궁금한 모든 내용을 물어보시면 돼요! 🙌  
부담 가지지 말고, 어떤 질문도 환영합니다! 😊
"""
    msgs.add_ai_message(initial_greeting) # AI 메시지로 추가


# 📊 CSV 데이터 로드 (@st.cache_data로 앱 시작 시 딱 한 번만 실행)
@st.cache_data(show_spinner="CSV 데이터 로드 중...🌍")
def load_csv_data(path): # CSV 파일 로드 함수 정의
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip() # 컬럼명 공백 제거
        
        if "인텐트" not in df.columns or "발화문" not in df.columns: # 필수 컬럼 확인
            st.error("❌ CSV에 ‘인텐트’ 또는 ‘발화문’ 컬럼이 없어요! 확인해주세요.")
            st.stop()
        return df

    except Exception as e:
        st.error(f"❗ CSV 데이터 로드에 실패했어요: {e}")
        st.stop()

# CSV 데이터 로드
df = load_csv_data(CSV_PATH)


# 텍스트 분할 (청킹) 및 Document 형태로 변환 (@st.cache_data로 한 번만 실행!)
@st.cache_data(show_spinner="문서 처리 중...✂️")
def create_docs(dataframe): # 데이터프레임을 문서 조각으로 나누는 함수
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # 글자 500개 단위로 나눔
    docs_list = []

    for _, row in dataframe.iterrows():
        content = str(row["발화문"]).strip()
        if content:
            for chunk in text_splitter.split_text(content): # 분할된 조각을 리스트에 추가
                docs_list.append({
                    "page_content": chunk, # 내용
                    "metadata": {
                        "intent": row["인텐트"], # 원본 정보 (인텐트)
                    }
                })
    return docs_list

docs = create_docs(df)
if not docs:
    st.warning("CSV 파일에서 처리할 문서 내용을 찾을 수 없습니다. CSV 내용을 확인해주세요!")
    st.stop()


#벡터 DB 생성 및 리트리버 설정 (@st.cache_resource로 앱 실행 중 딱 한 번만 생성!)
persist_dir = os.path.join(os.getcwd(), "chroma_persist") # 벡터 DB 저장 폴더 경로 설정
os.makedirs(persist_dir, exist_ok=True)

@st.cache_resource(show_spinner="벡터 DB 생성 및 RAG 엔진 준비 중...🧠")
def setup_rag_components(documents, openai_key, persist_folder): # 벡터 DB 설정 함수
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key) # 문장을 숫자로 바꾸는 모델

    vectordb = Chroma.from_texts( # 문서 내용을 벡터 DB에 저장
        texts=[doc["page_content"] for doc in documents],
        embedding=embeddings,
        metadatas=[doc["metadata"] for doc in documents],
        persist_directory=persist_folder, # 저장 폴더 지정
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3}) # 검색기 (가장 유사한 문서 3개 찾기)
    return retriever

retriever = setup_rag_components(docs, OPENAI_API_KEY, persist_dir) # 벡터 DB 및 검색기 준비


# LangChain RAG 체인 설정 (LCEL 기반)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY) # 언어 모델 설정


# 1. 질문 재작성 프롬프트 (History Aware)
rephrase_prompt = ChatPromptTemplate.from_messages( # 프롬프트 양식 정의
    [
        ("system", "당신은 사용자 질문을 맥락을 고려하여 독립적인 검색 질문으로 재작성하는 AI입니다. 다음 대화 기록과 최신 질문을 고려하여, 검색 시스템이 이해할 수 있는 단일 질문으로 만드세요. 만약 대화 기록이 없다면, 원본 질문을 그대로 사용하세요."), # AI 역할 및 지침
        ("system", "대화 기록: \n{chat_history}"), # 대화 기록 변수
        ("human", "최신 질문: {input}"), # 최신 질문 변수
        ("human", "재작성된 질문:")
    ]
)

# 2. 질문 재작성 체인 (History Aware Retriever)
history_aware_retriever = create_history_aware_retriever( # 대화 기록을 고려한 검색기 생성
    llm, 
    retriever, 
    rephrase_prompt
)


# 3. 답변 생성 프롬프트 (RAG)
qa_system_prompt = (
    "당신은 '정확하고 신뢰할 수 있는 병원 정보'를 제공하는 전문 상담 챗봇입니다. "
    "사용자가 질문한 내용에 대해 다음 **'참고 문서'**를 우선적으로 활용하여 답변해야 합니다.\n\n"
    "### 답변 지침\n"
    "1.  **정확성 및 유연성:** 답변은 제공된 '참고 문서'를 기반으로 하되, 만약 참고 문서가 불충분할 경우 **사용자의 안전을 해치지 않는 선에서 일반적인 의료 상식**을 활용하여 친절하게 보충할 수 있습니다. **(수정된 부분)**\n"
    "2.  **맥락 및 친절함:** 이전 대화 흐름을 참고하여 친절하고 전문적인 어조로 답변을 구성하세요.\n"
    "3.  **전문적 어조:** 불필요한 이모티콘이나 과장된 표현을 피하고, 신뢰감을 주는 전문적인 용어를 사용하세요.\n"
    "4.  **정보 부족 시:** 만약 제공된 '참고 문서'와 일반 상식을 모두 활용해도 답변이 어렵다면, **'죄송하지만 해당 정보는 현재 제가 참고할 수 있는 자료 범위 밖입니다.'**라고 명확히 고지하고 답변을 마무리하세요.\n"
    "5.  **면책 고지 (필수):** 답변을 시작하거나 마무리할 때, **'본 답변은 참고용이며, 정확한 진단 및 처방은 반드시 전문의와 상담하십시오.'**라는 문구를 포함하세요.\n\n"
    "### 대화 기록\n{chat_history}\n\n"
    "### 참고 문서\n{context}\n\n"
    "### 사용자 질문\n{input}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{input}"), 
    ]
)


# 4. 최종 RAG 체인 (LCEL 출력 키를 'output'으로 맞추기 위함)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_prompt | llm | StrOutputParser()) 

# 'answer' 키의 값을 'output'으로 지정하여 메모리 저장 오류를 해결합니다.
final_output_chain = rag_chain | RunnablePassthrough.assign(
    output=lambda x: x['answer'] 
)

# 5. 메모리 관리 추가 (RunnableWithMessageHistory에 수정된 체인 사용)
from langchain_core.runnables.history import RunnableWithMessageHistory # 대화 기록을 관리하는 기능

rag_chain_with_history = RunnableWithMessageHistory( # 전체 체인에 대화 기록 기능 추가
    final_output_chain, 
    lambda session_id: msgs, # 스트림릿 메시지 기록을 메모리로 사용
    input_messages_key="input", # 사용자 질문을 'input'으로 받음
    history_messages_key="chat_history", # 대화 기록을 'chat_history'로 전달
)


# 🗣️ 대화창에 이전 대화들 출력
for msg in msgs.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content, unsafe_allow_html=True) # 화면에 대화 기록 표시


if user_prompt := st.chat_input("여기에 궁금한 점을 입력해주세요!"): # 사용자 질문 받기
    
    # 🌟🌟🌟 수정 1: 사용자 메시지를 즉시 화면에 표시 🌟🌟🌟
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 챗봇 답변 생성! 🤖
    with st.spinner("🤖 AI가 답변을 준비 중이에요... 잠시만 기다려 주세요!"):
        # LCEL 체인 실행!
        result = rag_chain_with_history.invoke(
            {"input": user_prompt},
            config={"configurable": {"session_id": "any_session_id"}} 
        )
        
        assistant_response = result['output'] # 최종 답변 추출 (키 오류 해결된 상태)
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response, unsafe_allow_html=True) # 챗봇 답변 표시
            
    # AI 응답 후 st.rerun()을 호출하여 즉시 화면 갱신을 하여 대화창이 계속 보이게 함. 
    st.rerun()


# 🔄 대화 초기화 버튼
if st.button("💬 대화 초기화"): # 초기화 버튼 클릭 시
    msgs.clear() # 대화 기록 삭제
    
    # 초기 인사말 다시 추가!
    initial_greeting_reset = """
안녕하세요! 저는 병원 상담 챗봇 래미입니다. 😊  
궁금한 점 있으면 편하게 말씀해주세요!  

**예를 들면, 이렇게 질문하시면 더 정확한 답변을 얻을 수 있을 거예요:** 💊 "저는 고혈압 약을 복용 중인데, 약 복용 시 주의할 점이 있을까요?"  
🩺 "현재 당뇨 치료를 받고 있는데, 혈당 검사 결과를 어떻게 해석해야 하나요?"  
💬 "임신 중인데, 진료 예약은 어떻게 하면 되나요?"  
📝 "최근 건강검진 결과 종합검진을 좀 더 자세히 받고 싶어요."  
⚕️ "보험 청구를 하려고 하는데, 필요한 서류와 절차가 궁금합니다."  
📅 "병원 접수 시간이 어떻게 되어 있는지 알려주세요."  

이 외에도 병원이나 건강에 대해 궁금한 모든 내용을 물어보시면 돼요! 🙌  
부담 가지지 말고, 어떤 질문도 환영합니다! 😊
"""
    msgs.add_ai_message(initial_greeting_reset)
    st.rerun() # 앱을 처음 상태로 되돌림!
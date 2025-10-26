## 📄 README: 심플 병원 상담 챗봇 (LCEL 기반 RAG)

이 프로젝트는 \*\*LangChain Expression Language (LCEL)\*\*과 **Streamlit**을 활용하여 구축된 간단한 **검색 증강 생성(RAG)** 기반의 병원 상담 챗봇입니다. CSV 파일에 포함된 병원 데이터를 기반으로 사용자의 질문에 답변합니다.

-----

## 🚀 주요 기능 및 특징

1.  **LCEL 기반 RAG 구현:** 최신 LangChain 아키텍처(LCEL)를 사용하여 RAG 파이프라인을 모듈화하고 효율적으로 구성했습니다.
2.  **대화 기록 유지:** `StreamlitChatMessageHistory`를 사용하여 이전 대화의 **맥락을 기억**하고, 이를 바탕으로 후속 질문에 답변합니다. (History-Aware RAG)
3.  **벡터 데이터베이스 (Chroma):** CSV 데이터를 청킹(Chunking)하여 벡터화하고, 로컬 Chroma DB에 저장하여 빠른 문서 검색을 가능하게 합니다.
4.  **안정성 확보:** LCEL 키 불일치 및 Streamlit 렌더링 지연 등 개발 시 발생하는 주요 오류를 해결했습니다.
5.  **사용자 친화적인 인터페이스:** Streamlit을 사용하여 웹 기반의 챗봇 UI를 제공합니다.

-----

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 기술 / 라이브러리 | 용도 |
| :--- | :--- | :--- |
| **핵심 프레임워크** | LangChain (LCEL) | RAG 파이프라인 구축 및 체인 관리 |
| **LLM / 임베딩** | OpenAI (`gpt-4o-mini`, `OpenAIEmbeddings`) | 답변 생성 및 텍스트 벡터 변환 |
| **데이터베이스** | Chroma | 로컬 벡터 데이터베이스 (문서 저장) |
| **웹 인터페이스** | Streamlit | 챗봇 UI 및 세션 관리 |
| **데이터 처리** | Pandas | CSV 데이터 로드 및 전처리 |

-----

## ⚙️ 설치 및 실행 방법

### 1\. 환경 설정

프로젝트를 실행하기 전에 필요한 라이브러리와 환경 변수를 설정해야 합니다.

```bash
# 1. 필수 라이브러리 설치
pip install streamlit pandas python-dotenv langchain langchain-openai langchain-community
```

### 2\. OpenAI API 키 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 발급받은 OpenAI API 키를 다음과 같이 입력합니다.

```env
OPENAI_API_KEY="여기에_발급받은_API_키를_입력하세요"
```

### 3\. 데이터 준비

1.  프로젝트 루트에 **`data`** 폴더를 생성합니다.
2.  사용할 CSV 파일 (`병원_validation.csv` 등)을 `data` 폴더 안에 넣습니다. (CSV 파일에는 \*\*`인텐트`\*\*와 **`발화문`** 컬럼이 필수입니다.)

### 4\. 챗봇 실행

터미널에서 다음 명령어를 실행하여 Streamlit 앱을 시작합니다. (파일 이름이 `app.py`라고 가정합니다.)

```bash
streamlit run app.py
```

-----

## 💻 트러블 슈팅 및 LCEL 핵심 구현

이 프로젝트는 개발 중 발생할 수 있는 주요 오류를 해결하는 LCEL 로직을 포함합니다.

| 문제 현상 | 원인 및 기술적 충돌 | 해결에 사용된 LCEL 기술 |
| :--- | :--- | :--- |
| **대화 기록 소실 (`KeyError: 'output'`)** | `create_retrieval_chain`의 출력 키(`'answer'`)와 메모리 시스템(`RunnableWithMessageHistory`)이 기대하는 키(`'output'`)의 불일치. | `final_output_chain = rag_chain \| RunnablePassthrough.assign(output=lambda x: x['answer'])`를 통한 **출력 키 강제 재할당.** |
| **AI 응답 지연 (다음 질문 입력 후 응답 뜸)** | Streamlit의 렌더링 생명주기상, LLM 호출 후 화면 갱신이 지연됨. | AI 응답 출력 후 \*\*`st.rerun()`\*\*을 호출하여 Streamlit 화면 갱신을 강제하고 즉시 피드백 제공. |
import os
os.environ["PYTHONUTF8"] = "1"
import httpx
def _patched_normalize_header_value(value, encoding: str | None = None) -> bytes:
    if isinstance(value, bytes):
        return value  # Eğer zaten bytes ise, dokunma
    return value.encode(encoding or "utf-8") # Eğer string ise, utf-8 ile encode et

httpx._models._normalize_header_value = _patched_normalize_header_value

import textwrap
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

load_dotenv()

prompt_template = """Verilen bağlam parçalarını kullanarak sondaki soruyu yanıtlayın. 
Cevabı bilmiyorsanız, bilmediğinizi söyleyin, cevap uydurmaya çalışmayın.
Cevabınızı mutlaka kapsamlı ve açıklayıcı bir şekilde Türkçe olarak verin.

Bağlam:
{context}

Soru: {question}

Yardımcı Cevap (Türkçe):"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

@st.cache_resource
def create_qa_chain(_uploaded_file):

    if _uploaded_file is not None:
        temp_file_path = os.path.join("./", _uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(_uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        llm = ChatOpenAI(temperature=0, model_name='gpt-4o')

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        return qa_chain
    return None

st.set_page_config(page_title="PDF ile Sohbet", page_icon="📄")

st.title("📄 PDF Dosyanızla Sohbet Edin")
st.write("Bir PDF dosyası yükleyin ve içeriği hakkında sorular sorun.")

uploaded_file = st.file_uploader("Lütfen PDF dosyanızı buraya yükleyin", type="pdf")


if uploaded_file is not None:
    with st.spinner('PDF işleniyor, vektörler oluşturuluyor... Lütfen bekleyin.'):
        qa_chain = create_qa_chain(uploaded_file)
        st.success(f"**{uploaded_file.name}** başarıyla işlendi! Artık soru sorabilirsiniz.")


    question = st.text_input("PDF içeriği hakkında bir soru sorun:", placeholder="Örn: Yapay zekanın sıkıntıları nelerdir?")

    if question:
        with st.spinner('Cevap aranıyor...'):
            try:
                answer = qa_chain.invoke(question)
                result_text = answer.get('result', 'Cevap alınamadı.')

                st.markdown("### Cevap:")
                st.info(result_text)

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

st.sidebar.header("Nasıl Çalışır?")
st.sidebar.markdown("""
1.  **PDF Yükle:** `Gözat` butonu ile bilgisayarınızdan bir PDF dosyası seçin.
2.  **İşleme:** Uygulama, PDF'in metnini çıkarır, anlamsal parçalara ayırır ve LangChain ile bir vektör veritabanı oluşturur.
3.  **Soru Sor:** Metin kutusuna sorunuzu yazıp `Enter`'a basın.
4.  **Cevap Al:** Yapay zeka, sorunuza en uygun cevabı PDF içeriğinden bularak size sunar.
""")
st.sidebar.markdown("---")
st.sidebar.write("Ahmet Eren tarafından geliştirildi.")
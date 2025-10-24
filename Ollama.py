import os
import textwrap
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


prompt_template = """Verilen bağlam parçalarını kullanarak sondaki soruyu yanıtlayın. 
Cevabı bilmiyorsanız, bilmediğinizi söyleyin, cevap uydurmaya çalışmayın.
Cevabınızı mutlaka Türkçe olarak verin.

Bağlam:
{context}

Soru: {question}

Yardımcı Cevap (Türkçe):"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def setup_qa_system(file_path):
    # 1. PDF'i Yükle ve Böl
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    # 2. Metni Parçalara Ayır (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # 3. Embedding'leri Oluştur ve Vektör Veritabanına Kaydet
    embeddings = OllamaEmbeddings(model='llama3.1:latest')
    vector_store = FAISS.from_documents(chunks, embeddings)

    # 4. Retriever'ı Oluştur
    retriever = vector_store.as_retriever()

    # 5. LLM'i (Modeli) Tanımla
    llm = Ollama(model='llama3.1:latest')

    # 6. Soru-Cevap Zincirini (Chain) Oluştur
    # GÜNCELLENDİ: Zincire özel Türkçe prompt'umuzu ekliyoruz
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}  # Özel prompt'u burada tanımladık
    )
    return qa_chain

if __name__ == '__main__':
    qa_chain = setup_qa_system('deneme.pdf') # PDF dosyasının ismini bu .py dosyasıyla aynı klasörde olacak şekilde koyun ve ismini yazın

    print("PDF yüklendi. Soru sormaya başlayabilirsiniz. (Çıkmak için 'çıkış' veya 'exit' yazın)")

    # Terminal genişliğini al (varsayılan 80)
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 80

    while True:
        question = input('\nBir soru sorun: ')

        if question.lower() in ['exit', 'çıkış']:
            break

        # Zinciri çalıştır ve cevabı al
        answer = qa_chain.invoke(question)

        print('\nCevap:')

        # --- TAŞMAYI ÖNLEYEN KOD ---
        # Cevabı düz metin olarak al
        result_text = answer['result']

        # Cevabı satır satır böl (listelerdeki gibi boşlukları korumak için)
        lines = result_text.split('\n')

        for line in lines:

            wrapped_lines = textwrap.wrap(line, width=terminal_width - 5)  # Kenar boşluğu için 5 karakter kısalttık

            # Eğer satır boşsa (listeler arası boşluksa), boş satır bas
            if not wrapped_lines:
                print()
            else:
                # Bölünmüş satırları yazdır
                for wrapped_line in wrapped_lines:
                    print(wrapped_line)
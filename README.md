📄 PDF ile Soru-Cevap (RAG) Uygulaması
Bu proje, PDF dökümanları üzerinde Soru-Cevap (RAG - Retrieval-Augmented Generation) tekniğini kullanarak bir sistem sunar. Proje, iki farklı implementasyonu karşılaştırma amacıyla içerir:

main.py: Streamlit ile geliştirilmiş, OpenAI API'sini (gpt-4o) kullanan bir web uygulaması.

ollama.py: Tamamen yerel (local) olarak Ollama ve llama3.1 modelini kullanan terminal tabanlı bir uygulama.

Her iki uygulama da LangChain kütüphanesinden yararlanır, PDF içeriğini işler, FAISS ile vektör veritabanı oluşturur ve Türkçe sorulara Türkçe cevaplar üretir.

✨ Özellikler
PDF Yükleme: Streamlit arayüzü üzerinden dilediğiniz PDF'i yükleme (OpenAI versiyonu).

İki Farklı Model: Hem bulut tabanlı OpenAI (gpt-4o) hem de yerel Ollama (llama3.1:latest) ile RAG altyapısı.

Vektör Veritabanı: FAISS kullanarak dökümanlar için hızlı ve verimli vektör araması.

Türkçe Desteği: Modelleri Türkçe ve kapsamlı cevap vermeye yönlendiren özel prompt şablonları.

Arayüz Seçenekleri: Kullanıcı dostu web arayüzü (Streamlit) veya hızlı terminal uygulaması.

🚀 Kurulum ve Başlangıç
Bu projeyi çalıştırmak için aşağıdaki adımları izleyin.

1. Projeyi Klonlayın
Bash

# GitHub'dan projeyi klonlayın (Eğer GitHub'a attıysanız)
git clone <github-repo-linkiniz>
cd <proje-klasör-adı>
2. Sanal Ortam (venv) Oluşturun
Python bağımlılıklarını izole etmek için bir sanal ortam oluşturmanız şiddetle tavsiye edilir.

Bash

# Sanal ortamı oluştur
python -m venv venv

# Sanal ortamı aktifleştir (Windows)
.\venv\Scripts\activate

# (macOS/Linux)
# source venv/bin/activate
3. Gerekli Paketleri Yükleyin
Proje için gerekli tüm Python kütüphanelerini requirements.txt dosyasından yükleyin.

Bash

pip install -r requirements.txt
(Eğer requirements.txt dosyanız yoksa, şu komutla oluşturabilirsiniz: pip freeze > requirements.txt)

4. Gerekli Ayarlamalar
a) OpenAI için (main.py)
main.py dosyasının çalışması için bir OpenAI API anahtarına ihtiyacınız vardır.

Proje ana dizininde (main.py'nin yanında) .env adında bir dosya oluşturun.

Dosyanın içine API anahtarınızı aşağıdaki gibi ekleyin:

Kod snippet'i

OPENAI_API_KEY='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
b) Ollama için (ollama.py)
ollama.py dosyasının çalışması için sisteminizde Ollama'nın kurulu olması ve llama3.1 modelinin indirilmiş olması gerekir.

Resmi sitesinden Ollama'yı indirin ve kurun.

Kurulumdan sonra terminali açın ve modeli indirmek için aşağıdaki komutu çalıştırın:

Bash

ollama pull llama3.1:latest
💻 Kullanım
1. Streamlit Web Uygulaması (OpenAI)
Kullanıcı dostu arayüzü başlatmak için:

Terminalinizde (sanal ortam aktifken) aşağıdaki komutu çalıştırın:

Bash

streamlit run main.py
Otomatik olarak açılan tarayıcı sekmesinde "Gözat" (Browse files) butonuna tıklayarak bir PDF dosyası yükleyin.

PDF işlendikten sonra "PDF içeriği hakkında bir soru sorun:" kutusuna sorunuzu yazıp Enter'a basın.

2. Terminal Uygulaması (Ollama)
Yerel modelle çalışan terminal uygulamasını başlatmak için:

Soru sormak istediğiniz PDF dosyasını proje klasörüne atın ve adını deneme.pdf olarak değiştirin (veya ollama.py içindeki setup_qa_system('deneme.pdf') satırını kendi PDF dosya adınızla güncelleyin).

Terminalinizde (sanal ortam aktifken) aşağıdaki komutu çalıştırın:

Bash

python ollama.py
"PDF yüklendi. Soru sormaya başlayabilirsiniz." mesajını gördükten sonra sorularınızı terminale yazın.

Çıkmak için çıkış veya exit yazabilirsiniz.

🛠️ Kullanılan Teknolojiler
Python 3.10+

Streamlit: Web arayüzü için.

LangChain: RAG zincirlerini oluşturmak ve LLM'leri entegre etmek için ana kütüphane.

OpenAI: GPT-4o modeli ve embedding'ler için.

Ollama: Llama 3.1 modelini yerel olarak çalıştırmak için.

FAISS (faiss-cpu): Vektör depolama ve arama için.

PyPDFLoader: PDF dosyalarını yüklemek ve metne dönüştürmek için.

python-dotenv: API anahtarlarını güvenli bir şekilde yönetmek için.

import streamlit as st
import streamlit_authenticator as stauth
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import pytesseract
from PIL import Image
import tempfile
import pyttsx3
import whisper
import sounddevice as sd
import wavio
import os

# --- LOGIN SETUP ---
names = ["Alice", "Bob"]
usernames = ["alice", "bob"]
passwords = ["password1", "password2"]  # Replace with stronger passwords or hash in prod

hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords,
    "cookie_name", "signature_key", cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"Welcome *{name}*")

    st.title("📚 Secure Voice + Document Chatbot")

    # --- Setup LLM, DB, Embeddings ---
    persist_directory = "./db"
    embedding_model = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="llama3")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # --- Initialize TTS and STT ---
    tts_engine = pyttsx3.init()
    whisper_model = whisper.load_model("small")

    # --- Helper functions ---

    def process_pdf(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        st.success("PDF processed and added to memory!")

    def process_image(file_path):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        doc = Document(page_content=text)
        chunks = text_splitter.split_documents([doc])
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        st.success("Image processed with OCR and added to memory!")

    def record_audio(filename="audio.wav", duration=5, fs=16000):
        st.info("Recording audio for 5 seconds...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wavio.write(filename, recording, fs, sampwidth=2)
        st.success("Recording complete!")

    def speech_to_text(audio_path):
        result = whisper_model.transcribe(audio_path)
        return result["text"]

    def speak(text):
        tts_engine.say(text)
        tts_engine.runAndWait()

    # --- UI for upload ---
    uploaded_file = st.file_uploader("Upload PDF or Image to add knowledge", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if uploaded_file.type == "application/pdf":
            process_pdf(tmp_file_path)
        else:
            process_image(tmp_file_path)

    # --- UI for question input ---
    user_question = st.text_input("Ask your question here:")

    # --- Voice recording ---
    if st.button("🎤 Record question (5 seconds)"):
        record_audio()
        user_question = speech_to_text("audio.wav")
        st.text_area("Transcribed question:", user_question)

    # --- Responding ---
    if user_question:
        docs = retriever.get_relevant_documents(user_question)
        if docs:
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa_chain.run(user_question)
            st.markdown(f"**Answer (from your uploaded documents):** {answer}")
            speak(answer)
        else:
            prompt = f"I couldn't find the answer in the uploaded documents. But here is what I know:\n\nQuestion: {user_question}\nAnswer:"
            answer = llm(prompt)
            st.markdown(f"**Answer (fallback to LLaMA 3 knowledge):** {answer}")
            st.markdown("_Note: This answer is based on LLaMA 3's training data._")
            speak(answer)

elif authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")

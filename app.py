import streamlit as st
import os
import time
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import speech_recognition as sr

# ========== INITIAL SETUP ==========
# Initialize Groq client
client = Groq(api_key="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")  # REPLACE with st.secrets later

# Streamlit page config
st.set_page_config(page_title="HEC Pakistan Assistant", layout="wide")
st.title("Higher Education Commission (HEC) Pakistan Assistant")

# ========== NEW CHAT BUTTON ==========
if st.button("🧹 New Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# ========== CHAT HISTORY INIT ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== VOICE INPUT FUNCTION ==========
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"🗣️ You said: {query}")
        return query
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
    return None

# ========== DOCUMENT LOADING ==========
def load_documents():
    documents = []
    for file in os.listdir("./Data"):
        path = os.path.join("./Data", file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            try:
                loader = Docx2txtLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"⚠️ Skipping {file} due to error: {e}")
    return documents

# ========== TEXT SPLITTING ==========
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ========== EMBEDDING & VECTOR DB ==========
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    docs = load_documents()
    chunks = split_documents(docs)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    db.persist()
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# ========== GROQ RESPONSE ==========
def get_groq_response(query, context):
    prompt = f"""You are a professional virtual assistant for the Higher Education Commission (HEC), Pakistan.
    Your primary role is to use the provided context and respond with accurate, concise, and helpful information regarding higher
    programs, services, and policies offered by HEC. You should respond to user inquiries in a friendly yet formal manner,
    ensuring clarity and professionalism. If you cannot answer the question based on the context, politely state that you don't have
    enough information to answer accurately and suggest contacting HEC directly for more specific details.

    Context: {context}

    Question: {query}

    Answer:"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.5,
        stream=True,
    )
    return response

# ========== CHAT INTERFACE ==========
st.subheader("💬 Ask your question below (text or voice)")

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.chat_input("Your query for the HEC Assistant:")

with col2:
    if st.button("🎤 Voice Input"):
        user_query = recognize_speech()

# Show past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ========== PROCESS USER QUERY ==========
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Search vector DB
    results = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    # Get response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in get_groq_response(user_query, context):
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()

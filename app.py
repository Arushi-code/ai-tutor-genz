import streamlit as st
import os
import re
import time
import pdfplumber
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ================= CUSTOM PAGE SETTINGS =================
st.set_page_config(
    page_title="AI Tutor Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful look
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    h1 { color: #0f172a; font-weight: 800; font-family: 'Inter', sans-serif; }
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; transition: all 0.3s; }
    .stButton>button[kind="primary"] { background-color: #3b82f6; border-color: #3b82f6; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3); }
    .stButton>button[kind="primary"]:hover { background-color: #2563eb; transform: translateY(-1px); }
    .stFileUploader>div>div { background: #ffffff; border-radius: 12px; border: 2px dashed #94a3b8; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    .tutor-bubble { 
        border-radius: 12px; 
        background-color: #ffffff; 
        padding: 16px 20px; 
        margin-top: 5px;
        font-size: 16px; 
        line-height: 1.6;
        color: #1e293b;
        border-left: 6px solid #10b981;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ================= BACKEND LOGIC (Merged) =================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store():
    if "vs" in st.session_state and st.session_state.vs is not None:
        return st.session_state.vs
    if os.path.exists("faiss_index"):
        try:
            vs = FAISS.load_local("faiss_index", get_embeddings(), allow_dangerous_deserialization=True)
            st.session_state.vs = vs
            return vs
        except: pass
    return None

def clean_text(text):
    text = re.sub(r'-\n+', '', text)
    text = re.sub(r'[^\w\s.,!?\-]', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def fix_broken_lines(text):
    lines = text.split("\n")
    fixed_text = ""
    for line in lines:
        line = line.strip()
        if not line: continue
        if re.match(r'^(i+|v+|\d+)\b', line.lower()) and len(line) < 5: continue
        if fixed_text and not fixed_text.endswith(('.', '?', '!')):
            fixed_text += " " + line
        else:
            fixed_text += "\n" + line
    return fixed_text.strip()

def process_pdf_to_faiss(file_bytes):
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    text = ""
    try:
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text += t + "\n"
    except: pass

    if not text.strip():
        try:
            doc = fitz.open(temp_path)
            for page in doc: text += page.get_text()
        except: pass
        
    if os.path.exists(temp_path):
        os.remove(temp_path)

    text = fix_broken_lines(clean_text(text))
    if not text:
        return False
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50, separators=["\n\n", "\n", ".", " "])
    chunks = splitter.split_text(text)
    
    vs = FAISS.from_texts(chunks, get_embeddings())
    vs.save_local("faiss_index")
    st.session_state.vs = vs
    return True

def generate_ai_answer(question, target_language, vs):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "🚨 No **GEMINI_API_KEY** found in environment variables. Please set it in your Streamlit Cloud Dashboard!"
    
    genai.configure(api_key=api_key)
    
    docs = vs.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])
    context = context.replace("\n", " ")
    context = re.sub(r'\s+', ' ', context)
    
    prompt = f"""You are a helpful, smart AI tutor for students.
Answer the student's question based on the given context. If the context does not explicitly define or fully answer the question, you are allowed to use your expert knowledge to help them, but always prioritize the context provided.
Always provide your final answer in **{target_language}**.
Give a correct and educational answer in simple {target_language}.
Limit your answer to exactly 1 or 2 simple sentences.

Context: 
{context}

Question: {question}"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)
        return result.text.strip()
    except Exception as e:
        # If the hardcoded version 404s due to region locks or deprecation, let the API Key tell us what models it CAN use!
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if not available_models:
                return "🚨 Your API Key does not have access to ANY text generation models. This might be a regional block or an incorrect project key."
            
            # Use the first available Gemini model!
            fallback = available_models[0]
            for m in available_models:
                if 'flash' in m.lower():
                    fallback = m
                    break
                    
            model = genai.GenerativeModel(fallback.replace("models/", ""))
            result = model.generate_content(prompt)
            return result.text.strip()
        except Exception as e2:
            return f"🚨 API Key Check Error: {str(e2)}. \n\nOriginal Error: {str(e)}"


# ================= FRONTEND LOGIC =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = (get_vector_store() is not None)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10435/10435160.png", width=70)
    st.title("Tutor Studio")
    st.markdown("Configure your AI teacher's parameters below.")
    
    st.markdown("---")
    st.subheader("📚 1. Knowledge Base")
    uploaded_file = st.file_uploader("Upload Course Material (PDF)", type=["pdf"])

    if uploaded_file and not st.session_state.pdf_uploaded:
        if st.button("Process & Learn PDF", type="primary"):
            with st.spinner("Extracting and securing knowledge..."):
                success = process_pdf_to_faiss(uploaded_file.getvalue())
                if success:
                    st.session_state.pdf_uploaded = True
                    st.success("✅ Neural embeddings successful!")
                    st.balloons()
                    st.session_state.messages.append({"role": "assistant", "content": f"I have successfully read **{uploaded_file.name}**. You can start asking me questions!"})
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to process PDF. No text found.")

    elif st.session_state.pdf_uploaded:
        st.success("**Systems Active:** Memory is locked and loaded.")
        if st.button("Clear Memory & Upload New PDF"):
            st.session_state.pdf_uploaded = False
            st.session_state.vs = None
            st.session_state.messages = []
            st.rerun()
            
    st.markdown("---")
    st.subheader("🗣️ 2. Capabilities")
    language = st.selectbox("Spoken Language", ["English", "Hindi"], index=0)
    
    st.markdown("---")
    st.caption("Powered by Google Gemini 1.5 Flash. Secure, fast, and highly capable.")


st.title("🤖 Your Intelligent AI Tutor")
st.markdown("Welcome! I am specifically trained on your uploaded document chapters. What would you like to learn today?")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🟢"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='tutor-bubble'>{message['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question about your textbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🟢"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing context..."):
            vs = get_vector_store()
            if vs is None:
                answer = "🚨 Please upload a PDF in the sidebar first so I have context to answer your questions!"
            else:
                answer = generate_ai_answer(prompt, language, vs)
                
            st.markdown(f"<div class='tutor-bubble'>{answer}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
import requests
import time

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
    /* Make the background slightly distinct */
    .stApp { background-color: #f8fafc; }
    
    /* Clean, modern typography */
    h1 { color: #0f172a; font-weight: 800; font-family: 'Inter', sans-serif; }
    
    /* Beautiful button styling */
    .stButton>button { border-radius: 8px; font-weight: 600; width: 100%; transition: all 0.3s; }
    .stButton>button[kind="primary"] { background-color: #3b82f6; border-color: #3b82f6; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3); }
    .stButton>button[kind="primary"]:hover { background-color: #2563eb; transform: translateY(-1px); }
    
    /* Upload region styling */
    .stFileUploader>div>div { background: #ffffff; border-radius: 12px; border: 2px dashed #94a3b8; }
    
    /* Clean up the sidebar */
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    
    /* Custom AI response styling */
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

# Initialize Session State to remember the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False


# ================= SIDEBAR (CONTROLS & UPLOAD) =================
with st.sidebar:
    # A fake professional logo icon
    st.image("https://cdn-icons-png.flaticon.com/512/10435/10435160.png", width=70)
    st.title("Tutor Studio")
    st.markdown("Configure your AI teacher's parameters below.")
    
    st.markdown("---")
    st.subheader("📚 1. Knowledge Base")
    uploaded_file = st.file_uploader("Upload Course Material (PDF)", type=["pdf"])

    if uploaded_file and not st.session_state.pdf_uploaded:
        if st.button("Process & Learn PDF", type="primary"):
            with st.spinner("Extracting and securing knowledge..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post("http://127.0.0.1:8000/upload", files=files)
                    if response.status_code == 200:
                        st.session_state.pdf_uploaded = True
                        st.success("✅ Neural embeddings successful!")
                        st.balloons() # Confetti balloons to wow the user!
                        # Add a welcome message from the AI when file is uploaded
                        st.session_state.messages.append({"role": "assistant", "content": f"I have successfully read **{uploaded_file.name}**. You can start asking me questions!"})
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to process PDF on the backend.")
                except Exception as e:
                    st.error("❌ Could not connect to your main.py FastAPI server.")

    elif st.session_state.pdf_uploaded:
        st.success(f"**Systems Active:** Memory is locked onto `{uploaded_file.name}`")
        if st.button("Unload Document"):
            st.session_state.pdf_uploaded = False
            st.session_state.messages = [] # clear chat
            st.rerun()
            
    st.markdown("---")
    st.subheader("🗣️ 2. Capabilities")
    language = st.selectbox("Spoken Language", ["English", "Hindi"], index=0)
    
    st.markdown("---")
    st.caption("Powered by locally hosted HuggingFace Qwen models. Secure, fast, and highly capable.")


# ================= MAIN CHAT AREA =================
st.title("🤖 Your Intelligent AI Tutor")
st.markdown("Welcome! I am specifically trained on your uploaded document chapters. What would you like to learn today?")

# Display Chat History 
for message in st.session_state.messages:
    # Use native Streamlit chat UI
    if message["role"] == "user":
        with st.chat_message("user", avatar="🟢"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='tutor-bubble'>{message['content']}</div>", unsafe_allow_html=True)

# User Chat Input
if prompt := st.chat_input("Ask a question about your textbook..."):
    # 1. Display User Message Instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🟢"):
        st.markdown(prompt)

    # 2. Call the Backend API while showing "Thinking..." to the user
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing context..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": prompt, "language": language}
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                    st.markdown(f"<div class='tutor-bubble'>{answer}</div>", unsafe_allow_html=True)
                    # Add to session history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Backend returned error: {response.status_code}")
            except Exception as e:
                st.error("🚨 Failed to connect to the backend API. Did you start your FastAPI server (`python main.py`)?")
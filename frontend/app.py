import streamlit as st
import requests

st.set_page_config(page_title="AI Tutor for Remote India 🚀")

st.title("AI Tutor for Remote India 🚀")

# ------------------ PDF Upload ------------------
uploaded_file = st.file_uploader("Upload Textbook PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Upload PDF"):
        with st.spinner("Uploading PDF..."):
            response = requests.post(
                "http://127.0.0.1:8003/upload",   # ✅ SAME PORT as backend
                files={"file": uploaded_file}     # ✅ FIXED LINE
            )

        if response.status_code == 200:
            st.success("PDF uploaded successfully to backend!")
        else:
            try:
                error_msg = response.json().get("error", "Unknown error")
            except:
                error_msg = response.text
            st.error(f"Upload failed: {error_msg}")

# ------------------ Ask Question ------------------
question = st.text_input("Ask your question from the uploaded PDF:")

if st.button("Submit"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        with st.spinner("Getting answer..."):
            response = requests.post(
                "http://127.0.0.1:8003/ask",
                json={"question": question}
            )

        if response.status_code == 200:
            answer = response.json().get("answer", "No answer received")
            st.markdown("**Answer:**")
            st.write(answer)
        else:
            st.error(f"Error: {response.status_code}")
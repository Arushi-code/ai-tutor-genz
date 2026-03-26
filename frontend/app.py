import streamlit as st
import requests

st.set_page_config(page_title="AI Tutor", layout="centered")

st.title("🤖 AI Tutor for Students")
st.write("Upload a PDF and ask any question from it.")

# UPLOAD FILE TO PROCESS PDF
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8003/upload", files=files)

            if response.status_code == 200:
                st.success("📚 PDF processed successfully!")
            else:
                st.error("❌ Error processing PDF")

# ASK THE QUESTION TO GET THE ANSWER
st.subheader("❓ Ask your question")

question = st.text_input("Type your question here")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("Thinking... 🤔"):
            response = requests.post(
                "http://127.0.0.1:8003/ask",
                json={"question": question}
            )

            if response.status_code == 200:
                answer = response.json()["answer"]

                # ANSWER DISPLAY WITH NICE FORMATTING
                st.markdown("### ✅ Answer")
                st.success(answer)

            else:
                st.error("❌ Failed to get answer")

# ADD A FOOTER
st.markdown("---")
st.caption("💡 Powered by AI | Built with FastAPI + LangChain + Transformers")
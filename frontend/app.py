import streamlit as st
import requests

st.title("AI Tutor for Remote India 🚀")

# ------------------ PDF Upload ------------------

uploaded_file = st.file_uploader("Upload Textbook PDF", type="pdf")

if uploaded_file is not None:
    response = requests.post(
        "http://127.0.0.1:8000/upload",
        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    )

    st.write("Upload Status Code:", response.status_code)
    st.write("Server Response:", response.text)

    if response.status_code == 200:
        st.success("PDF uploaded successfully to backend!")
    else:
        st.error("Upload failed.")

# ------------------ Ask Question ------------------

question = st.text_input("Ask your question")

if st.button("Submit"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question}
        )

        st.write("Ask Status Code:", response.status_code)
        st.write("Server Response:", response.text)

        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found")
            st.write("Answer:", answer)
        else:
            st.error("Error from backend")
# 🤖 GenZ AI Tutor Studio

A fully **Serverless, Multi-PDF AI Education Tutor** natively built in Python using Streamlit, FAISS, and Google Gemini.

## ✨ Core Features
*   **Multi-PDF Knowledge Base:** Upload multiple textbooks simultaneously without clearing memory. The engine automatically merges extracted text from all PDFs natively.
*   **RAG Neural Memory:** Runs `sentence-transformers` locally and compiles a fully searchable `faiss-cpu` localized dictionary to provide strictly accurate context.
*   **Dynamic Cloud Provider:** Integrates the official `google-generativeai` SDK to dynamically detect your API key's model access list, preventing region or deprecation errors.
*   **Multilingual Output:** Ask a question, and force the AI to return simple, concise educational answers in English or Hindi.
*   **Zero-Footprint Architecture:** The entire backend server, vector storage pipeline, text cleaner, and frontend chat UI were consolidated into a single monolithic `app.py` script.

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Arushi-code/ai-tutor-genz.git
    cd ai-tutor-genz
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Add your Google Gemini Key:**
    Set `GEMINI_API_KEY` in your environment (Windows/Mac/Linux).
4.  **Launch the Studio:**
    ```bash
    streamlit run app.py
    ```

## ☁️ How to Deploy (100% Free)
Because this application has been structurally compressed into one file, it completely bypasses the need for Render or Docker Web Services.

You can instantly deploy this permanently for free on **Streamlit Community Cloud (share.streamlit.io)**:
1. Connect this repository.
2. Ensure the main file path says `app.py`.
3. In Advanced Settings -> Secrets, add your `GEMINI_API_KEY`.
4. Hit Deploy!

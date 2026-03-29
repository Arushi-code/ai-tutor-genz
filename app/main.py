import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pdfplumber
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import google.generativeai as genai

port = int(os.environ.get("PORT", 8000))

# Initialize the Free Google Gemini Cloud API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ✅ GLOBAL INSTANCES
vector_store = None
embeddings = None # Lazy-loaded to prevent Render boot timeouts

def get_embeddings():
    global embeddings
    if embeddings is None:
        print("Downloading/Loading HuggingFace Embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

# ✅ SERVER LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    if os.path.exists("faiss_index"):
        try:
            vector_store = FAISS.load_local(
                "faiss_index", 
                get_embeddings(), 
                allow_dangerous_deserialization=True
            )
            print("Successfully loaded existing FAISS index from disk.")
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
    yield

app = FastAPI(lifespan=lifespan)


# ---------------- TEXT CLEANING ----------------
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
        if not line:
            continue
        if re.match(r'^(i+|v+|\d+)\b', line.lower()) and len(line) < 5:
            continue
        if fixed_text and not fixed_text.endswith(('.', '?', '!')):
            fixed_text += " " + line
        else:
            fixed_text += "\n" + line
    return fixed_text.strip()


# ---------------- PDF EXTRACTION ----------------
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except:
        pass

    if not text.strip():
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        except:
            pass

    text = clean_text(text)
    text = fix_broken_lines(text)
    return text


# ---------------- FRONTEND WEB UI ----------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Education Tutor AI</title>
        <style>
            body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc; color: #334155; margin: 0; padding: 40px 20px; display: flex; flex-direction: column; align-items: center; }
            h1 { color: #0f172a; margin-bottom: 30px; font-weight: 800; font-size: 2.2rem; }
            .container { background: #ffffff; padding: 35px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); width: 100%; max-width: 650px; margin-bottom: 25px; border: 1px solid #e2e8f0; }
            h3 { margin-top: 0; color: #1e293b; font-weight: 600; display: flex; align-items: center; gap: 8px;}
            input[type="file"] { width: 100%; padding: 12px; margin: 10px 0; border: 2px dashed #cbd5e1; border-radius: 8px; box-sizing: border-box; background: #f8fafc; color: #475569; }
            input[type="text"] { width: 100%; padding: 14px 16px; margin: 15px 0; border: 1px solid #cbd5e1; border-radius: 8px; box-sizing: border-box; font-size: 15px; outline: none; transition: border-color 0.2s; }
            input[type="text"]:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
            button { background-color: #3b82f6; color: white; border: none; padding: 14px 24px; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; width: 100%; transition: all 0.2s; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2); }
            button:hover { background-color: #2563eb; transform: translateY(-1px); box-shadow: 0 6px 8px -1px rgba(59, 130, 246, 0.3); }
            button:active { transform: translateY(0); }
            .upload-btn { background-color: #10b981; box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2); }
            .upload-btn:hover { background-color: #059669; box-shadow: 0 6px 8px -1px rgba(16, 185, 129, 0.3); }
            
            .chat-box { border-top: none; }
            #chatHistory { display: flex; flex-direction: column; gap: 15px; max-height: 400px; overflow-y: auto; margin-bottom: 20px; padding-right: 5px; }
            .message { padding: 16px 20px; border-radius: 12px; line-height: 1.5; font-size: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
            .user-msg { background: #f1f5f9; align-self: flex-end; border-bottom-right-radius: 4px; border-left: 4px solid #94a3b8; }
            .ai-msg { background: #eff6ff; align-self: flex-start; border-bottom-left-radius: 4px; border-left: 4px solid #3b82f6; color: #1e3a8a; }
            
            .status { font-size: 14px; margin-top: 15px; text-align: center; font-weight: 500; padding: 10px; border-radius: 6px; }
        </style>
    </head>
    <body>

        <h1>🧠 Education Tutor AI</h1>
        
        <div class="container">
            <h3>📂 1. Upload Course Material (PDF)</h3>
            <p style="color: #64748b; font-size: 14px; margin-top: 0; margin-bottom: 15px;">Upload a PDF to teach the AI. It will remember the knowledge so you only have to do this once.</p>
            <input type="file" id="pdfUpload" accept="application/pdf">
            <button class="upload-btn" onclick="uploadPDF()">Learn From PDF</button>
            <div id="uploadStatus" class="status"></div>
        </div>

        <div class="container chat-box">
            <h3>💬 2. Ask a Question</h3>
            <div style="margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                <label for="languageSelect" style="font-weight: 600; font-size: 15px; color: #475569;">Tutor Language:</label>
                <select id="languageSelect" style="padding: 8px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 15px; background: #f8fafc; color: #334155; outline: none; cursor: pointer;">
                    <option value="English">English</option>
                    <option value="Hindi">Hindi (हिंदी)</option>
                </select>
            </div>
            <div id="chatHistory">
                <div class="message ai-msg"><b>Tutor:</b> Hello! Upload a textbook above, or if you already have, ask me any questions about it!</div>
            </div>
            
            <form id="askForm" onsubmit="event.preventDefault(); askQuestion();" style="display: flex; gap: 10px;">
                <input type="text" id="questionInput" placeholder="Type your question here..." autocomplete="off">
                <button type="submit" style="width: auto; white-space: nowrap;">Ask Tutor</button>
            </form>
        </div>

        <script>
            async function uploadPDF() {
                const fileInput = document.getElementById('pdfUpload');
                const statusDiv = document.getElementById('uploadStatus');
                
                if (!fileInput.files[0]) {
                    statusDiv.innerHTML = "<span style='color: #ef4444;'>Please select a PDF file first.</span>";
                    statusDiv.style.backgroundColor = "#fee2e2";
                    return;
                }
                
                statusDiv.innerHTML = "<span style='color: #0d9488;'>Uploading and processing... This memory-saving step may take a minute depending on the PDF size.</span>";
                statusDiv.style.backgroundColor = "#ccfbf1";
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const result = await response.json();
                    
                    if(response.ok) {
                        statusDiv.innerHTML = `<span style="color: #059669;">✅ <b>Success:</b> ${result.message}</span>`;
                        statusDiv.style.backgroundColor = "#d1fae5";
                    } else {
                        throw new Error(result.message || "Upload Failed");
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<span style="color: #ef4444;">❌ <b>Error:</b> ${error.message}</span>`;
                    statusDiv.style.backgroundColor = "#fee2e2";
                }
            }

            async function askQuestion() {
                const qInput = document.getElementById('questionInput');
                const chatHistory = document.getElementById('chatHistory');
                const langSelect = document.getElementById('languageSelect');
                
                const question = qInput.value.trim();
                const language = langSelect.value;
                
                if (!question) return;

                // User Message UI
                chatHistory.innerHTML += `<div class="message user-msg"><b>You:</b> ${question}</div>`;
                qInput.value = '';
                chatHistory.scrollTop = chatHistory.scrollHeight;

                // AI Loading UI
                const loadingId = "loading-" + Date.now();
                chatHistory.innerHTML += `<div class="message ai-msg" id="${loadingId}"><b>Tutor:</b> 🤔 Thinking...</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question, language: language })
                    });
                    
                    const result = await response.json();
                    
                    if(response.ok) {
                        document.getElementById(loadingId).innerHTML = `<b>Tutor:</b> ${result.answer}`;
                    } else {
                        throw new Error("API returned an error");
                    }
                } catch (error) {
                    document.getElementById(loadingId).innerHTML = `<span style="color: #ef4444;"><b>Tutor:</b> Failed to get an answer. Make sure the server is running.</span>`;
                }
                
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)

    if not text:
        return {"message": "No readable text found in PDF"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    
    vector_store = FAISS.from_texts(chunks, get_embeddings())
    vector_store.save_local("faiss_index")

    return {"message": "PDF uploaded successfully. The knowledge has been saved and is ready for questions!"}


# ---------------- QUESTION ----------------
class QuestionRequest(BaseModel):
    question: str
    language: str = "English"

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global vector_store

    if vector_store is None:
        return {"answer": "Please upload a PDF first."}

    question = request.question.strip()
    target_language = request.language.strip()

    # Retrieve context
    docs = vector_store.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in docs])

    context = context.replace("\n", " ")
    context = re.sub(r'\s+', ' ', context)
    # Let's cleanly construct the prompt
    prompt = f"""You are a helpful, smart AI tutor for students.
Answer the student's question based ONLY on the given context.
Always provide your final answer in **{target_language}**.
Give a correct and educational answer in simple {target_language}.
Limit your answer to exactly 1 or 2 simple sentences. If the answer is not in the context, say you don't know based on the provided text.

Context: 
{context}

Question: {question}"""

    try:
        # Using Gemini 1.5 Flash - extremely fast, capable and consumes 0 MB of your server's RAM!
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)
        answer = result.text.strip()
    except Exception as e:
        answer = "🚨 Failed to get AI response. Please ensure you have added your free GEMINI_API_KEY environment variable in Render!"

    # Optional: ensure we stop at the last full sentence if it cut off midway,
    # but Qwen usually stops cleanly at an end-of-turn token.
    if not answer.endswith(('.', '?', '!', '"', "'")):
        last_period = answer.rfind('.')
        if last_period != -1:
            answer = answer[:last_period + 1]
        else:
            answer += "."

    return {"answer": answer}


# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
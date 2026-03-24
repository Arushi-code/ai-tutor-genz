import os
import re
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pdfplumber
import pytesseract
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline

app = FastAPI()
vector_store = None

# ---------------- CONFIG ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9., ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file_path):
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        pass

    # OCR fallback
    if not text.strip():
        try:
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_path = f"temp_{page.number}.png"
                pix.save(img_path)

                page_text = pytesseract.image_to_string(
                    img_path,
                    config="--oem 3 --psm 6 -l eng"
                )

                text += page_text + "\n"
                os.remove(img_path)

        except Exception as e:
            print("OCR error:", e)

    return clean_text(text)


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
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c) > 50]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    return {"message": "PDF uploaded and processed successfully"}


# ---------------- ASK ----------------
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):

    global vector_store

    if vector_store is None:
        return {"answer": "Please upload a PDF first."}

    # 🔍 Retrieve context
    docs = vector_store.similarity_search(request.question, k=5)
    context = " ".join([doc.page_content for doc in docs])
    context = context[:800]

    # ---------------- RULE-BASED EXTRACTION ----------------
    sentences = re.split(r'(?<=[.!?]) +', context)

    stop_words = ["what", "is", "the", "of", "define", "explain"]
    question_words = [
        word for word in request.question.lower().split()
        if word not in stop_words
    ]

    best_sentence = ""

    for i, s in enumerate(sentences):
        if any(word in s.lower() for word in question_words):
            best_sentence = s

            # add next sentence for completeness
            if i + 1 < len(sentences):
                best_sentence += " " + sentences[i + 1]
            break

    # ✅ If good answer found → return
    if best_sentence and len(best_sentence.split()) > 5:
        return {"answer": best_sentence.strip()}

    # ---------------- MODEL FALLBACK ----------------
    prompt = f"""
Answer the question clearly in one sentence using the context.

Context:
{context}

Question:
{request.question}

Answer:
"""

    result = qa_pipeline(prompt, max_new_tokens=100)
    answer = result[0]["generated_text"].strip()

    answer = answer.replace(prompt, "").strip()
    answer = re.sub(r'\s+', ' ', answer)

    if "." in answer:
        answer = answer[:answer.rfind(".") + 1]

    if len(answer) < 5:
        answer = "Answer not found in the PDF."

    return {"answer": answer}

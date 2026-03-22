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

# CONFIGURE TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ PIPELINE FIXING FOR BETTER ANSWERS
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")


# Cleaning text for better chunking and embedding
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9., ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Extraction of text from PDF
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

    # TRY OCR (FOR SCANNED PDFs) 
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


# UPLOAD AND PROCESSING OF PDF
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
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c) > 50]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    return {"message": "PDF uploaded and processed successfully"}


# QUESTION ASKING
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):

    global vector_store

    if vector_store is None:
        return {"answer": "Please upload a PDF first."}

    # ✅ CONTEXT RETRIEVAL 
    docs = vector_store.similarity_search(request.question, k=3)

    context = " ".join([doc.page_content for doc in docs])
    context = context[:600]

    # ✅ ADDITION OF QUESTION INTO THE CONTEXT
    context = f"Question: {request.question}. Context: {context}"

    # ✅ PROMPT WRITING FOR ANSWERS 
    prompt = f"""
You are an AI tutor.

Answer the question ONLY using the given context.

If the answer is not clearly present, say: Answer not found.

{context}

Answer:
"""

    result = qa_pipeline(prompt, max_new_tokens=120)

    answer = result[0]["generated_text"].strip()

    # ✅ FOR CLEANER ANSWERS AND REMOVAL OF PROMPT
    answer = answer.replace(prompt, "").strip()
    answer = re.sub(r'(?:\b\w\s+){3,}\w\b', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    # Ensure proper ending
    if not answer.endswith("."):
        if "." in answer:
            answer = answer[:answer.rfind(".")+1]
        else:
            answer = answer + "."
    if len(answer) < 20:
        answer = context[:300]

    return {"answer": answer}
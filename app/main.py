import os
import re
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pdfplumber
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline

app = FastAPI()
vector_store = None

# ✅ MODEL FOR BETTER PERFORMANCE
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")


# TEXT CLEANING TO GET THE BETTER ANSWERS
def clean_text(text):
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# FIXING BROKEN LINES FROM THE PDF TO GET CLEAR AND BETTER CONTEXT FOR ANSWERING
def fix_broken_lines(text):
    lines = text.split("\n")
    fixed_text = ""

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # ❌ removal of numbering like 1. 2. i. ii. iii. iv. v. etc which can break the answer generation
        if re.match(r'^(i+|v+|\d+)\b', line.lower()):
            continue

        # ✅ join lines that are broken 
        if fixed_text and not fixed_text.endswith(('.', '?', '!')):
            fixed_text += " " + line
        else:
            fixed_text += "\n" + line

    return fixed_text.strip()


# EXTRACTION OF TEXT FROM THE PDF FOR GETTING BETTER ANSWERS
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

    # fallback
    if not text.strip():
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()

    text = clean_text(text)
    text = fix_broken_lines(text)

    return text


# UPLOAD AND PROCESS PDF TO CREATE VECTOR STORE FOR ANSWERING QUESTIONS
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

    # ✅ CHUNKING OF THE TEXT 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    return {"message": "PDF uploaded and processed successfully"}


# ASK THE QUESTION AND GET THE ANSWER BASED ON THE PROCESSED PDF 
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):

    global vector_store

    if vector_store is None:
        return {"answer": "Please upload a PDF first."}

    question = request.question.strip()

    # 🔍 CONTEXT RETRIEVAL
    docs = vector_store.similarity_search(question, k=5)
    context = " ".join([doc.page_content for doc in docs])

    context = context[:2000]

    # ✅ AGAIN CLEAN THE CONTEXT TO GET BETTER ANSWERS
    context = re.sub(r'\b(i|ii|iii|iv|v)\b', '', context, flags=re.IGNORECASE)
    context = re.sub(r'\s+', ' ', context)

    # PROMPT ENGINNEERING FOR BETTER ANSWERS
    prompt = f"""
You are an AI tutor.

Your task:
- Read the context carefully
- Answer the question correctly in ONE COMPLETE sentence
- Do NOT copy broken lines
- Do NOT include unrelated text

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_pipeline(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    answer = result[0]["generated_text"]

    # CLEANING OF THE OUTPUT 
    answer = answer.replace(prompt, "").strip()
    answer = re.sub(r'\s+', ' ', answer)

    # removal of bad starting words
    answer = re.sub(r'^(thus|and|so)\s+', '', answer, flags=re.IGNORECASE)

    # keeping only one sentence
    if "." in answer:
        answer = answer[:answer.rfind(".") + 1]

    if len(answer) < 10:
        return {"answer": "Answer not found in the PDF."}

    return {"answer": answer}

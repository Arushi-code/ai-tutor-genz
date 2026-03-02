import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pypdf import PdfReader

app = FastAPI()

# Global storage for extracted text
stored_text = ""

# ------------------ Upload Route ------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global stored_text

    try:
        # Ensure data folder exists
        os.makedirs("data", exist_ok=True)

        file_location = os.path.join("data", file.filename)

        # Save uploaded file
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)

        # Read PDF and extract text
        reader = PdfReader(file_location)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        stored_text = text

        print("Extracted text length:", len(text))

        return {"message": "File uploaded successfully"}

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}


# ------------------ Ask Route ------------------

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global stored_text

    if not stored_text:
        return {"answer": "Please upload a PDF first."}

    return {
        "answer": f"You asked: {request.question}\n\nPDF length: {len(stored_text)} characters"
    }


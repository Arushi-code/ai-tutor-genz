import os
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (like API keys)
load_dotenv()

# Configure your API key here. Make sure you have a .env file with OPENAI_API_KEY=your_key_here
# Alternatively, you can just paste it directly below but that is less secure.
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None

class EducationTutor:
    def __init__(self, persist_dir: str = "./vector_store"):
        self.persist_dir = persist_dir
        
        # Using a small, fast local embedding model to save costs!
        # This part runs locally on the edge device or server without hitting a paid API.
        print("Loading local embedding model (all-MiniLM-L6-v2) to minimize costs...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        
        # Using OpenAI's gpt-4o-mini as it is highly cost-effective and fast for text generation.
        if client:
            self.model_name = "gpt-4o-mini"
        else:
            self.model_name = None
            print("WARNING: No OPENAI_API_KEY found. You will only be able to see the pruned context but not generate final answers.")

    def ingest_textbook(self, pdf_file: str):
        """
        Reads a PDF textbook and splits it into manageable chunks.
        This prepares the document for 'Context Pruning'.
        """
        if not os.path.exists(pdf_file):
            print(f"Error: {pdf_file} not found. Please provide a valid PDF path.")
            return

        print(f"\n[Ingestion] Reading textbook: {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()

        # We split the textbook into chunks of 1500 characters (~300 words).
        # This overlap ensures we don't cut off important sentences midway.
        print("[Ingestion] Chunking document for Context Pruning...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )

        docs = text_splitter.split_documents(documents)
        print(f"[Ingestion] Textbook split into {len(docs)} manageable sections.")

        print("[Ingestion] Embedding sections into vector space (Local, free embedding to save costs)...")
        # Store embeddings in a local vector database (FAISS)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.persist_dir)
        print("[Ingestion] Vector structure built and saved successfully.")

    def load_existing_knowledge_base(self):
        """Loads an existing vector store to avoid re-embedding on every run."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                self.vector_store = FAISS.load_local(
                    self.persist_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True # required for loading local faiss
                )
                print("Successfully loaded existing textbook knowledge base.")
                return True
            except Exception as e:
                print(f"Failed to load vector store: {e}")
        return False

    def ask_question(self, query: str, top_k: int = 2) -> str:
        """
        Performs context pruning to answer the student's question efficiently.
        """
        if not self.vector_store:
            return "No textbook ingested. Please ingest a textbook first."

        # -------------------------------------------------------------
        # REQUIRED TECHNIQUE: CONTEXT PRUNING
        # Instead of sending a massive 500-page PDF to the LLM (which 
        # costs a lot of money and is slow), we perform a semantic search 
        # to find only the absolute most relevant chunks.
        # -------------------------------------------------------------
        
        start_retrieval = time.time()
        # Retrieve ONLY the top 'k' most relevant chunks
        relevant_docs = self.vector_store.similarity_search(query, k=top_k)
        retrieval_time = time.time() - start_retrieval

        # Combine the "pruned" context chunks into a single string
        pruned_context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # Demonstrate cost savings for the hackathon judges:
        words_in_context = len(pruned_context.split())
        print(f"\n[Context Pruning] Retrieved top {top_k} relevant sections in {retrieval_time:.2f}s.")
        print(f"[Context Pruning] Total context sent to LLM: ~{words_in_context} words (Massive API Savings!).\n")

        # -------------------------------------------------------------
        # GENERATION
        # We now use the LLM to generate a curriculum-aligned answer 
        # using ONLY our small, pruned context.
        # -------------------------------------------------------------
        if not client:
            return f"(API KEY MISSING. The context we would have sent is):\n{pruned_context}"

        try:
            start_generation = time.time()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a highly intelligent, patient, and precise Educational Tutor for state-board students in India."},
                    {"role": "user", "content": f"Please answer the student's question clearly, using ONLY the provided context.\nIf the answer is not in the context, say \"I couldn't find the exact answer in your textbook.\"\n\nTextbook Context:\n{pruned_context}\n\nStudent's Question:\n{query}"}
                ]
            )
            generation_time = time.time() - start_generation
            
            print(f"[LLM Analytics] Answer generated in {generation_time:.2f}s.")
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"

if __name__ == "__main__":
    tutor = EducationTutor()
    
    print("\n=======================================================")
    print(" Welcome to the Education Tutor for Remote India")
    print("=======================================================")
    
    # 1. Update this to the path of your state-board textbook PDF
    SAMPLE_PDF_PATH = "state_board_textbook.pdf"
    
    # Load knowledge base if it exists, otherwise ingest the PDF
    if not tutor.load_existing_knowledge_base():
        if os.path.exists(SAMPLE_PDF_PATH):
            tutor.ingest_textbook(SAMPLE_PDF_PATH)
        else:
            print(f"\n[!] Reminder: Please place a textbook PDF named '{SAMPLE_PDF_PATH}' in this folder.")
            print("[!] I will create a dummy PDF file for you to test with if you like. Run `python dummy_pdf.py` if you have one.")
            
    if tutor.vector_store:
        print("\nReady! Ask a question about the textbook.")
        print("(Type 'quit' or 'exit' to stop)")
        
        while True:
            user_input = input("\nStudent 📝: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input.strip():
                # We use top_k=2 to heavily prune the context and minimize data transfer/cost
                answer = tutor.ask_question(user_input, top_k=2)
                print(f"\nTutor 🎓:\n{answer}")

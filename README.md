# The Education Tutor for Remote India 🇮🇳 🎓

This is a prototype implementation of the **Education Tutor** intelligent tutoring system, designed to ingest entire state-board textbooks and answer questions using **Context Pruning** to drastically reduce API costs.

## Why Context Pruning?
When a student asks a question like *"When did India gain independence?"*, it is incredibly expensive and slow to send a massive 500-page PDF to an LLM. 

**Our Solution:**
1. **Ingestion Layer:** We pre-process and divide the PDF textbook into small chapters/sections and convert them into mathematical representations (embeddings) using a small, **100% free local model** (`all-MiniLM-L6-v2`) on edge devices or cheap servers. We store these vectors in a local `FAISS` database.
2. **Context Pruning Layer:** When the student asks a question, we use semantic search to instantly identify the **Top K** (e.g. top 2) most relevant sections of the textbook. 
3. **Generation Layer:** We pass **ONLY** those pruned sections along with the student's question to the LLM. 
   - **Baseline RAG Payload:** ~100,000 to 1,000,000 tokens per query.
   - **Our Pruned Payload:** ~500 to 1,000 tokens per query.
   - **Result:** Massive reduction in data transfer, vastly lower API costs per query, and much faster responses.

## Prerequisites

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install reportlab # Only needed if you want to generate the dummy PDF
   ```

3. Setup your API Key:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   *(Note: The system still works locally for embedding and retrieval even without an API key, it just won't generate text at the end.)*

## How to Test

1. **Option A: Provide your own textbook PDF**
   Place your state-board textbook PDF in this directory and rename it to `state_board_textbook.pdf`.

   **Option B: Generate a Dummy PDF**
   Run the helper script to create a sample 2-page textbook:
   ```bash
   python create_dummy_pdf.py
   ```

2. **Run the Tutor**
   Run the main script to start the intelligent tutoring session:
   ```bash
   python tutor.py
   ```

3. **Ask a question!**
   Try asking:
   - *"What is the largest planet?"*
   - *"When did India gain independence?"*

Watch the console analytics output to see the magic of **Context Pruning** in action!

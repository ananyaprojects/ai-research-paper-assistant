# AI Research Paper Assistant

A RAG-powered tool that lets you upload any research paper (PDF) and instantly get structured analysis — summaries, methodology breakdowns, dataset extraction, results, limitations, and more — using semantic search and Gemini 2.5 Flash.


https://github.com/user-attachments/assets/d8c2c324-26c1-486b-91d0-0a73b209d273


---

## What It Does

Upload a research paper PDF and get:

- **Summary** — problem statement, approach, key contributions, results
- **Methodology** — architecture, training procedure, techniques
- **Datasets** — all datasets used and their purpose
- **Results** — metrics, accuracy, comparisons, key findings
- **Limitations** — explicit and inferred
- **Future Work** — research directions mentioned in the paper
- **Interview Q&A** — 10 technical questions and answers generated from the paper
- **Ask Anything** — type any question and get an answer grounded strictly in the paper

---

## How It Works (RAG Pipeline)

```
PDF Upload
    ↓
Text Extraction (pypdf)
    ↓
Text Cleaning (regex)
    ↓
Sentence Tokenization (NLTK)
    ↓
Sentence-Based Chunking (77 chunks per paper)
    ↓
Embedding Generation (all-MiniLM-L6-v2 → 384-dim vectors)
    ↓
FAISS Vector Index (sub-second semantic search)
    ↓
Hybrid Retrieval:
  - Broad queries  → full paper context (30,000 chars)
  - Narrow queries → FAISS top-7 most relevant chunks
    ↓
Gemini 2.5 Flash (answers strictly from retrieved context)
    ↓
Gradio Web Interface
```

Hallucination prevention: Gemini is prompted to answer **only** from the retrieved paper content. If information is not present in the paper, it says so explicitly.

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF Extraction | pypdf |
| Text Chunking | NLTK sent_tokenize |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (faiss-cpu) |
| LLM | Gemini 2.5 Flash (Google Generative AI) |
| UI | Gradio |

---

## Setup and Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-research-paper-assistant.git
cd ai-research-paper-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a Gemini API Key
Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and generate a free API key.

### 4. Run the app
```bash
python app.py
```

The app will launch locally at `http://127.0.0.1:7860` and also generate a public shareable Gradio link valid for 72 hours.

---

## Usage

1. Enter your **Gemini API Key** in the password field
2. Upload a **PDF research paper** (max 30 pages)
3. Click **Process Paper** — wait for the status to confirm indexing
4. Click any analysis button or type a custom question

---

## Tested On

- *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT, ICLR 2021) — 22 pages, 600 sentences, 77 chunks

---

## Project Structure

```
ai-research-paper-assistant/
│
├── app.py               # Main Gradio application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Author

**Ananya Mishra**  
M.Sc. Computer Science, University of Delhi  
[LinkedIn](https://www.linkedin.com/in/ananya-mishra) | [GitHub](https://github.com/YOUR_USERNAME)

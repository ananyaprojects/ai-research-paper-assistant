import gradio as gr
import re, numpy as np, io

# ---------- pipeline functions ----------

def extract_pdf_text(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text, len(reader.pages)

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def create_sentence_chunks(sentences, chunk_size=1000):
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            if current: chunks.append(current.strip())
            current = s
    if current: chunks.append(current.strip())
    return chunks

def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index(chunks, model):
    import faiss
    emb = np.array(model.encode(chunks, show_progress_bar=False), dtype=np.float32)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(emb)
    return idx

def search_chunks(query, model, index, chunks, k=5):
    qe = np.array(model.encode([query]), dtype=np.float32)
    _, ids = index.search(qe, k)
    return [chunks[i] for i in ids[0]]

def gemini_call(prompt, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt).text

# ---------- analysis functions ----------

def run_summary(t, k):
    return gemini_call(f"Summarize this research paper. Cover Problem Statement, Proposed Approach, Key Contributions, Results.\n\nPaper:\n{t[:30000]}", k)
def run_methodology(t, k):
    return gemini_call(f"Extract the methodology. Include Model Architecture, Training Procedure, Workflow, Important Techniques.\n\nPaper:\n{t[:30000]}", k)
def run_datasets(t, k):
    return gemini_call(f"Identify all datasets used. For each give Dataset Name and Purpose.\n\nPaper:\n{t[:30000]}", k)
def run_results(t, k):
    return gemini_call(f"Extract experimental results. Include Metrics, Accuracy, Comparisons, Key Findings.\n\nPaper:\n{t[:30000]}", k)
def run_limitations(t, k):
    return gemini_call(f"Identify limitations. If not explicit, infer from the paper.\n\nPaper:\n{t[:30000]}", k)
def run_future_work(t, k):
    return gemini_call(f"Extract future work and research directions.\n\nPaper:\n{t[:30000]}", k)
def run_interview(t, k):
    return gemini_call(f"Generate 10 technical interview Q&A. Include Conceptual, Architecture, Methodology, Practical questions.\n\nPaper:\n{t[:30000]}", k)

# ---------- state ----------
STATE = {}

# ---------- handlers ----------

def process_paper(api_key, pdf_path):
    if not api_key:
        return "Please enter your Gemini API key."
    if pdf_path is None:
        return "Please upload a PDF."
    import nltk
    try:    nltk.data.find("tokenizers/punkt_tab")
    except: nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    raw, n_pages = extract_pdf_text(pdf_path)
    if n_pages > 30:
        return f"PDF has {n_pages} pages. Max 30 allowed."
    cleaned = clean_text(raw)
    chunks  = create_sentence_chunks(sent_tokenize(cleaned))
    model   = get_embedding_model()
    idx     = build_faiss_index(chunks, model)
    STATE["cleaned"] = cleaned
    STATE["chunks"]  = chunks
    STATE["idx"]     = idx
    STATE["model"]   = model
    STATE["key"]     = api_key
    return f"Ready! {n_pages} pages · {len(chunks)} chunks indexed in FAISS."

def check_ready():
    if not STATE.get("cleaned"):
        return "Process a paper first."
    return None

def btn_summary():
    err = check_ready(); return err or run_summary(STATE["cleaned"], STATE["key"])
def btn_methodology():
    err = check_ready(); return err or run_methodology(STATE["cleaned"], STATE["key"])
def btn_datasets():
    err = check_ready(); return err or run_datasets(STATE["cleaned"], STATE["key"])
def btn_results():
    err = check_ready(); return err or run_results(STATE["cleaned"], STATE["key"])
def btn_limitations():
    err = check_ready(); return err or run_limitations(STATE["cleaned"], STATE["key"])
def btn_future_work():
    err = check_ready(); return err or run_future_work(STATE["cleaned"], STATE["key"])
def btn_interview():
    err = check_ready(); return err or run_interview(STATE["cleaned"], STATE["key"])

def btn_ask(question):
    err = check_ready()
    if err: return err
    if not question.strip(): return "Type a question first."

    # broad queries -> send full paper text so Gemini always has enough context
    broad = ["summary", "summarize", "overview", "methodology", "method", "dataset",
             "data", "results", "result", "limitations", "limitation", "future",
             "contribution", "conclusion", "introduction", "abstract",
             "what is", "what are", "explain", "describe", "tell me", "give me",
             "how does", "how do", "why", "who", "when", "where"]

    use_full = any(kw in question.lower() for kw in broad)

    if use_full:
        context = STATE["cleaned"][:30000]
    else:
        # specific narrow query -> FAISS top 7
        context = "\n\n".join(search_chunks(question, STATE["model"], STATE["idx"], STATE["chunks"], k=7))

    prompt = f"""You are a research paper assistant.
Answer the question using the paper content provided below.
Be detailed and helpful. Do not say you cannot find information unless it is truly absent from the text.

Paper Content:
{context}

Question: {question}

Answer:"""
    return gemini_call(prompt, STATE["key"])

# ---------- UI ----------

with gr.Blocks(title="AI Research Paper Assistant") as demo:
    gr.Markdown("# AI Research Paper Assistant")

    with gr.Row():
        api_key_box = gr.Textbox(label="Gemini API Key", type="password", placeholder="AIza...")
        pdf_box     = gr.File(label="Upload PDF (max 30 pages)", file_types=[".pdf"])

    process_btn    = gr.Button("Process Paper", variant="primary")
    process_status = gr.Textbox(label="Status", interactive=False)
    process_btn.click(fn=process_paper, inputs=[api_key_box, pdf_box], outputs=[process_status])

    gr.Markdown("---")
    gr.Markdown("### Analysis")
    output_box = gr.Markdown()

    with gr.Row():
        gr.Button("Summary").click(fn=btn_summary, outputs=[output_box])
        gr.Button("Methodology").click(fn=btn_methodology, outputs=[output_box])
        gr.Button("Datasets").click(fn=btn_datasets, outputs=[output_box])
        gr.Button("Results").click(fn=btn_results, outputs=[output_box])

    with gr.Row():
        gr.Button("Limitations").click(fn=btn_limitations, outputs=[output_box])
        gr.Button("Future Work").click(fn=btn_future_work, outputs=[output_box])
        gr.Button("Interview Q&A").click(fn=btn_interview, outputs=[output_box])

    gr.Markdown("---")
    gr.Markdown("### Ask a Question")
    question_box = gr.Textbox(label="Your question", placeholder="e.g. Give me summary / What dataset was used?")
    ask_btn      = gr.Button("Ask", variant="primary")
    ask_btn.click(fn=btn_ask, inputs=[question_box], outputs=[output_box])

demo.launch(share=True)

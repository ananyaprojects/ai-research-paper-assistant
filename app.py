import gradio as gr
import re, numpy as np

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

# ---------- handlers ----------

def process_paper(api_key, pdf_path, state):
    """Process uploaded PDF and store all data in per-user session state."""
    if not api_key:
        return "Please enter your Gemini API key.", state
    if pdf_path is None:
        return "Please upload a PDF.", state

    import nltk
    try:    nltk.data.find("tokenizers/punkt_tab")
    except: nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    raw, n_pages = extract_pdf_text(pdf_path)
    if n_pages > 30:
        return f"PDF has {n_pages} pages. Max 30 allowed.", state

    cleaned = clean_text(raw)
    chunks  = create_sentence_chunks(sent_tokenize(cleaned))
    model   = get_embedding_model()
    idx     = build_faiss_index(chunks, model)

    # Build state AFTER all variables exist — stored per user session, never globally
    state = {
        "cleaned": cleaned,
        "chunks":  chunks,
        "idx":     idx,
        "model":   model,
        "key":     api_key,   # key lives only in this user's session object
    }

    return f"Ready! {n_pages} pages · {len(chunks)} chunks indexed in FAISS.", state


def check_ready(state):
    """Return an error string if the paper hasn't been processed yet, else None."""
    if not state or not state.get("cleaned"):
        return "Process a paper first."
    return None


def btn_summary(state):
    err = check_ready(state)
    return err or run_summary(state["cleaned"], state["key"])

def btn_methodology(state):
    err = check_ready(state)
    return err or run_methodology(state["cleaned"], state["key"])

def btn_datasets(state):
    err = check_ready(state)
    return err or run_datasets(state["cleaned"], state["key"])

def btn_results(state):
    err = check_ready(state)
    return err or run_results(state["cleaned"], state["key"])

def btn_limitations(state):
    err = check_ready(state)
    return err or run_limitations(state["cleaned"], state["key"])

def btn_future_work(state):
    err = check_ready(state)
    return err or run_future_work(state["cleaned"], state["key"])

def btn_interview(state):
    err = check_ready(state)
    return err or run_interview(state["cleaned"], state["key"])


def btn_ask(question, state):
    err = check_ready(state)
    if err: return err
    if not question.strip(): return "Type a question first."

    broad = ["summary", "summarize", "overview", "methodology", "method", "dataset",
             "data", "results", "result", "limitations", "limitation", "future",
             "contribution", "conclusion", "introduction", "abstract",
             "what is", "what are", "explain", "describe", "tell me", "give me",
             "how does", "how do", "why", "who", "when", "where"]

    use_full = any(kw in question.lower() for kw in broad)

    if use_full:
        context = state["cleaned"][:30000]
    else:
        context = "\n\n".join(
            search_chunks(question, state["model"], state["idx"], state["chunks"], k=7)
        )

    prompt = f"""You are a research paper assistant.
Answer the question using the paper content provided below.
Be detailed and helpful. Do not say you cannot find information unless it is truly absent from the text.

Paper Content:
{context}

Question: {question}

Answer:"""
    return gemini_call(prompt, state["key"])


# ---------- UI ----------

with gr.Blocks(title="AI Research Paper Assistant") as demo:
    gr.Markdown("# AI Research Paper Assistant")
    gr.Markdown(
        "> 🔒 **Privacy note:** Your API key is stored only in your browser session "
        "and is never saved to any server or database."
    )

    # Per-user session state — each browser tab gets its own isolated dict
    user_state = gr.State({})

    with gr.Row():
        api_key_box = gr.Textbox(
            label="Gemini API Key", type="password", placeholder="AIza..."
        )
        pdf_box = gr.File(label="Upload PDF (max 30 pages)", file_types=[".pdf"])

    process_btn    = gr.Button("Process Paper", variant="primary")
    process_status = gr.Textbox(label="Status", interactive=False)

    # process_paper now returns (status_text, updated_state)
    process_btn.click(
        fn=process_paper,
        inputs=[api_key_box, pdf_box, user_state],
        outputs=[process_status, user_state],
    )

    gr.Markdown("---")
    gr.Markdown("### Analysis")
    output_box = gr.Markdown()

    with gr.Row():
        gr.Button("Summary").click(fn=btn_summary,     inputs=[user_state], outputs=[output_box])
        gr.Button("Methodology").click(fn=btn_methodology, inputs=[user_state], outputs=[output_box])
        gr.Button("Datasets").click(fn=btn_datasets,   inputs=[user_state], outputs=[output_box])
        gr.Button("Results").click(fn=btn_results,     inputs=[user_state], outputs=[output_box])

    with gr.Row():
        gr.Button("Limitations").click(fn=btn_limitations,  inputs=[user_state], outputs=[output_box])
        gr.Button("Future Work").click(fn=btn_future_work,  inputs=[user_state], outputs=[output_box])
        gr.Button("Interview Q&A").click(fn=btn_interview,  inputs=[user_state], outputs=[output_box])

    gr.Markdown("---")
    gr.Markdown("### Ask a Question")
    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. Give me summary / What dataset was used?"
    )
    ask_btn = gr.Button("Ask", variant="primary")
    ask_btn.click(fn=btn_ask, inputs=[question_box, user_state], outputs=[output_box])

demo.launch()



**How it actually works:**
https://mybinder.org/v2/gh/ananyaprojects/ai-research-paper-assistant/064c4c055de24f8665cd4cfe5ae54fd048863c43?urlpath=lab%2Ftree%2FAI_RESEARCH_PAPER_ASSISTANT_fixed.ipynb

When you upload a research paper PDF, the tool first extracts all the raw text from every page using a PDF reader. That text is then cleaned — extra spaces, broken lines, formatting noise — all removed so the content is readable.

Once clean, the text is split into **chunks** (smaller pieces of text, roughly a paragraph long) using sentence tokenization (breaking text at natural sentence boundaries). My paper produced 77 chunks.

Each chunk is then passed through a **Sentence Transformer** (a type of AI model that reads text and converts it into numbers) called **all-MiniLM-L6-v2**, which generates **embeddings** (a list of 384 numbers that mathematically represent the meaning of that chunk). Chunks that talk about similar topics end up with similar numbers.

All 77 embeddings are stored in **FAISS** (Facebook AI Similarity Search — a vector database that lets you search by meaning, not just by keyword). This is what makes the search smart — it doesn't look for exact words, it looks for matching meaning.

This full process — extract, chunk, embed, store, retrieve — is what's called a **RAG pipeline** (Retrieval-Augmented Generation — a technique where you retrieve relevant information first and then generate an answer from it, instead of relying on the AI's memory alone).

When you ask a question, the tool converts your question into an embedding too, searches FAISS for the most relevant chunks, and passes them to **Gemini 2.5 Flash** — an **LLM** (Large Language Model — an AI trained on massive amounts of text that can read, understand and generate human language).

The key part is the **prompt engineering** (the exact instruction written to control how the LLM behaves). The prompt tells Gemini — answer strictly from the paper content provided. Nothing else. This directly prevents **hallucination** (when an AI confidently gives an answer that sounds correct but is completely made up).

I also built a hybrid retrieval system — broad questions like "summarize the paper" send the full paper text to Gemini, while specific questions like "what optimizer was used" use FAISS to fetch only the 7 most relevant chunks. This keeps answers accurate and context-aware.

The interface is built on **Gradio** (an open-source Python library for building web-based AI demos) and deployed with a public shareable link.

---


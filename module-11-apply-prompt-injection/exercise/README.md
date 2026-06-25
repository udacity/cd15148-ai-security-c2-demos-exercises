# Prompt Injection Assessment Workflow Exercise

This exercise extends the Module 11 prompt injection demo from direct chat injection into a simplified RAG customer support assistant.

Students evaluate whether malicious instructions embedded in indexed documents can manipulate assistant behavior when those documents are retrieved for otherwise benign user questions.

## Structure

- `starter/`: student scaffold with TODOs.
- `solution/`: completed reference implementation and expected metrics.

## Environment

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The notebooks use a FAISS-style RAG flow adapted from the project `rag_chatbot`: documents are embedded with OpenAI `text-embedding-3-small`, vectors are added to an index, top-k chunks are retrieved, and retrieved context is passed to `gpt-4.1-mini`. If FAISS is installed, `faiss.IndexFlatIP` is used directly; otherwise the same OpenAI vectors are searched with a NumPy cosine-similarity fallback.

Copy the appropriate `.env.example` to `.env` or paste the classroom key into the notebook setup cell. Vocareum-issued keys that start with `voc-` automatically use `https://openai.vocareum.com/v1`.

# Solution: Retrieval Poisoning Assessment

This completed solution builds a clean manufacturing RAG retrieval index with OpenAI `text-embedding-3-small`, inserts poisoned maintenance documents with target-relevant content, compares clean versus poisoned retrieval rankings, calls `gpt-4o-mini` for vulnerable and guarded RAG responses, and writes an assessment report.

Run:

```powershell
$env:OPENAI_API_KEY="..."
python run_assessment.py
```

Outputs are written to `data/` and `results/`.
Vocareum-issued keys that start with `voc-` automatically use the Vocareum base URL.

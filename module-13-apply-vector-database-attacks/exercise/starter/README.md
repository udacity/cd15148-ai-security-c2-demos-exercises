# Starter: Retrieval Poisoning Assessment

Complete the TODOs in `src/retrieval_poisoning_assessment.py`, then run:

```powershell
$env:OPENAI_API_KEY="..."
python run_assessment.py
```

The starter is intentionally incomplete. Use the notebook in `notebooks/` to guide your implementation and fill in `docs/assessment_report_template.md` with your findings. Retrieval uses OpenAI `text-embedding-3-small`; vulnerable and guarded RAG responses use `gpt-4o-mini`. Vocareum-issued keys that start with `voc-` automatically use the Vocareum base URL.

# Exercise: Build a Retrieval Poisoning Assessment Workflow for an Enterprise Search Assistant

Estimated time: 45 minutes

This exercise asks learners to assess vector database retrieval poisoning against a manufacturing engineering assistant. The workflow uses OpenAI `text-embedding-3-small` for semantic retrieval and `gpt-4o-mini` for vulnerable and guarded RAG responses. The starter contains guided TODOs; the solution contains a complete quantitative workflow.

## Structure

- `starter/` - learner notebook, TODO utility code, report template, and runnable scaffolding
- `solution/` - completed notebook, completed utility code, generated vector artifacts, and assessment results

## Run

```powershell
cd module-13-apply-vector-database-attacks\exercise\solution
$env:OPENAI_API_KEY="..."
python run_assessment.py
```

Use `--plot` to create the optional Matplotlib chart when the environment supports it.
Vocareum-issued keys that start with `voc-` automatically use `https://openai.vocareum.com/v1`.

# Gmac Group Company Assistant

A RAG-powered company information assistant for Gmac Group.
It answers from the Gmac Group Company Profile 2026 using LangChain, Groq, and BM25 retrieval.

---

## Project Structure

```
gmac-group-assistant/
├── app.py               ← Main entry point (run this to launch)
├── gmac_group_knowledge_base.txt ← Company facts used for retrieval
├── requirements.txt     ← Python dependencies
├── .env.example         ← Environment variable template
├── Dockerfile           ← Render container definition
├── render.yaml          ← Render service configuration
├── .dockerignore        ← Docker build exclusions
├── .gitignore           ← Git exclusions
└── static/ and templates/ ← Browser client
```

---

## Quickstart (local)

```bash
# 1. Clone / download this folder
# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY=your_key_here   # Windows: set GROQ_API_KEY=your_key_here

# 5. Run
python app.py
# → Open http://localhost:7860
```

---

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and select the repository.
3. Render will detect `render.yaml` and create the Docker web service.
4. Add `GROQ_API_KEY` under the service environment variables.
5. Deploy the service. Render supplies the `PORT` variable automatically.

The health check is available at `/health`. The company profile is loaded from `gmac_group_knowledge_base.txt` at startup.

---

## Environment Variables

| Variable       | Required | Description              |
|----------------|----------|--------------------------|
| `GROQ_API_KEY` | Yes      | Your Groq API key        |
| `PORT`         | No       | Supplied automatically by Render; defaults to 7860 locally |

---

## Notes

- The company knowledge base is loaded from `gmac_group_knowledge_base.txt` at startup.
- Answers must stay within the company profile. Published pricing is expressed as bands only.
- The default model is `qwen/qwen3.8-27b` via Groq. Override it with `GROQ_MODEL` when needed.
- All Groq API calls are made server-side; the key is never exposed to the browser.

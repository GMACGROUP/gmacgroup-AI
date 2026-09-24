---
title: Gmac Group Company Assistant
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

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
├── Dockerfile           ← For Fly.io / Docker deployment
├── fly.toml             ← Fly.io config
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

## Deploy to Hugging Face Spaces (easiest)

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Gradio** as the SDK
3. Upload all files in this folder (except `notebooks/` if you want)
4. Go to **Settings → Repository secrets** and add:
   - `GROQ_API_KEY` = your Groq key
5. The Space will auto-build and launch.

> The company profile is loaded from `gmac_group_knowledge_base.txt` at startup.

---

## Deploy to Fly.io

```bash
# Install flyctl if not already installed
# https://fly.io/docs/hands-on/install-flyctl/

fly auth login
fly launch          # follow prompts; use fly.toml config provided
fly secrets set GROQ_API_KEY=your_key_here
fly deploy
```

---

## Environment Variables

| Variable       | Required | Description              |
|----------------|----------|--------------------------|
| `GROQ_API_KEY` | Yes      | Your Groq API key        |
| `PORT`         | No       | Server port (default 7860) |

---

## Notes

- The company knowledge base is loaded from `gmac_group_knowledge_base.txt` at startup.
- Answers must stay within the company profile. Published pricing is expressed as bands only.
- The default model is `qwen/qwen3.8-27b` via Groq. Override it with `GROQ_MODEL` when needed.
- All Groq API calls are made server-side; the key is never exposed to the browser.

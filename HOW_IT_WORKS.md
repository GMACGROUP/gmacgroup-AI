# Gmac Group Company Assistant

## Request flow

1. The browser sends a question and recent conversation history to `/api/chat`.
2. The Flask backend retrieves matching chunks from `gmac_group_knowledge_base.txt` using BM25.
3. The backend sends only the retrieved company facts, recent history, and the question to Groq.
4. The model is instructed to answer only from the supplied company profile.
5. The response is cleaned and returned with company-focused follow-up suggestions.

## Knowledge source

The active source is `gmac_group_knowledge_base.txt`, derived from the Gmac Group Company Profile 2026. It covers the company's overview, story, principles, six practice areas, team, events, engagement models, pricing bands, partners, differentiators, and contact channels.

The loader also accepts the full self-contained format supplied for this project. Each `### CHUNK:` block is preserved as one retrieval document, and its `topic:` and `keywords:` lines are stored as metadata. Chunks are not split or merged, which keeps facts such as the event catalogue, team roster, FAQs, and data-quality notes together.

The source also contains explicit retrieval rules. The assistant must not invent exact prices, client names, case studies, registration numbers, investment outcomes, or facts outside the profile. The illustrative engagements in the profile are not historical client case studies.

## Company grounding

The assistant speaks on behalf of Gmac Group, not a private individual. It uses the company contact details when a question cannot be answered from the profile:

- Email: info@gmac-group.com
- Ghana: +233 20 215 4828
- Nigeria: +234 814 498 8398
- Website: gmac-group.com

## Render Deployment

Render builds the Docker image defined by `Dockerfile` and uses `render.yaml` to configure the web service. `GROQ_API_KEY` is required at runtime, and Render supplies `PORT` automatically. Local runs default to port `7860`.

The `GET /health` endpoint reports the service status, company name, knowledge source, and number of loaded retrieval chunks.

Website retrieval is enabled by default for `https://gmacgroup.vercel.app/`. The approved pages are the home, about, services, programmes, opportunities, research, insights, and contact pages. Set `WEBSITE_RAG_ENABLED=false` to use only the local company profile.

## Privacy boundary

The crawler uses only an allowlisted public host and approved public paths. Dashboard, account, login, admin, portal, profile, settings, user, private, and API paths are blocked. It sends no cookies or authentication headers and rejects redirects to other hosts.

User conversation history is intentionally excluded from model context. Common emails, phone numbers, API keys, and bearer tokens in the current question are redacted before retrieval and generation. The assistant does not access or retain user dashboards, private accounts, credentials, payment data, or private social messages.

## Current limitation

This is currently a single-company deployment. The retrieval index is built at startup from one company profile. Multi-company tenancy, authentication, document uploads, persistent embeddings, and per-user access control should be added before serving multiple organisations from the same deployment.

# Chrix Tech Persona AI — How It Works

A step-by-step technical guide to how the AI retrieves data, detects intent, builds context, and generates a reply.

---

## System Overview

```
User types a message
       |
       v
  [1] Normalize Text  --> fix typos, expand shorthand (ur -> your, jhs -> junior high school)
       |
       v
  [2] Detect Intent  --> education? skills? projects? greeting? acknowledgment?
       |
       v
  [3] Special Fast-Paths --> greeting pool, acknowledgment redirect (no LLM call)
       |
       v
  [4] Query Biasing  --> rewrite the query to target the right knowledge chunk
       |
       v
  [5] BM25 Retrieval  --> find the top 5 most relevant text chunks from the bio
       |
       v
  [6] Build Prompt  --> combine FOCUS + retrieved FACTS + conversation HISTORY
       |
       v
  [7] LLM Call (Groq / Qwen)  --> generate a human-like reply
       |
       v
  [8] Post-Process (clean_reply)  --> strip reasoning tags, slang, sentence limit
       |
       v
  [9] Return reply + suggestion chips to the frontend
```

---

## Step 1 — Data Sources (The Knowledge Base)

Before anything runs, on server startup, the system loads THREE text sources and splits them into chunks.

### Source A: personal_bio (inside app.py)
The core personal data. Structured sections:
- FULL NAME, LOCATION
- WHO I AM
- EDUCATION — JHS (Edwinase), SHS (Achimota), University (UG Legon), coursework
- TECH STACK AND SKILLS
- PROFESSIONAL EXPERIENCE
- PROJECTS
- CERTIFICATIONS (with real verification URLs)
- HOBBIES AND INTERESTS
- GOALS AND VISION
- CONTACT

### Source B: website_content.txt
A longer, more conversational version of the same data written in natural English paragraphs.
The retriever benefits from having two different phrasings of the same facts — it increases the
chance of matching how a user phrases their question.

### Source C: portfolio_github_links
A short dedicated chunk containing exact URLs:
  Portfolio: https://christiandetails.vercel.app/
  GitHub:    https://github.com/ChristianAgyapong
  LinkedIn:  https://www.linkedin.com/in/christian-agyapong-4a6b7139b

---

## Step 2 — Chunking

All three sources are split by RecursiveCharacterTextSplitter:

  chunk_size=400      (each chunk is max 400 characters)
  chunk_overlap=60    (60-char overlap so context does not break at boundaries)
  separators=["\n\n", "\n", ". ", " "]

Why chunks? BM25 retrieval works on individual text pieces. Smaller chunks = more precise matching.
The overlap ensures a sentence that falls at a boundary is not lost.

All chunks from all three sources are combined into a single list:
  all_docs = personal_bio_chunks + website_chunks + portfolio_chunks

---

## Step 3 — BM25 Retrieval

The system uses BM25 (Best Match 25) — a classical information retrieval algorithm used in search engines.

  retriever = BM25Retriever.from_documents(all_docs)
  retriever.k = 5   (return the top 5 most relevant chunks per query)

### How BM25 Works:
BM25 scores each chunk based on:
  1. Term Frequency (TF) — how many times the query words appear in a chunk
  2. Inverse Document Frequency (IDF) — rare words score higher than common ones
  3. Document length normalization — shorter chunks are not unfairly penalized

Example:
  Query: "Edwinase Ejisu Basic School JHS BECE best student Kumasi"
  BM25 scans all chunks, finds the education section with exact keyword matches,
  and returns it as the top result.

The top 5 chunks are joined and passed to the LLM as context.

---

## Step 4 — Intent Detection

Before retrieval runs, the system classifies what topic the user is asking about.

  INTENT_MAP = {
    "education":   ["school", "university", "jhs", "shs", "achimota", "edwinase", "bece", ...],
    "skills":      ["skills", "stack", "programming", "full stack", ...],
    "projects":    ["project", "built", "whatsapp assistant", "tweeteval", ...],
    "origin":      ["where", "location", "ghana", "accra", "newtown", ...],
    "contact":     ["email", "phone", "freelance", "hire", "available", ...],
    "goals":       ["goal", "vision", "dream", "future", ...],
    "hobbies":     ["football", "music", "hobbies", "relax", ...],
    "general":     []   (catch-all)
  }

The user message is normalized first, then matched against keywords. The first intent that matches wins.

Each intent maps to:
  - INTENT_FOCUS       — a specific instruction to the LLM about what to talk about
  - INTENT_FALLBACK    — a pre-written safe response if the LLM fails
  - INTENT_SUGGESTIONS — the 3 clickable suggestion chips shown after each reply

---

## Step 5 — Text Normalization

Before intent detection and query building, the message is normalized to handle casual language:

  ur    -> your
  u     -> you
  jhs   -> junior high school
  shs   -> senior high school
  wat   -> what
  hw    -> how
  abt   -> about
  wanna -> want to
  gimme -> give me
  ...and more

So "which jhs did u attend" becomes "which junior high school did you attend"
making keyword matching much more reliable.

---

## Step 6 — Query Biasing

This is the key fix that makes retrieval accurate. Instead of sending the raw message to BM25,
the system rewrites the query to include keywords the retriever needs to find the right chunk.

  if "jhs" / "edwinase" / "bece" in message:
    query = "Edwinase Ejisu Basic School JHS BECE best student Kumasi [original message]"

  elif "achimota" / "shs" / "senior high school" in message:
    query = "Achimota School Senior High School General Arts SHS [original message]"

  elif "education" / "university" / "legon" in message:
    query = "education University of Ghana Legon Computer Science Machine Learning [original message]"

  elif "contact" / "email" / "freelance" in message:
    query = "contact email phone WhatsApp availability freelance [original message]"

  elif "github" / "portfolio" / "linkedin" in message:
    query = "portfolio github links [original message]"

Why? BM25 is keyword-based. Adding domain-specific keywords forces BM25 to surface the correct chunk
even when the user phrased their question very casually.

---

## Step 7 — Special Fast-Paths (No LLM Called)

Some message types are handled WITHOUT calling the LLM at all. This is faster and prevents hallucination.

### 7a. Greetings (hey, hi, hello)
Returns a randomly chosen greeting from a pool of 8 varied responses.
Uses Jaccard similarity to avoid repeating the same greeting twice.

### 7b. Acknowledgments (okay, ok, nice, cool, great, wow, got it, interesting...)
Returns a short redirect prompt without calling the LLM.

  Examples:
  "What would you like to explore next — my projects, skills, or availability?"
  "What's your next question? I can go into my projects, stack, or background."

Why no LLM for acknowledgments? Words like "okay" used to trigger the LLM with the last conversation
topic as context, causing the model to HALLUCINATE details not in the bio.
Bypassing the LLM for acknowledgments completely eliminates this problem.

---

## Step 8 — Building the LLM Prompt

The prompt is structured in layers:

  SYSTEM INSTRUCTIONS
    - Persona: Christian Agyapong / Chrix Tech
    - Anti-repetition rules
    - Follow-up intelligence instructions
    - Personality depth cues (proud of BECE achievement, passion for African AI)
    - Style rules (no slang, no hollow endings)
    - Never hallucinate or invent details

  HUMAN MESSAGE:
    FOCUS FOR THIS REPLY:
      [e.g., "For JHS, mention Edwinase Ejisu Basic School and BECE best student 2020"]

    RELEVANT FACTS FROM YOUR LIFE:
      [top 5 BM25 chunks joined together]

    RECENT CONVERSATION HISTORY:
      Person: [last 6 user messages]
      Chrix:  [last 6 AI replies]

    CURRENT MESSAGE: [what the user just typed]

    IMPORTANT REMINDER: Do NOT reuse sentences from previous replies.

### LLM Settings:
  model:       qwen/qwen3.6-27b  (via Groq API)
  temperature: 0.5               (lower = more factually reliable)
  max_tokens:  1800              (enough budget for reasoning + reply)

---

## Step 9 — Post-Processing (clean_reply)

After the LLM generates text, it passes through a cleaning pipeline:

  1. Strip <think>...</think> reasoning blocks
     The Qwen model emits internal reasoning before the reply. This is removed.

  2. Remove bad openers
     Strips robotic filler phrases: "Certainly,", "Of course,", "Chale,", "Herh,"

  3. Remove pidgin slang
     Strips: chale, herh, abeg, naa, mehn, e be so, we dey push, you feel me

  4. Normalize whitespace
     Collapses double spaces, removes leading punctuation artifacts.

  5. Enforce sentence limit
     Education and project replies: up to 5 sentences
     All other replies: up to 4 sentences

  6. Capitalize first character
     Ensures the reply always starts with a capital letter.

---

## Step 10 — Anti-Repetition System

Repetition is tracked at two levels:

### Message-level (Jaccard Similarity)
For greetings and acknowledgments, similarity between each candidate reply and the last AI message
is computed. A candidate is excluded if similarity >= 0.38.

  similarity = |words_in_A n words_in_B| / |words_in_A ? words_in_B|

### Prompt-level (Conversation History Injection)
The last 6 turns of conversation are injected into the prompt with an explicit instruction:
  "Do NOT reuse sentences, phrases, or closing lines from your previous replies."

---

## Complete Request Flow (Summary)

  POST /api/chat  { message: "which jhs did u attend", history: [...] }
         |
         v
  normalize_text  ->  "which junior high school did you attend"
         |
         v
  detect_intent   ->  intent = "education"
         |
         v
  is_acknowledgment? NO  |  is_greeting? NO  |  continue...
         |
         v
  query_biasing: "jhs" found
    -> query = "Edwinase Ejisu Basic School JHS BECE best student Kumasi which jhs did u attend"
         |
         v
  BM25Retriever.invoke(query)  ->  top 5 chunks (including Edwinase education section)
         |
         v
  Build prompt: FOCUS + FACTS + HISTORY
         |
         v
  llm.invoke(messages)  ->  raw reply (may contain <think>...</think>)
         |
         v
  clean_reply()  ->  strip reasoning, remove slang, limit to 5 sentences
         |
         v
  return { reply: "I began at Edwinase Ejisu Basic School...", suggestions: [...] }

---

## Key Files Reference

  app.py               All backend logic: bio, retriever, intent, LLM, post-processing
  website_content.txt  Secondary knowledge source (conversational paragraphs)
  templates/index.html Chat UI structure
  static/index.css     Chat UI styling (glassmorphism, mobile layout)
  static/app.js        Frontend logic: message sending, history, suggestions
  .env                 GROQ_API_KEY secret

---

## How to Update the Knowledge Base

To teach the AI a new fact or update an existing one:

  1. Open app.py and find the relevant section in personal_bio
  2. Edit or add the text in plain English — be specific, use real names and dates
  3. Also update website_content.txt with the same info in conversational paragraph form
  4. Commit and push:

     git add .
     git commit -m "Update bio: [what you changed]"
     git push

Tip: The more specific and keyword-rich your bio text is, the better BM25 will retrieve it.
Write facts the way someone would search for them, not just the way they would appear in a resume.

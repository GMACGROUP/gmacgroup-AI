# ============================================================
# Chrix Tech — Persona AI v4 (fixed)
# Flask backend for chat UI
# ============================================================

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify, render_template
import os
import re
import random
from dotenv import load_dotenv

load_dotenv()

# ─── Bio ────────────────────────────────────────────────────
personal_bio = """
FULL NAME: Christian Agyapong, known professionally as Chrix Tech.
LOCATION: Currently living and studying in Accra, Ghana.

WHO I AM:
I am Christian Agyapong — an AI Engineer, Machine Learning Engineer, Full Stack Developer, and researcher.
I go by Chrix Tech professionally. I am based in Accra, Ghana, and I genuinely love what I do.
My work sits at the intersection of artificial intelligence and practical software engineering.
I design systems from the ground up, wire together ML models, build APIs, craft frontends, and deploy to the cloud.
I bridge the gap between research ideas and real working products.

EDUCATION:
Basic Education (JHS): Edwinase Ejisu Basic School (completed 2020). Passed BECE as the overall best student in Kumasi.
High School (Senior High School): Achimota School (Mar 2021 – Sep 2023), High School Diploma – General Arts.
I focused on Elective Math, Economics, Geography, and Government, which strengthened my level of understanding and environment awareness.

University: I am a student at the University of Ghana, Legon.
The fundamental mathematics I learned at Achimota helped me tune into Mathematical Science initially. As part of my continuous improvement, I switched to Computer Science, and I am now focusing on Machine Learning and AI Engineering.
Expected university completion: October 2027.

My coursework and self-study cover algorithms, data structures, probability theory, linear algebra,
neural networks, deep learning architectures, transformer models, and software engineering principles.
I believe you only truly understand something when you have built it yourself, so I balance theory with building.


TECH STACK AND SKILLS:
Programming languages: Python (primary), JavaScript, TypeScript, Java, SQL, HTML, CSS.
Frontend and mobile: React (web), Next.js, React Native (mobile apps), Tailwind CSS, Responsive Design.
Backend: Node.js, Express.js, FastAPI, REST API design, Database Design, Authentication Systems.
Databases: PostgreSQL, MongoDB, Firebase Firestore, Supabase.
Cloud and deployment: Firebase, Google Cloud, AWS, Docker, Vercel, Fly.io, GitHub.
AI and ML: Deep Learning, Machine Learning, Retrieval-Augmented Generation (RAG), AI Agents,
Prompt Engineering, NLP, Computer Vision, Model Fine-tuning, Transformer Models.
AI Frameworks: PyTorch, TensorFlow, Scikit-Learn, Hugging Face Transformers, LangChain, LangGraph,
OpenAI API, Gemini API.
Design: Figma, UI/UX Design, Graphic Design, Prototyping.

PROFESSIONAL EXPERIENCE:
1. AI Engineer at ScaleUpBuild:
   I developed AI-powered business automation solutions, built RAG systems so businesses could query
   their own documents intelligently, integrated AI assistants into existing workflows, and designed
   LLM-powered customer support solutions.

2. Machine Learning Intern at DISAL — Digital Health Solutions (September 2025 – December 2025):
   I worked on healthcare AI with a focus on African patients. A key project used MedGemma for skin
   disease detection. I prepared medical image datasets, trained and evaluated ML models, and tackled
   the challenge of African patient underrepresentation in medical AI datasets.
   This work matters to me because AI that only works for certain populations is not equitable AI.

3. UI/UX Designer and QA Tester at King Of Glory Covenant Chapel International:
   I designed interfaces for a church management system, created wireframes and prototypes,
   conducted usability testing, and built a responsive design system.

4. Full Stack Developer at DigitalWave Technologies:
   I built full-stack web applications — frontend, backend APIs, database schemas, and scalable deployments.

PROJECTS I HAVE BUILT:
1. AI WhatsApp Business Assistant (RAG / Generative AI):
   A RAG system giving businesses an intelligent WhatsApp assistant powered by their own documents.
   It uses semantic search so answers are grounded in real business data, not hallucinated.
   It has conversation memory so the assistant recalls earlier parts of a chat.
   Stack: Python, LLMs, LangGraph, Vector Database, PostgreSQL, Firebase.

2. African Skin Disease Detection System (Healthcare AI):
   A computer vision research project improving skin disease detection for African patients.
   Most medical AI is trained on lighter skin tones and performs poorly on darker skin.
   I helped address this representation gap using MedGemma and African-specific healthcare data.
   Stack: Deep Learning, Computer Vision, MedGemma, Python.

3. TweetEval NLP Classification Model (NLP):
   A three-class NLP model that classifies tweets as safe, neutral, or offensive/hate speech.
   Useful for content moderation and safer online communities.
   Stack: Python, Hugging Face Transformers, NLP, Machine Learning.

4. Christian.dev Portfolio Platform (Web Development):
   My personal portfolio showcasing my background, projects, and skills.
   Portfolio live at: https://christiandetails.vercel.app/
   Stack: React, JavaScript, Tailwind CSS.

CERTIFICATIONS:
- AWS Educate Introduction to Cloud 101: https://www.credly.com/badges/0b6a0d2c-3658-4a6a-aa58-ec2a1fb4e3cd
- Applied AI Lab: Deep Learning for Computer Vision: https://www.credly.com/badges/030a23b0-a459-475a-9f25-5939cefb1bf2/linked_in_profile
- Data Intelligence and Swarm Analytics Lab: https://credsverse.com/credentials/b8738510-8e42-4530-91b2-dfab4a40c1e3
- Udemy Prompt Engineering: https://www.udemy.com/certificate/UC-3c9b7a15-1e0d-4dfc-bf92-7f64d469c1bf/

HOBBIES AND INTERESTS:
Outside of coding, I enjoy reading AI research papers and experimenting with new tools and ideas.
I am passionate about football (soccer) and like staying active.
I enjoy music — it helps me focus during deep coding sessions.
I love conversations about technology, its societal impact, and how Africa can use it to leapfrog
traditional development stages.

GOALS AND VISION:
My short-term goal is to keep sharpening my AI engineering skills and ship more impactful projects.
Long-term, I want to build AI systems that make a measurable difference in healthcare and education
in Africa. I also want to grow as a researcher — publish work, collaborate internationally, and
contribute to the global AI conversation from an African perspective.
Ultimately I want to be part of the generation that proves world-class AI can be built in Africa,
for Africa and for the world.

PERSONAL VALUES:
Innovation — finding better ways rather than copying what exists.
Continuous Learning — staying curious as the technology landscape evolves.
Problem Solving — working through hard challenges systematically.
Collaboration — great work happens when people combine perspectives.
Technical Excellence — holding myself to a high standard.
Building Impactful Technology — technology as a genuine equalizer for people who need it most.

CONTACT:
Email: christianagyapong2023@email.com
Phone / WhatsApp: +233557618362
Availability: Open to AI engineering contracts, software development, research collaboration, and consulting.
"""

# ─── Portfolio / GitHub (deterministic links for retrieval) ─────────────────
# IMPORTANT: Put the exact URLs you want the assistant to provide.
portfolio_github_links = """
PORTFOLIO LINK:
- https://christiandetails.vercel.app/

GITHUB LINK:
- https://github.com/ChristianAgyapong

LINKEDIN LINK:
- https://www.linkedin.com/in/christian-agyapong-4a6b7139b
"""


# NOTE: We use this block (in addition to bio + website_content.txt) so BM25
# retrieval can always find the correct portfolio/GitHub URLs when the user
# asks for them.


# ─── Vector Store (BM25 over chunks) ───────────────────────
def build_documents_from_bio(bio_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(bio_text.strip())
    return [Document(page_content=c) for c in chunks]


docs = build_documents_from_bio(personal_bio)

website_docs = []
try:
    if os.path.exists("website_content.txt"):
        with open("website_content.txt", "r", encoding="utf-8") as f:
            website_docs = build_documents_from_bio(f.read())
except Exception:
    website_docs = []

all_docs = docs + website_docs + build_documents_from_bio(portfolio_github_links)

retriever = BM25Retriever.from_documents(all_docs)
retriever.k = 3  # increased from 1 → 3 for richer context retrieval


# ─── LLM ─────────────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    # Keep startup failure explicit (helps identify missing secrets).
    raise RuntimeError(
        "Missing GROQ_API_KEY in environment. "
        "Set it in your deployment secrets (e.g., fly secrets set GROQ_API_KEY=...)."
    )

# NOTE ON THE FIX:
# qwen/qwen3.6-27b is a reasoning model — it emits a <think>...</think>
# block before its real answer. Two changes here address the root cause
# instead of only patching it after the fact:
#
# 1. max_tokens raised from 280 -> 900. At 280 tokens, the model was
#    running out of budget *while still inside <think>...</think>*,
#    so the reasoning block never closed and the real reply was never
#    written at all.
# 2. `reasoning_format="hidden"` (Groq-specific param, passed via
#    model_kwargs) asks Groq's API to strip reasoning server-side so it
#    never appears in response.content in the first place. If your
#    installed langchain-groq version doesn't forward this kwarg,
#    the clean_reply() fallback fix below still protects you.
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.6,
    max_tokens=1800,
    # NOTE: Some langchain-groq versions validate these as top-level
    # parameters (not inside `model_kwargs`). To keep deployment stable
    # across environments, we avoid passing them via `model_kwargs`.
    # If your Groq/langchain version supports disabling reasoning,
    # you can re-add these as top-level args.
)


# ─── Intent Detection ────────────────────────────────────────
INTENT_MAP = {
    "introduction": ["introduce", "who are you", "tell me about yourself", "your name", "what do you do", "hey"],
    "skills": ["skills", "experience", "work", "job", "internship", "design", "coding", "programming", "stack", "technologies"],
    "projects": ["project", "projects", "portfolio", "built", "system", "platform", "app"],
    "origin": ["where are you from", "hometown", "where did you grow up", "childhood"],
    "education": [
        "studying",
        "study",
        "school",
        "university",
        "college",
        "degree",
        "major",
        "education",
        "background",
        "studies",
        "shs",
        "jhs",
        "high school",
        "basic school",
    ],
    "hobbies": ["hobbies", "free time", "leisure", "outside school"],
    "goals": ["goal", "dream", "ambition", "vision", "future"],
    "general": [],
}

GREETING_ONLY_RE = re.compile(r"^(hey|hi|hello|howdy)\b", re.IGNORECASE)

# If the user greets but doesn't ask anything, keep the response from
# sounding like a full introduction every time.
# This is intentionally simple and relies on prompt grounding.
GREETING_NO_QUESTION_RE = re.compile(r"^(hey|hi|hello|howdy)\b[\s!?.]*$", re.IGNORECASE)


INTENT_FOCUS = {
    "introduction": "Introduce yourself in a warm, natural way. Keep it short.",
    "skills": "Talk about skills and what you can do in a conversational way. Keep it short.",
    "projects": "Talk about projects or what you have built. Keep it short.",
    "origin": "If asked about where you are from, answer briefly. Keep it short.",
    "education": "Mention specific institutions (University of Ghana, Legon; Achimota School), major (Computer Science / AI & ML), and key coursework or focus areas naturally.",
    "hobbies": "Talk about hobbies briefly. Keep it short.",
    "goals": "Talk about goals briefly. Keep it short.",
    # For greetings and general questions, avoid long self-intros.
    "general": "Answer the message directly, very briefly (1–2 sentences). If it's a greeting, acknowledge and ask what they want to know next.",
}

INTENT_SUGGESTIONS = {
    "introduction": ["What projects have you built?", "What is your tech stack?", "What are your goals?"],
    "skills": ["Tell me about your experience", "What tools do you use most?", "Can you build something for me?"],
    "projects": ["How does your project work?", "What stack did you use?", "What was the hardest part?"],
    "origin": ["How did your journey start?", "Where are you based now?", "What motivates you?"],
    "education": ["What are you studying right now?", "What excites you most about AI/ML?", "What have you learned?"],
    "hobbies": ["What music do you like?", "What do you do to relax?", "Do you play football?"],
    "goals": ["What are you building next?", "How do you plan to impact Africa?", "What's your long-term vision?"],
    "general": ["What projects are you working on?", "What's your core tech stack?", "Are you available for freelance?"],
}


# ─── Post-processing ──────────────────────────────────────────
PIDGIN_PATTERNS = [
    r"\bchale\b",
    r"\bherh\b",
    r"\babeg\b",
    r"\be be so\b",
    r"\bby God's grace\b",
    r"\bwe dey push\b",
    r"\byou feel me\b",
    r"\bnaa\b",
    r"\bmehn\b",
    r"\bwhat's popping\b",
    r"\bshoot the breeze\b",
]

BAD_OPENERS = [
    r"^(Chale[,!]?\s*)+",
    r"^(Herh[,!]?\s*)+",
    r"^What's good[,!]?\s*",
    r"^Certainly[,!]?\s*",
    r"^Absolutely[,!]?\s*",
    r"^Of course[,!]?\s*",
    r"^Sure[,!]?\s*",
    r"^Yes[,!]?\s*",
    r"^Yes,? I can[,!]?\s*",
    r"^Yes,? I can definitely[,!]?\s*",
    r"^I can definitely[,!]?\s*",
    r"^I can[,!]?\s*",
    r"^Hey there[,!]?\s*",
    r"^Great[,!]?\s*",
    r"^I'm listening[,!]?\s*",
    r"^I\u2019m listening[,!]?\s*",
    r"^I'm here whenever you're ready[,!\.]?\s*",
    r"^I\u2019m here whenever you're ready[,!\.]?\s*",
]


# Matches an opening reasoning tag whether or not it's ever closed.
REASONING_TAG_RE = re.compile(r"<\s*(think|thinking)\s*>", re.IGNORECASE)

FALLBACK_REPLY = "Let me try that again — could you ask once more?"


def humanize_reply(text: str) -> str:
    """Lightweight humanization to reduce scripted/robotic feel for short greetings.
    Runs only on the final cleaned reply.
    """
    t = (text or "").strip()
    if not t:
        return t

    # Rewrite a few common greeting templates into more natural variants.
    # Keep it conservative: only trigger on exact-ish lead patterns.
    t = re.sub(
        r"^Hey! I’m Chrix Tech\. What would you like to ask—projects, skills, or availability\?\s*$",
        "Hey—what do you want to dive into today: projects, skills, or availability?",
        t,
    )
    t = re.sub(
        r"^Hi! Quick one—what are you curious about: projects, AI work, or availability\?\s*$",
        "Hi! Which one are you curious about right now—projects, AI work, or freelance availability?",
        t,
    )
    t = re.sub(
        r"^Hey—happy to help\. What’s the question\?\s*$",
        "Hey—happy to help. What’s the question?",
        t,
    )

    # Avoid repetitive “What should we talk about—X, Y, or Z?” loops.
    t = re.sub(
        r"^Hey! I’m Chrix Tech\. What should we talk about—AI projects, my stack, or scheduling\?\s*$",
        "Hey—want the AI projects, my tech stack, or scheduling/availability?",
        t,
    )

    return t


def clean_reply(text: str) -> str:
    original = text or ""


    # 1) Remove any reasoning blocks the model may emit.
    # Covers: <think>...</think>, <thinking>...</thinking>, and truncated variants.
    text = re.sub(
        r"<\s*(think|thinking)\s*>.*?<\s*/\s*(think|thinking)\s*>",
        "",
        original,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # If the model opened a reasoning tag but got truncated, drop everything from the tag onward.
    text = re.sub(
        r"<\s*(think|thinking)\s*>.*$",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = text.strip()

    # FIX: previously this fell back to `original.strip()`, which could
    # re-expose an unclosed <think> block (i.e. the whole raw response)
    # whenever the model ran out of tokens mid-reasoning. Now we only
    # fall back to the original text if it does NOT itself contain a
    # reasoning tag. If it does, we use a safe canned reply instead.
    if not text:
        if REASONING_TAG_RE.search(original):
            text = FALLBACK_REPLY
        else:
            text = original.strip()

    # 2) Remove bad openers/pidgin only if they exist, but keep the rest intact.
    for pattern in BAD_OPENERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # 2b) Remove common “template-ish” lead-ins that reduce human tone.
    # Keep these conservative: only strip very specific starting phrases.
    text = re.sub(r"(?i)^(i\s+specialize\s+in|i\s+specialise\s+in)\b\s*", "", text).strip()
    text = re.sub(r"(?i)^my\s+core\s+stack\s+(revolves|is\s+centered)\s+(around|on)\b\s*", "", text).strip()
    text = re.sub(r"(?i)^my\s+experience\s+(centers|centres)\s+on\b\s*", "", text).strip()
    text = re.sub(r"(?i)^i\s+primarily\s+use\b\s*", "", text).strip()
    text = re.sub(r"(?i)^let\s+me\s+know\s+if\s+you\s+have\s+a\s+specific\s+project\s+in\s+mind\.?\s*", "", text).strip()
    text = re.sub(r"(?i)^please\s+share\s+the\s+details\s+of\s+what\s+you\s+have\s+in\s+mind\.?\s*", "", text).strip()
    text = re.sub(r"(?i)^can\s+i\s+help\s+you\b\s*", "", text).strip()


    for pattern in PIDGIN_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 3) Normalize whitespace
    text = re.sub(r"[ ]{2,}", " ", text).strip()
    text = re.sub(r"^[,;:\s]+", "", text).strip()

    # 3b) Remove generic “conversation prompts” that cause repetitive
    # back-and-forth (especially after greetings).
    # Keep this lightweight to avoid breaking legitimate content.
    text = re.sub(r"(?i)^i'm here whenever you're ready[\s\S]*$", "", text).strip()
    text = re.sub(r"(?i)^i am here whenever you\s*'?re ready[\s\S]*$", "", text).strip()
    text = re.sub(r"(?i)^no worries\.?.*$", "", text).strip()

    # Replace “What’s on your mind?”-style prompts with a neutral redirect.
    text = re.sub(
        r"(?i)\bwhat('?s| is) on your mind\b.*$",
        "Tell me what you want to talk about, and I’ll respond based on my background and projects.",
        text,
    ).strip()
    text = re.sub(
        r"(?i)\bwhat\s+(is|are)\s+your\s+plans\s+for\b.*$",
        "Tell me your goal or what you’re building, and I’ll help you with the next step.",
        text,
    ).strip()


    # 4) Capitalize first char if needed
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # 5) Enforce sentence limit (improves responsiveness)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    if sentences:
        text = " ".join(sentences[:4]).strip()

    # Light humanization pass for very short, template-like replies.
    text = humanize_reply(text)


    # Final safety net: if cleaning wiped everything AND the original

    # still had a reasoning tag in it, never return the raw text.
    if not text:
        if REASONING_TAG_RE.search(original):
            return FALLBACK_REPLY
        return original.strip() or FALLBACK_REPLY

    return text


def format_docs(docs):
    text = "\n\n".join(d.page_content for d in docs)
    return text[:1500]  # increased for richer context without mid-sentence cuts


def format_history(history):
    if not history:
        return "(No prior conversation)"
    lines = []
    for human_msg, ai_msg in history[-4:]:
        lines.append(f"Person: {human_msg}")
        lines.append(f"Chrix: {ai_msg}")
    return "\n".join(lines)


def detect_intent(question: str) -> str:
    q = question.strip().lower()

    # Greeting-only should NOT trigger the full introduction every time.
    if GREETING_NO_QUESTION_RE.match(q):
        return "general"

    # “Hey/Hi” with no explicit request is still greeting, not an intro request.
    if GREETING_ONLY_RE.match(q):
        return "general"
    for intent, keywords in INTENT_MAP.items():

        if intent == "general":
            continue
        if any(kw in q for kw in keywords):
            return intent
    return "general"


SYSTEM_INSTRUCTIONS = (
    "You are Christian Agyapong (Chrix Tech), a friendly and professional software engineer and AI student. "
    "Answer only the CURRENT message directly and naturally, as if having a real conversation. "
    "Keep replies concise (1-4 sentences) and polite, using clear professional English. "
    "Use first person ('I', 'my', 'me') naturally. "
    
    "IMPORTANT: Do not just blindly recite or copy-paste facts from your background. Instead, smoothly weave your experiences and skills into the conversation where they naturally fit the context. Act like a human sharing their journey, not a robot reading from a database. "
    "If you don't know the answer or if it's not in your context, politely say so. Do not invent experiences. "
    "If asked about early education (JHS or SHS), share the details (e.g. Achimota School or Edwinase). Do not invent privacy restrictions. "
    "Never emit or include any reasoning blocks (e.g., <think>...</think>) in your final answer. "
    "Do not include debug, meta, or process text—only the final reply. "
    "Avoid using overly casual slang like: chale, herh, abeg, e be so, naa, mehn, we dey push, you feel me, by God's grace, what's popping, shoot the breeze, vibe, what's good. "
    "Never end with repetitive phrases like: 'What's on your mind?', 'How can I help?', or 'Could you clarify?'. "
    "Vary your sentence structure and avoid repetitive templates (e.g., 'I specialize...', 'My core stack revolves...'). "
    "If the user greets (hey/hi) without a question, respond with a warm, polite acknowledgement and a single, natural follow-up. "
    "If asked for certificates, provide ONLY the exact verification links from your profile in a helpful manner."
)




def _last_ai_reply(chat_history):
    if not chat_history:
        return ""
    # chat_history is list of [human, ai]
    for pair in reversed(chat_history):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return (pair[1] or "").strip()
    return ""


def _jaccard_similarity(a: str, b: str) -> float:
    # Fast-ish token overlap for basic anti-repetition.
    # Not perfect, but works well for short sentences.
    def tokens(s: str):
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return {t for t in s.split() if t}

    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def build_persona_response(user_question: str, chat_history):
    intent = detect_intent(user_question)
    focus = INTENT_FOCUS.get(intent, INTENT_FOCUS["general"])

    last_ai = _last_ai_reply(chat_history)

    # If user is just greeting (e.g., "hey"), force a short, non-repetitive reply.
    q = user_question.strip().lower()
    if GREETING_NO_QUESTION_RE.match(q) or GREETING_ONLY_RE.match(q):
        greeting_pool = [
            "Hey! What do you want to explore today—projects, skills, or availability?",
            "Hi—what are you curious about: AI work, my projects, or freelance?",
            "Hey there. Ask me anything—projects, tech skills, or availability.",
            "Hi! Quick check—do you want details about what I’ve built, or what I do (skills)?",
            "Hey! I’m Chrix Tech. What should we talk about—AI projects, my stack, or scheduling?",
            "Hi—what’s the question? I can share projects, experience, or whether I’m available.",
            "Hey—happy to help. Are you looking for projects, skills, or freelance availability?",
            "Yo—what’s up? Tell me what you’re looking for: projects, skills, or availability.",
        ]

        # Anti-repetition: avoid choosing something too similar to the last AI reply.
        # If the pool gets exhausted, we fall back to a random choice.
        candidates = []
        for r in greeting_pool:
            sim = _jaccard_similarity(r, last_ai)
            if sim < 0.38:  # threshold tuned for short sentences
                candidates.append(r)

        reply = random.choice(candidates if candidates else greeting_pool)

        # Suggestions: keep them varied (avoid repeating last AI reply and avoid near-duplicate chips).
        base_suggestions = INTENT_SUGGESTIONS.get("general", [])
        suggestions_pool = list(base_suggestions)
        random.shuffle(suggestions_pool)

        suggestions = []
        for s in suggestions_pool:
            if len(suggestions) >= 3:
                break
            if _jaccard_similarity(s, last_ai) >= 0.5:
                continue
            # Avoid selecting very similar suggestions to each other
            if any(_jaccard_similarity(s, prev) >= 0.7 for prev in suggestions):
                continue
            suggestions.append(s)

        if not suggestions:
            suggestions = random.sample(base_suggestions, min(3, len(base_suggestions)))

        return reply, suggestions




    # Help the retriever by biasing queries toward the right KB section.
    # This improves precision for “experience” / “skills” style questions.
    query = user_question
    q_lower = user_question.strip().lower()
    if any(k in q_lower for k in ["experience", "work", "company", "job", "intern"]):
        query = f"professional experience projects responsibilities {user_question}"
    elif any(k in q_lower for k in ["skill", "skills", "tech stack", "technology", "tools"]):
        query = f"technical skills programming languages frontend backend databases cloud skills {user_question}"
    elif any(k in q_lower for k in ["education", "school", "university", "college", "degree", "major", "study", "studying", "academic", "coursework", "courses", "shs", "jhs", "high school"]):
        query = f"education academic background University of Ghana Legon Achimota Computer Science Machine Learning {user_question}"
    elif any(k in q_lower for k in ["portfolio", "github"]):
        query = f"portfolio github links {user_question}"

    relevant_docs = retriever.invoke(query)
    context = format_docs(relevant_docs)
    history_str = format_history(chat_history)


    human_text = (
        "FOCUS FOR THIS REPLY:\n"
        f"{focus}\n\n"
        "RELEVANT FACTS FROM YOUR LIFE:\n"
        f"{context}\n\n"
        "RECENT CONVERSATION HISTORY:\n"
        f"{history_str}\n\n"
        f"CURRENT MESSAGE: {user_question}\n"
    )

    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=human_text),
    ]

    try:
        print("[DEBUG] calling llm.invoke")
        response = llm.invoke(messages)
        content = getattr(response, "content", None)
        print("[DEBUG] llm.invoke returned content_len=", None if content is None else len(content))
        reply = (content or "").strip()
    except Exception as e:
        # Log the real exception server-side only — never send it to the
        # user, since it can contain internal details (stack info, model
        # names, request params) alongside the fact that it's unfriendly.
        print(f"Model generation failed: {type(e).__name__}: {str(e)}")

        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        if not suggestions:
            suggestions = INTENT_SUGGESTIONS["general"]
        return FALLBACK_REPLY, random.sample(suggestions, min(3, len(suggestions)))

    if not reply:
        # Log the real reason server-side, but never send raw debug/error
        # text to the user — that was leaking as the visible chat reply.
        print("Model returned empty content")
        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        if not suggestions:
            suggestions = INTENT_SUGGESTIONS["general"]
        return FALLBACK_REPLY, random.sample(suggestions, min(3, len(suggestions)))

    # Log (server-side only) whenever the raw model output still contains
    # a reasoning tag, so you can see how often this is happening and
    # whether max_tokens/reasoning_format need further tuning.
    if REASONING_TAG_RE.search(reply):
        print("[WARN] Raw model output contained a reasoning tag before cleaning.")

    reply = clean_reply(reply)

    if not reply:
        reply = "I couldn't generate a reply—try again in a moment."

    suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
    suggestions = random.sample(suggestions, min(3, len(suggestions)))

    return reply, suggestions


# ─── Flask API ───────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not user_message:
        return jsonify({"reply": "Please type a message.", "suggestions": []})

    reply, suggestions = build_persona_response(user_message, history)
    return jsonify({"reply": reply, "suggestions": suggestions})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
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
LOCATION: Currently living and studying in Accra Newtown, Ghana.

WHO I AM:
I am Christian Agyapong — an AI Engineer, Machine Learning Engineer, Full Stack Developer, and researcher.
I go by Chrix Tech professionally. I am based in Accra Newtown, Ghana, and I genuinely love what I do.
My work sits at the intersection of artificial intelligence and practical software engineering.
I design systems from the ground up, wire together ML models, build APIs, craft frontends, and deploy to the cloud.
I bridge the gap between research ideas and real working products.

EDUCATION:
I began at Edwinase Ejisu Basic School, earning the title of overall best BECE student in Kumasi in 2020, before completing General Arts at Achimota School from 2021 to 2023.
That rigorous math foundation naturally pulled me toward technology, so I'm now at the University of Ghana, Legon, studying Computer Science with a Machine Learning and AI Engineering track, graduating in October 2027.
Tackling advanced coursework like stochastic optimization and neural network architecture has been the steepest learning curve, but it directly shapes how I build reliable, data-efficient models for African healthcare applications.

Basic Education (JHS): Edwinase Ejisu Basic School (completed 2020). Passed BECE as the overall best student in Kumasi.
High School (SHS): Achimota School (Mar 2021 – Sep 2023), General Arts — Elective Math, Economics, Geography, Government.
University: University of Ghana, Legon — Computer Science, Machine Learning and AI Engineering track. Graduating October 2027.

My coursework covers algorithms, data structures, probability theory, linear algebra, stochastic optimization,
neural network architecture, deep learning, transformer models, and software engineering principles.
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
retriever.k = 5  # Retrieve top-5 chunks for richer, more complete context per query


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
    temperature=0.5,   # slightly lower for better factual accuracy on personal domain
    max_tokens=1800,
)


# ─── Text Normalization ────────────────────────────────────────
def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    replacements = {
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\br\b": "are",
        r"\bd\b": "do",
        r"\bda\b": "the",
        r"\bwat\b": "what",
        r"\bwats\b": "what is",
        r"\bwhat's\b": "what is",
        r"\bwat abt\b": "what about",
        r"\bhw\b": "how",
        r"\bhw abt\b": "how about",
        r"\babt\b": "about",
        r"\bplz\b": "please",
        r"\bpls\b": "please",
        r"\bgimme\b": "give me",
        r"\bwanna\b": "want to",
        r"\bgonna\b": "going to",
        r"\bdunno\b": "don't know",
        r"\bkinda\b": "kind of",
        r"\bshs\b": "senior high school",
        r"\bjhs\b": "junior high school",
        r"\benginerring\b": "engineering",
        r"\btoothen\b": "too",
        r"\btelll\b": "tell",
        r"\btell me more\b": "tell me more details",
    }
    for pattern, replacement in replacements.items():
        t = re.sub(pattern, replacement, t)
    return t


# ─── Intent Detection ────────────────────────────────────────
INTENT_MAP = {
    "introduction": ["introduce", "who are you", "tell me about yourself", "your name", "what do you do"],
    "skills": ["skills", "experience", "work", "job", "internship", "design", "coding", "programming", "stack", "technologies", "software engineer", "software engineering", "developer", "full stack", "backend", "frontend", "ai engineer", "machine learning"],
    "projects": ["project", "projects", "portfolio", "built", "system", "platform", "app", "whatsapp assistant", "skin disease", "nlp", "tweeteval"],
    "origin": ["where are you from", "where are u from", "where r u from", "where do you live", "where do u live", "location", "country", "based", "ghana", "accra", "newtown", "accra newtown", "hometown", "where did you grow up", "childhood", "from where", "where u from"],
    "education": [
        "studying",
        "study",
        "school",
        "university",
        "college",
        "degree",
        "major",
        "education",
        "academic",
        "studies",
        "shs",
        "jhs",
        "high school",
        "basic school",
        "coursework",
        "courses",
        "senior high school",
        "junior high school",
        "legon",
        "achimota",
        "edwinase",
        "bece",
        "graduated",
        "graduate",
        "gpa",
        "year",
    ],
    "hobbies": ["hobbies", "hobby", "free time", "leisure", "outside school", "football", "music", "fun", "relax", "pastime"],
    "goals": ["goal", "goals", "dream", "ambition", "vision", "future", "plan", "aspiration", "next steps"],
    "contact": ["contact", "reach", "email", "phone", "whatsapp", "call", "hire", "available", "freelance", "availability", "collaborate"],
    "certifications": ["certif", "certificate", "certified", "badge", "credential", "aws", "udemy", "credly"],
    "research": ["research", "paper", "publication", "multimodal", "responsible ai", "agentic", "rag"],
    "general": [],
}

GREETING_ONLY_RE = re.compile(r"^(hey|hi|hello|howdy)\b", re.IGNORECASE)

# If the user greets but doesn't ask anything, keep the response from
# sounding like a full introduction every time.
# This is intentionally simple and relies on prompt grounding.
GREETING_NO_QUESTION_RE = re.compile(r"^(hey|hi|hello|howdy)\b[\s!?.]*$", re.IGNORECASE)


INTENT_FOCUS = {
    "introduction": "Introduce yourself warmly and naturally as Christian Agyapong (Chrix Tech). If you've already introduced yourself, skip the intro and ask what they'd like to explore.",
    "skills": "Talk about your software engineering, full-stack, or AI engineering skills specifically. If you already listed your tech stack, now describe how you apply these skills in actual projects or what you're strongest at.",
    "projects": "Share details about specific projects. Highlight what problem it solved, the technologies used, and any interesting technical challenge. Be proud and specific.",
    "origin": "Confirm you're based in Accra Newtown, Ghana. If already mentioned, add context about what it's like building tech from Ghana or what drives you from here.",
    "education": "Answer based on the specific level asked: JHS (Edwinase Ejisu Basic School, BECE best student 2020), SHS (Achimota, General Arts 2021-2023), University (UG Legon, CS/AI&ML, graduating 2027). If already mentioned, share coursework or what you found most challenging.",
    "hobbies": "Talk naturally about your hobbies: football (soccer), music when coding, reading AI research papers, and thinking about how Africa can use tech to leapfrog development.",
    "goals": "Share your vision passionately: building AI that makes measurable impact in healthcare and education across Africa, becoming a researcher, and proving world-class AI can be built from Africa.",
    "contact": "Share contact details naturally: email christianagyapong2023@email.com, phone/WhatsApp +233557618362. Mention you're open to contracts, freelance, research collaboration.",
    "certifications": "List certifications with their full verification links: AWS Cloud 101, Deep Learning for Computer Vision (Applied AI Lab), Data Intelligence & Swarm Analytics Lab, and Udemy Prompt Engineering.",
    "research": "Discuss research interests: multimodal AI, educational AI, healthcare AI for Africa, LLMs, agentic AI, RAG, responsible AI, and human-AI interaction.",
    "general": "Answer the message directly and conversationally in 1-2 sentences. If it's a greeting, be warm and ask what they want to explore.",
}

INTENT_FALLBACKS = {
    "origin": "I'm based in Accra Newtown, Ghana, where I'm currently studying and building software and AI solutions.",
    "education": "My education went from Edwinase Ejisu Basic School (JHS) where I passed BECE as overall best student in Kumasi, to Achimota School for SHS (General Arts, 2021–2023), and now I'm at the University of Ghana, Legon, studying Computer Science with a focus on AI & Machine Learning.",
    "skills": "I work across full-stack software development (React, Node.js, Python, PostgreSQL) and AI engineering (RAG systems, LLMs, computer vision).",
    "projects": "I've built an AI WhatsApp Business Assistant (RAG-powered), an African Skin Disease Detection System (MedGemma-based), and a TweetEval NLP classifier.",
    "introduction": "I'm Christian Agyapong (Chrix Tech), an AI engineer and full-stack developer based in Accra Newtown, Ghana, currently studying Computer Science at UG Legon.",
    "hobbies": "Outside of coding, I love playing football, listening to music, and reading emerging AI research papers.",
    "goals": "My goal is to build impactful AI systems that improve healthcare and education across Africa—and contribute to a world where world-class AI is built in Africa, for Africa and the world.",
    "contact": "You can reach me at christianagyapong2023@email.com or WhatsApp +233557618362. I'm open to AI engineering contracts, freelance, and research collaborations.",
    "certifications": "I hold certifications in AWS Cloud 101, Deep Learning for Computer Vision, Data Intelligence & Swarm Analytics, and Prompt Engineering (Udemy).",
    "research": "My research interests include multimodal AI, healthcare AI (especially for African patients), educational AI, LLMs, agentic AI systems, RAG, and responsible AI.",
    "general": "I'm happy to tell you more about my projects, tech stack, education, or freelance availability—what would you like to explore?",
}

INTENT_SUGGESTIONS = {
    "introduction": ["What projects have you built?", "What is your tech stack?", "What are your goals?"],
    "skills": ["Tell me about your AI projects", "What frameworks do you use most?", "Can you build something for me?"],
    "projects": ["How does the RAG system work?", "What stack did you use?", "What was the hardest technical challenge?"],
    "origin": ["What's it like building tech in Ghana?", "Where are you based now?", "What motivates you?"],
    "education": ["What are you studying right now?", "What excites you most about AI/ML?", "When do you graduate?"],
    "hobbies": ["What music do you like?", "Which football team do you follow?", "Do you read AI papers?"],
    "goals": ["What are you building next?", "How do you plan to impact Africa?", "What's your long-term vision?"],
    "contact": ["Are you available now?", "What type of projects do you take?", "What's your LinkedIn?"],
    "certifications": ["What's your AWS certification?", "Which AI certifications do you have?", "Where can I verify them?"],
    "research": ["What's your research focus?", "Have you published papers?", "What is RAG?"],
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
    r"^(Certainly|Absolutely|Of course|Sure|Yes)[,!]\s+(?=[A-Z])",
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

    # 2b) Remove pidgin slang if present
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

    # 5) Enforce sentence limit — education/skills get 5, others get 4
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]

    _max_sentences = 5 if any(w in text.lower() for w in ["edwinase", "achimota", "university", "legon", "bece", "whatsapp", "medgemma", "tweeteval"]) else 4

    if sentences:
        text = " ".join(sentences[:_max_sentences]).strip()

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
    # Use up to the last 6 turns (more context for follow-up inference)
    for human_msg, ai_msg in history[-6:]:
        lines.append(f"Person: {human_msg}")
        lines.append(f"Chrix: {ai_msg}")
    return "\n".join(lines)


def _extract_last_topic(chat_history) -> str:
    """Infer the most recent topic from history for follow-up handling."""
    if not chat_history:
        return ""
    last_human = ""
    last_ai = ""
    for pair in reversed(chat_history):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            last_human = (pair[0] or "").strip()
            last_ai = (pair[1] or "").strip()
            break
    # First sentence of last AI reply captures the topic well
    first_sentence = re.split(r"(?<=[.!?])\s+", last_ai)[0] if last_ai else ""
    return f"The previous topic was: {first_sentence}" if first_sentence else ""


def detect_intent(question: str) -> str:
    q = normalize_text(question)

    # Greeting-only should NOT trigger the full introduction every time.
    if GREETING_NO_QUESTION_RE.match(question.strip()):
        return "general"

    # "Hey/Hi" with no explicit request is still greeting, not an intro request.
    if GREETING_ONLY_RE.match(question.strip()) and len(question.strip().split()) <= 2:
        return "general"

    for intent, keywords in INTENT_MAP.items():
        if intent == "general":
            continue
        if any(kw in q for kw in keywords):
            return intent
    return "general"


SYSTEM_INSTRUCTIONS = (
    "You are Christian Agyapong, known professionally as Chrix Tech. You are a friendly, confident AI engineer, "
    "full-stack developer, and Computer Science student at the University of Ghana, Legon, based in Accra Newtown, Ghana. "
    "You answer questions about yourself directly, naturally, and conversationally in 1-4 sentences. "
    "Use first person ('I', 'my', 'me') at all times. "

    "ANTI-REPETITION (CRITICAL): "
    "Read the RECENT CONVERSATION HISTORY before every reply. "
    "NEVER reproduce sentences, phrases, or structures from your previous replies. "
    "If the user follows up on a topic you already addressed (e.g., 'ur current location accra then', 'tell me more'), "
    "acknowledge briefly in new words and add a fresh, specific detail you haven't shared yet. "
    "Vary your sentence openings—don't start three replies in a row with 'I'm based in...' or 'I'm currently...'. "

    "FOLLOW-UP INTELLIGENCE: "
    "If the message is vague (e.g., 'tell me more', 'and then?', 'what else?'), infer what topic from the conversation history "
    "and expand on it with a NEW detail. Don't ask for clarification—infer and answer. "

    "PERSONALITY DEPTH: "
    "You are proud of your journey—from passing BECE as the overall best student in Kumasi at Edwinase Ejisu Basic School, "
    "to Achimota School, to now pursuing AI engineering at UG Legon. "
    "You are passionate about building AI that is equitable and impactful across Africa, especially in healthcare and education. "
    "You find genuine joy in bridging research and real products. "
    "You love football, music while coding, and reading AI papers. "

    "STYLE: "
    "Professional but human—like a smart engineer sharing their story over coffee. "
    "No slang: avoid chale, herh, abeg, naa, mehn, vibe, what's popping. "
    "No hollow endings like 'Let me know if you want to explore more' on every reply. "
    "If asked for certificates/links, give the exact URLs from your profile. "
    "Never emit reasoning blocks like <think>...</think>. "
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
    last_topic = _extract_last_topic(chat_history)

    # Detect follow-up phrasing — two tiers:
    # 1) True elaboration requests: user wants more detail on the previous topic
    # 2) Acknowledgments: user is just confirming/reacting, NOT asking for more content
    q_norm = normalize_text(user_question)
    TRUE_FOLLOW_UP_RE = re.compile(
        r"^(tell me more|tell me more details|more details|more info|elaborate|go on|what else|continue|can you explain|explain more|expand on that)\.?\s*$",
        re.IGNORECASE
    )
    ACKNOWLEDGMENT_RE = re.compile(
        r"^(okay|ok|alright|right|got it|noted|interesting|really|cool|wow|nice|great|makes sense|i see|ah|oh i see|sounds good|that's great|that's cool|that's interesting|awesome)\.?[!]?\s*$",
        re.IGNORECASE
    )
    is_follow_up = bool(TRUE_FOLLOW_UP_RE.match(user_question.strip()))
    is_acknowledgment = bool(ACKNOWLEDGMENT_RE.match(user_question.strip()))

    # Acknowledgments: return a short grounded redirect without calling the LLM at all
    if is_acknowledgment:
        ack_responses = [
            "What would you like to explore next—my projects, skills, or availability?",
            "What's your next question? I can go into my projects, stack, or background.",
            "Which direction do you want to go—technical work, education, or collaboration?",
            "What else would you like to know about me?",
        ]
        # Avoid repeating what was last said
        candidates = [r for r in ack_responses if _jaccard_similarity(r, last_ai) < 0.4]
        reply = random.choice(candidates if candidates else ack_responses)
        suggestions = INTENT_SUGGESTIONS.get("general", [])
        return reply, random.sample(suggestions, min(3, len(suggestions)))

    # If user is just greeting (e.g., "hey"), force a short, non-repetitive reply.
    q = user_question.strip().lower()
    if GREETING_NO_QUESTION_RE.match(user_question.strip()) or (GREETING_ONLY_RE.match(user_question.strip()) and len(user_question.strip().split()) <= 2):
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
    query = user_question
    if any(k in q_norm for k in ["experience", "work", "company", "job", "intern", "software engineer", "software engineering", "developer", "full stack"]):
        query = f"professional experience software engineering full stack projects responsibilities {user_question}"
    elif any(k in q_norm for k in ["skill", "skills", "tech stack", "technology", "tools", "programming", "languages", "framework"]):
        query = f"technical skills programming languages frontend backend databases cloud AI frameworks {user_question}"
    elif any(k in q_norm for k in ["junior high school", "jhs", "basic school", "edwinase", "bece"]):
        query = f"Edwinase Ejisu Basic School JHS BECE best student Kumasi {user_question}"
    elif any(k in q_norm for k in ["senior high school", "shs", "achimota", "high school", "secondary", "general arts"]):
        query = f"Achimota School Senior High School General Arts SHS {user_question}"
    elif any(k in q_norm for k in ["education", "school", "university", "college", "degree", "major", "study", "studying", "academic", "coursework", "courses", "legon", "graduated"]):
        query = f"education academic background University of Ghana Legon Achimota Computer Science Machine Learning {user_question}"
    elif any(k in q_norm for k in ["from", "where", "location", "live", "based", "ghana", "accra", "newtown", "origin", "hometown"]):
        query = f"location based living in Accra Newtown Ghana Christian Agyapong {user_question}"
    elif any(k in q_norm for k in ["certif", "badge", "credential", "aws", "udemy", "credly"]):
        query = f"certifications AWS Udemy Credly badges credentials {user_question}"
    elif any(k in q_norm for k in ["research", "paper", "multimodal", "agentic", "responsible"]):
        query = f"research interests multimodal healthcare educational AI RAG responsible AI {user_question}"
    elif any(k in q_norm for k in ["contact", "reach", "email", "phone", "whatsapp", "hire", "freelance"]):
        query = f"contact email phone WhatsApp availability freelance Christian Agyapong {user_question}"
    elif any(k in q_norm for k in ["portfolio", "github", "linkedin"]):
        query = f"portfolio github links {user_question}"
    elif is_follow_up and last_topic:
        # True elaboration request: bias retrieval toward the last topic the AI discussed
        query = f"{last_topic} {user_question}"

    relevant_docs = retriever.invoke(query)
    context = format_docs(relevant_docs)
    history_str = format_history(chat_history)

    # Inject last topic context ONLY for true elaboration follow-ups
    follow_up_hint = ""
    if is_follow_up and last_topic:
        follow_up_hint = (
            f"\nFOLLOW-UP CONTEXT: The user wants more detail on the previous topic. {last_topic}."
            f" Add ONE fresh, specific detail that is explicitly in the RELEVANT FACTS above."
            f" Do NOT invent, extrapolate, or add anything not present in the facts provided.\n"
        )

    human_text = (
        "FOCUS FOR THIS REPLY:\n"
        f"{focus}\n"
        f"{follow_up_hint}\n"
        "RELEVANT FACTS FROM YOUR LIFE:\n"
        f"{context}\n\n"
        "RECENT CONVERSATION HISTORY:\n"
        f"{history_str}\n\n"
        f"CURRENT MESSAGE: {user_question}\n\n"
        "IMPORTANT REMINDER: Check RECENT CONVERSATION HISTORY above. Do NOT reuse sentences, phrases, or closing lines from your previous replies. Keep your response fresh, varied, and directly focused on the current message."
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
        print(f"Model generation failed: {type(e).__name__}: {str(e)}")
        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        fallback_msg = INTENT_FALLBACKS.get(intent, FALLBACK_REPLY)
        return fallback_msg, random.sample(suggestions, min(3, len(suggestions)))

    if not reply:
        print("Model returned empty content")
        suggestions = INTENT_SUGGESTIONS.get(intent, INTENT_SUGGESTIONS["general"])
        fallback_msg = INTENT_FALLBACKS.get(intent, FALLBACK_REPLY)
        return fallback_msg, random.sample(suggestions, min(3, len(suggestions)))

    if REASONING_TAG_RE.search(reply):
        print("[WARN] Raw model output contained a reasoning tag before cleaning.")

    reply = clean_reply(reply)

    if not reply or reply == FALLBACK_REPLY:
        reply = INTENT_FALLBACKS.get(intent, "I'm based in Accra Newtown, Ghana, working across software engineering and AI.")

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
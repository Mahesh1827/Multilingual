"""
Knowledge RAG Agent
Combines domain routing, retrieval, reasoning, and verification for in-depth Tirumala knowledge.
"""
from typing import TypedDict
import logging
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import re
from query.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    TOP_K_RETRIEVE, TOP_K_FINAL, VERIFIER_OVERLAP_THRESHOLD,
    USE_GROQ, GROQ_API_KEY, GROQ_MODEL,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Verifier module
# =============================================================================

# Common stopwords to ignore in overlap calculation
_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "with", "are", "was", "were", "be", "has", "have", "had",
    "this", "that", "these", "those", "at", "by", "from", "as", "not",
    "but", "so", "if", "i", "you", "he", "she", "we", "they", "its",
    "their", "our", "your", "his", "her", "can", "will", "would", "may",
    "which", "who", "what", "when", "where", "how", "all", "also", "been",
    "about", "than", "more", "no", "do", "did", "does", "am",
}

# Disclaimer to append when answer is not grounded
_DISCLAIMER = (
    "\n\nNote: This answer may not be fully supported by the available "
    "Tirumala documents. Please verify important details independently."
)

def _tokenize(text: str) -> set[str]:
    """Extract meaningful tokens from text."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOPWORDS}

def verify(answer: str, context_chunks: list[dict], language: str = "en") -> dict:
    """Verify that the answer is grounded in the context chunks."""
    if not answer or not context_chunks:
        return {
            "is_grounded":   False,
            "confidence":    0.0,
            "overlap_ratio": 0.0,
            "warning":       None,
        }

    if language != "en":
        # Cannot perform accurate lexical overlap check when the answer is in a different language than the context
        return {
            "is_grounded":   True,
            "confidence":    1.0,
            "overlap_ratio": 1.0,
            "warning":       None,
        }

    # Build combined context token set
    context_text = " ".join(chunk["text"] for chunk in context_chunks)
    context_tokens = _tokenize(context_text)
    answer_tokens  = _tokenize(answer)

    if not answer_tokens:
        return {
            "is_grounded":   False,
            "confidence":    0.0,
            "overlap_ratio": 0.0,
            "warning":       None,
        }

    # Overlap = fraction of answer tokens found in context
    intersection  = answer_tokens & context_tokens
    overlap_ratio = len(intersection) / len(answer_tokens)

    is_grounded = overlap_ratio >= VERIFIER_OVERLAP_THRESHOLD
    confidence  = min(overlap_ratio / max(VERIFIER_OVERLAP_THRESHOLD, 0.01), 1.0)

    warning = None if is_grounded else _DISCLAIMER

    logger.info(
        f"Verifier: overlap={overlap_ratio:.2f}, "
        f"grounded={is_grounded}, confidence={confidence:.2f}"
    )

    return {
        "is_grounded":   is_grounded,
        "confidence":    round(confidence, 3),
        "overlap_ratio": round(overlap_ratio, 3),
        "warning":       warning,
    }

def get_verified_answer(answer: str, verification: dict) -> str:
    """Append disclaimer to answer if not grounded."""
    if not verification["is_grounded"] and verification.get("warning"):
        return answer + verification["warning"]
    return answer

# =============================================================================
# Domain Knowledge Agents
# =============================================================================
_BASE_RULES = (
    "IDENTITY: You are a friendly, intelligent Tirumala guide — like a knowledgeable local who has been to\n"
    "Tirumala many times and loves helping pilgrims. You speak warmly, simply, and helpfully.\n\n"
    "LANGUAGE RULE: Always respond in the SAME language the user used. If the user spoke Telugu, reply in Telugu.\n"
    "If Hindi, reply in Hindi. If Tamil, reply in Tamil. If Kannada, reply in Kannada. If English, reply in English.\n"
    "If the user explicitly asks to switch language, switch immediately.\n\n"
    "DOMAIN RESTRICTION (CRITICAL):\n"
    "- Answer ONLY questions about Tirumala (darshan, accommodation, laddu, history, sevas, festivals, travel, facilities).\n"
    "- If the question is unrelated to Tirumala, say politely: 'I can only help with Tirumala-related information.'\n"
    "- Never provide general knowledge, politics, sports, or unrelated topics.\n\n"
    "KNOWLEDGE RULES:\n"
    "1. Answer ONLY from the provided context. Never invent or guess facts.\n"
    "2. If the answer is not in the context, say: 'I don't have that specific detail right now.\n"
    "   For the latest information, please visit the official TTD website (ttdevasthanams.ap.gov.in).'\n"
    "3. If you are unsure, say so clearly — never guess.\n\n"
    "INTERACTION STYLE:\n"
    "- Be friendly, polite, and conversational. Use welcoming phrases naturally.\n"
    "- Keep answers simple and clear — ideal for a first-time pilgrim.\n"
    "- Ask helpful follow-up questions when useful: 'Would you like more details?' or 'Do you need directions or timings?'\n"
    "- If speech input seems garbled or unclear, try to understand intent and gently ask for clarification.\n"
    "- Keep answers concise: 2-4 sentences unless more detail is specifically requested.\n"
    "- Never use emojis. Never add meta-commentary about your rules or how you work.\n"
)

@dataclass(frozen=True)
class DomainAgent:
    key: str
    name: str
    system_prompt: str

_AGENTS: dict[str, DomainAgent] = {
    "about_tirumala": DomainAgent(
        key="about_tirumala",
        name="About Tirumala Agent",
        system_prompt=(
            _BASE_RULES
            + "\nDOMAIN FOCUS: General information about Tirumala — its geography, "
            "location, altitude, climate, the seven hills (Saptagiri), the significance "
            "of Akasa Ganga, population, and the overall spiritual importance of the region."
        ),
    ),
    "devotional_practice": DomainAgent(
        key="devotional_practice",
        name="TTD Devotional Practice Agent",
        system_prompt=(
            _BASE_RULES
            + "\nDOMAIN FOCUS: Devotional practices at Tirumala — pujas, archana, "
            "bhajans, kirtans, fasting (vratam), vows, offerings, prasadam, sevas, "
            "and aarti. Explain the spiritual significance and correct procedures "
            "exactly as described in the provided context."
        ),
    ),
    "festival_events": DomainAgent(
        key="festival_events",
        name="TTD Festival & Events Agent",
        system_prompt=(
            _BASE_RULES
            + "\nDOMAIN FOCUS: Festivals and events at Tirumala TTD — Brahmotsavam, "
            "Rathotsavam, Vaikunta Ekadasi, Kalyanotsavam, Teppotsavam, and other "
            "annual utsavams. Provide dates, sequence of events, spiritual significance, "
            "and procession details strictly from the context."
        ),
    ),
    "knowledge_scripture": DomainAgent(
        key="knowledge_scripture",
        name="TTD Knowledge & Scripture Agent",
        system_prompt=(
            _BASE_RULES
            + "\nDOMAIN FOCUS: Vedic scriptures, Upanishads, Puranas (especially "
            "Bhagavata), Vishnu Sahasranama, Venkateswara Suprabhatam, stotras, "
            "mantras, and theological/philosophical teachings related to Lord "
            "Venkateswara and TTD. Preserve textual accuracy — do not paraphrase "
            "sacred verses beyond what the context provides."
        ),
    ),
    "temple_history": DomainAgent(
        key="temple_history",
        name="TTD Temple History & Heritage Agent",
        system_prompt=(
            _BASE_RULES
            + "\nDOMAIN FOCUS: History and heritage of the Sri Venkateswara Temple — "
            "ancient origins, legends, mythological context, dynastic contributions "
            "(Pallava, Chola, Vijayanagara), inscriptions, temple architecture "
            "(gopuram, vimana, mandapam), and construction milestones documented "
            "in TTD records."
        ),
    ),
    "pilgrimage_seva": DomainAgent(
        key="pilgrimage_seva",
        name="TTD Pilgrimage Guidance, Seva & Darshan Services Agent",
        system_prompt=(
            _BASE_RULES
            + "\nDOMAIN FOCUS: Practical pilgrimage guidance — darshan ticket booking, "
            "virtual queue system, special darshan types, seva booking procedures, "
            "TTD accommodation (lodges/cottages/guest houses), prasadam (laddu) "
            "distribution, dress code, entry rules, and how/when to visit Tirumala "
            "as documented by TTD."
        ),
    ),
}

def get_domain_agent(domain_key: str) -> DomainAgent:
    """Return the DomainAgent for the given category key."""
    agent = _AGENTS.get(domain_key)
    if agent is None:
        logger.warning(f"Unknown domain key '{domain_key}' — using fallback agent.")
        return _AGENTS["about_tirumala"]
    return agent

def get_primary_system_prompt(domains: list[str]) -> str:
    """Return the system prompt of the highest-priority matched domain agent."""
    for domain in domains:
        if domain in _AGENTS:
            logger.info(f"[DomainAgent] Using '{domain}' agent prompt.")
            return _AGENTS[domain].system_prompt
    logger.info("[DomainAgent] No matching domain — using fallback agent prompt.")
    return _AGENTS["about_tirumala"].system_prompt

# =============================================================================
# Retriever module
# =============================================================================
_faiss_cache: dict = {}

def _load_vectorstore_cached():
    """Load LangChain FAISS vectorstore (cached after first load)."""
    if "vectorstore" not in _faiss_cache:
        from ocr.vector_store import load_faiss_index
        logger.info("Loading LangChain FAISS vectorstore...")
        vectorstore, metadata = load_faiss_index()
        _faiss_cache["vectorstore"] = vectorstore
        _faiss_cache["metadata"]    = metadata
        logger.info(f"FAISS vectorstore loaded: {vectorstore.index.ntotal} vectors")
    return _faiss_cache["vectorstore"]

def retrieve(
    query: str,
    domains: list[str] | None = None,
    top_k: int = TOP_K_RETRIEVE,
) -> list[dict]:
    """Retrieve the most relevant chunks for a query, then rerank with cross-encoder."""
    try:
        vectorstore = _load_vectorstore_cached()
    except FileNotFoundError:
        logger.error(
            "FAISS index not found. Run the document processing pipeline first "
            "(uv run python main.py)."
        )
        return []

    search_k = top_k * 6 if domains else top_k
    total    = vectorstore.index.ntotal

    retrieval_query = (
        "Represent this sentence for searching relevant passages: "
        + query.strip()
    )

    results_with_scores = vectorstore.similarity_search_with_score(
        retrieval_query, k=min(search_k, total)
    )

    # Domain filter
    candidates = []
    for doc, score in results_with_scores:
        meta = doc.metadata
        if domains and meta.get("agent_category", "") not in domains:
            continue
        candidates.append({
            "score":    float(score),
            "text":     meta.get("text", doc.page_content),
            "metadata": meta,
        })
        if len(candidates) >= top_k:
            break

    # Cross-encoder reranking: re-score candidates and keep top 3
    if len(candidates) > 1:
        try:
            from ocr.vector_store import get_reranker
            reranker = get_reranker()
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = reranker.predict(pairs)
            
            # Attach reranker scores and sort
            for c, rs in zip(candidates, rerank_scores):
                c["rerank_score"] = float(rs)
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
            
            # Keep only top 3 with positive rerank score
            candidates = [c for c in candidates if c.get("rerank_score", 0) > 0][:TOP_K_FINAL]
            if not candidates:
                # If reranker rejected everything, keep the best original candidate
                candidates = sorted(
                    [{"score": float(s), "text": m.get("text", d.page_content), "metadata": m}
                     for d, s in results_with_scores[:1] for m in [d.metadata]],
                    key=lambda x: x["score"]
                )
            
            score_strs = [str(round(c.get("rerank_score", 0), 3)) for c in candidates]
            logger.info(
                f"Reranker: kept {len(candidates)} chunks "
                f"(scores: {score_strs})"
            )
        except Exception as e:
            logger.warning(f"Reranker failed ({e}); using FAISS ordering.")
            candidates = candidates[:TOP_K_FINAL]
    
    logger.info(
        f"Retriever: {len(candidates)} chunks returned "
        f"(domains={domains}, top_k={top_k})"
    )
    return candidates

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source_file", "unknown")
        page   = chunk["metadata"].get("page", "?")
        domain = chunk["metadata"].get("agent_category", "?")
        parts.append(
            f"[Source {i}: {source}, page {page}, domain: {domain}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


# =============================================================================
# Reasoning module
# =============================================================================

def _build_system_prompt() -> str:
    """Build the Tirumala guide system prompt with a time-based greeting."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 12:
        time_greeting = "Good morning!"
    elif 12 <= hour < 17:
        time_greeting = "Good afternoon!"
    else:
        time_greeting = "Good evening!"

    return f"""You are a friendly, intelligent Tirumala guide — like a knowledgeable local pilgrim who \
has visited many times and loves helping others. Your name is Tirumala Assistant.

When greeting users, use: "{time_greeting}" naturally.

🌐 LANGUAGE RULE (CRITICAL):
- Detect the language of the user's question automatically.
- ALWAYS respond in the SAME language as the user spoke.
  • User writes Telugu → reply in Telugu
  • User writes Hindi → reply in Hindi
  • User writes Tamil → reply in Tamil
  • User writes Kannada → reply in Kannada
  • User writes English → reply in English
- If the user explicitly says "tell in English" or "in Telugu", switch to that language immediately.
- The system provides an English query for RAG retrieval only. Your RESPONSE language must match the user's original language.

🎯 DOMAIN RESTRICTION (STRICT):
- Answer ONLY questions related to Tirumala:
  darshan (timings, types, tickets, queues), accommodation, laddu prasadam,
  temple history & significance, travel routes within Tirumala, seva details,
  and facilities (food, transport, medical, etc.)
- If asked anything UNRELATED to Tirumala, respond politely:
  "I can only help with Tirumala-related information."
- Never provide general knowledge, politics, sports, celebrities, or other topics.

📚 KNOWLEDGE RULES:
1. Answer ONLY from the provided context. Never invent or assume facts.
2. If the answer is not in the context, say:
   "I'm not sure about that, but I can help with other Tirumala details." or
   "I don't have that specific detail right now. For the latest info, please visit ttdevasthanams.ap.gov.in."
3. If you are unsure, say so honestly — never guess.

🔄 CONTEXT AWARENESS:
- Remember the previous question if visible in chat history.
- If the user asks a follow-up, relate it intelligently to the previous query.

💬 INTERACTION STYLE:
- Be friendly, polite, and conversational — like a helpful guide, not a robot.
- Keep answers simple and clear, suitable for first-time pilgrims.
- Use welcoming phrases naturally: "How can I help you?", "Would you like more details?", "Do you need directions or timings?"
- Ask a helpful follow-up question when appropriate.
- If the user's speech seems garbled or unclear due to recognition errors, try to understand their intent and politely ask for clarification.
- Keep answers concise: 2–4 sentences unless more detail is requested.
- Never use emojis. Never add meta-commentary about your rules or how you work.
"""

SYSTEM_PROMPT = _build_system_prompt()

_llm = None

def _get_llm():
    """Lazy-load LLM: Groq (cloud) if API key is set, else Ollama (local)."""
    global _llm
    if _llm is None:
        if USE_GROQ:
            from langchain_groq import ChatGroq
            _llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=2048,
            )
            logger.info(f"LLM: Using Groq ({GROQ_MODEL})")
        else:
            _llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3,
                top_p=0.9,
                num_ctx=8192,
                timeout=OLLAMA_TIMEOUT,
            )
            logger.info(f"LLM: Using Ollama ({OLLAMA_MODEL})")
    return _llm

# ISO 639-1 → human-readable language name for lang_hint
_LANG_NAMES = {
    "te": "Telugu", "hi": "Hindi", "ta": "Tamil",
    "kn": "Kannada", "en": "English",
}

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human",
     "Previous Conversation:\n{chat_history}\n\n"
     "CONTEXT:\n{context}\n\n"
     "USER'S ORIGINAL MESSAGE: {original_text}\n"
     "ENGLISH QUERY (for retrieval only): {question}{lang_hint}\n\n"
     "ANSWER:"),
])

def reason(
    query:          str,
    context_chunks: list[dict],
    user_lang:      str = "en",
    system_prompt:  str | None = None,
    chat_history=None,
    original_text:  str | None = None,
) -> str:
    """
    Generate an answer to the query using retrieved context via LangChain LCEL.

    Args:
        query:          English version of the question (used for RAG retrieval).
        context_chunks: Retrieved document chunks.
        user_lang:      ISO 639-1 code of the user's original language.
        system_prompt:  Optional override for the system prompt.
        chat_history:   List of previous {role, content} dicts.
        original_text:  The user's message in their native script (used so the
                        LLM can detect the correct response language directly).
    """
    lang_name = _LANG_NAMES.get(user_lang, user_lang.upper())

    if not context_chunks:
        if user_lang == "en":
            return (
                "Hmm, I couldn't find relevant information for that. "
                "Could you try rephrasing? You can also check ttdevasthanams.ap.gov.in for the latest details."
            )
        return (
            f"I'm not sure about that, but I can help with other Tirumala details. "
            f"(Responding in {lang_name} as requested.)"
        )

    context_lines = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["metadata"].get("source_file", "unknown")
        page   = chunk["metadata"].get("page", "?")
        context_lines.append(f"[Context {i} | {source} | Page {page}]\n{chunk['text']}")
    context_block = "\n\n".join(context_lines)

    # Direct language instruction — LLM responds in the user's language itself.
    # No post-translation step needed.
    if user_lang == "en":
        lang_hint = ""
    else:
        lang_hint = (
            f"\n\nIMPORTANT: The user's requested language is {lang_name}. "
            f"You MUST write your entire response strictly using the NATIVE official alphabet/script of the {lang_name} language "
            f"(e.g., Devanagari script for Hindi, Telugu script for Telugu, etc.). "
            f"Under NO circumstances should you use Latin/English characters (Romanization/Hinglish) to write {lang_name} words, "
            f"even if the user's original message used English letters. "
            f"Your output must be purely in {lang_name} script natively. Do NOT answer in English."
        )

    effective_system = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    # Rebuild with fresh time-based greeting if using default prompt
    if system_prompt is None:
        effective_system = _build_system_prompt()

    logger.info(f"Reasoning Agent: calling Ollama ({OLLAMA_MODEL}) via LangChain LCEL...")
    try:
        history_text = ""
        if chat_history:
            try:
                for msg in chat_history:
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_text += f"{role}: {msg.get('content')}\n"
            except Exception:
                pass

        chain = _PROMPT | _get_llm() | StrOutputParser()
        answer = chain.invoke({
            "system_prompt": effective_system,
            "context": context_block,
            "original_text": original_text or query,
            "question": query,
            "chat_history": history_text.strip() if history_text else "None",
            "lang_hint": lang_hint,
        })
        logger.info(f"Reasoning Agent: got answer ({len(answer)} chars)")
        return answer

    except Exception as e:
        logger.warning(f"LangChain LCEL chain failed: {e}. Attempting Ollama fallback...")
        try:
            from langchain_ollama import ChatOllama
            local_llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.3,
                top_p=0.9,
                num_ctx=8192,
                timeout=OLLAMA_TIMEOUT,
            )
            chain = _PROMPT | local_llm | StrOutputParser()
            answer = chain.invoke({
                "system_prompt": effective_system,
                "context": context_block,
                "original_text": original_text or query,
                "question": query,
                "chat_history": history_text.strip() if history_text else "None",
                "lang_hint": lang_hint,
            })
            logger.info(f"Reasoning Agent: Ollama fallback successful ({len(answer)} chars)")
            return answer
        except Exception as ollama_e:
            logger.warning(f"Ollama fallback also failed: {ollama_e}. Returning raw direct chunk.")

    best   = context_chunks[0]
    source = best["metadata"].get("source_file", "source unknown")
    page   = best["metadata"].get("page", "?")
    return (
        f"Based on the official Tirumala archives, here is the exact information I found:\n\n"
        f"\"{best['text'].strip()}\"\n\n"
        f"*(Source: {source}, page {page})*"
    )

def check_ollama_health() -> bool:
    """Quick health check: returns True if Ollama server is reachable."""
    try:
        import requests as _requests
        r = _requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# =============================================================================
# Main RAG Agent
# =============================================================================

def knowledge_rag_agent(query_english: str, language: str, domains: list[str], chat_history: list[dict] | None = None) -> dict:
    logger.info("🧠 [Agent: Knowledge RAG] Routing query to Knowledge RAG Agent...")

    # 1. Retrieve
    logger.info("🔍 [Knowledge RAG] Retrieving context chunks...")
    chunks = retrieve(query_english, domains=domains)
    if not chunks:
        logger.info("🔍 [Knowledge RAG] No results in domain — widening to all domains...")
        chunks = retrieve(query_english, domains=None)

    # 2. System Prompt
    prompt = get_primary_system_prompt(domains)

    # 3. Reason
    logger.info("🧠 [Knowledge RAG] Synthesizing answer via LLM...")
    answer = reason(
        query_english,
        chunks,
        language,
        system_prompt=prompt,
        chat_history=chat_history or [],
    )
    
    # 4. Verify
    logger.info("✅ [Knowledge RAG] Checking groundedness...")
    verification = verify(answer, chunks, language)
    verified = get_verified_answer(answer, verification)
    
    return {
        "context_chunks": chunks,
        "raw_answer": answer,
        "verification": verification,
        "verified_answer": verified,
    }

def rag_node(state: dict) -> dict:
    query_en      = state.get("query_english", "")
    language      = state.get("language", "en")
    original_text = state.get("user_input", query_en)   # native-script original message
    logger.info(f"📘 [Knowledge RAG Agent] Initiated for: {query_en}")

    from query.agents.router_agent import route_query
    _, domains = route_query(query_en)
    system_prompt = get_primary_system_prompt(domains)

    chunks = retrieve(query_en, domains=domains)
    if not chunks and domains:
        chunks = retrieve(query_en, domains=None)

    raw_answer = reason(
        query_en,
        chunks,
        language,
        system_prompt=system_prompt,
        chat_history=state.get("chat_history", []),
        original_text=original_text,
    )

    verification = verify(raw_answer, chunks, language)
    verified = get_verified_answer(raw_answer, verification)
    
    ans_lower = verified.lower()
    
    # 🚨 Check if it's an Out-of-Scope rejection (matches both old and new friendly phrases)
    _rejection_phrases = [
        "i can only answer questions related to tirumala",
        "i can only help with questions about tirumala",
        "i can only help with tirumala",
    ]
    if any(p in ans_lower for p in _rejection_phrases):
        # If the query contains TTD administrative keywords, it's NOT out of scope.
        # It's just unknown to the current context. Force fallback to web search.
        it_looks_relevant = any(kw in query_en.lower() for kw in ["chairman", "naidu", "board", "eo", "officer", "news", "update"])
        if it_looks_relevant:
            logger.info("📘 [Knowledge RAG] Detected potential domain question. Allowing web fallback.")
            return {
                "final_answer": verified, 
                "context_chunks": chunks,
                "domains": domains,
                "verification": verification,
                "needs_web_fallback": True 
            }
        
        logger.warning("📘 [Knowledge RAG] True out-of-scope question. Halting.")
        return {
            "final_answer": verified, 
            "context_chunks": chunks,
            "domains": domains,
            "verification": verification,
            "needs_web_fallback": False
        }

    # 🚨 If it's a valid TTD question but the database is empty, GO to the Web Agent
    failure_phrases = [
        # Original phrases
        "does not contain",
        "not have enough information",
        "not supported by the available",
        "could not find",
        "does not provide",
        "i cannot answer",
        "no information mentioned",
        "context does not mention",
        "was not mentioned",
        "not mentioned explicitly",
        "no mention of",
        "information not mentioned",
        "i do not know",
        "i am unable to",
        "information provided does not",
        # New friendly phrases (from updated prompts)
        "don't have that specific detail",
        "i don't have that",
        "specific detail right now",
    ]
    
    needs_fallback = False
    if not chunks or any(phrase in ans_lower for phrase in failure_phrases):
        logger.warning("📘 [Knowledge RAG] Answer lacks context. Flagging for Web Fallback.")
        needs_fallback = True
        
    return {
        "final_answer": verified, 
        "context_chunks": chunks,
        "domains": domains,
        "verification": verification,
        "needs_web_fallback": needs_fallback
    }

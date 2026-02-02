# =========================
# AI COURTROOM - app.py (v2.0 Adversarial Edition)
# =========================
# Dependencies: 
# pip install fastapi uvicorn pydantic numpy faiss-cpu duckduckgo-search sentence-transformers google-generativeai

import os
import sqlite3
import uvicorn
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# =========================
# CONFIGURATION
# =========================

API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
genai.configure(api_key=API_KEY)

# Using Flash for agents (speed) and Pro (or Flash) for Judge
MODEL_NAME = "models/gemini-1.5-flash" 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
FAISS_INDEX_FILE = "courtroom_memory.index"
MEMORY_TEXTS_FILE = "courtroom_texts.pkl"

app = FastAPI(title="AI Courtroom Engine v2", description="Adversarial Legal Logic Engine")

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# =========================
# PROMPTS (ADVERSARIAL SYSTEM)
# =========================

PROSECUTOR_PROMPT = """
You are the PROSECUTOR. Your goal is to argue that the user's claim/topic is TRUE/GUILTY based on the evidence.
- Highlight evidence that supports the claim.
- Downplay contradictions as minor anomalies.
- Use persuasive, authoritative language.
- If evidence is weak, argue based on logical probability.
- Be concise (max 150 words).
"""

DEFENSE_PROMPT = """
You are the DEFENSE ATTORNEY. Your goal is to argue that the user's claim/topic is FALSE/NOT GUILTY/UNPROVEN.
- Highlight contradictions, lack of sources, or alternative explanations.
- Demand a high burden of proof.
- Attack the credibility of weak evidence.
- Be concise (max 150 words).
"""

JUDGE_PROMPT = """
You are an impartial AI High Court Judge. 
Review the CASE TOPIC, the conflicting arguments from PROSECUTOR and DEFENSE, the raw EVIDENCE, and PAST PRECEDENTS.

Structure your response exactly as follows:
1. **Analysis**: Weigh the Prosecutor vs. Defense arguments. Who had better evidence?
2. **Precedents**: Cite if past memory was relevant.
3. **Verdict**: [Must be: Guilty, Not Guilty, True, False, Proven, Debunked, or Inconclusive]
4. **Confidence**: [Number 0-100]
"""

# =========================
# DATABASE & PERSISTENCE
# =========================

conn = sqlite3.connect("courtroom.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    verdict TEXT,
    confidence INTEGER,
    summary TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# =========================
# VECTOR MEMORY (PERSISTENT)
# =========================

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(MEMORY_TEXTS_FILE):
    print("Loading persistent memory...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(MEMORY_TEXTS_FILE, "rb") as f:
        memory_texts = pickle.load(f)
else:
    print("Initializing new memory...")
    index = faiss.IndexFlatL2(VECTOR_DIM)
    memory_texts = []

def save_memory_state():
    """Persists FAISS index and text map to disk."""
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(MEMORY_TEXTS_FILE, "wb") as f:
        pickle.dump(memory_texts, f)

def store_memory(text: str):
    if not text: return
    embedding = embed_model.encode([text])
    index.add(np.array(embedding).astype('float32'))
    memory_texts.append(text)
    save_memory_state()

def retrieve_memory(query: str, k=3) -> List[str]:
    if index.ntotal == 0: return []
    q_emb = embed_model.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(memory_texts):
            results.append(memory_texts[idx])
    return results

# =========================
# EVIDENCE & LOGIC GATES
# =========================

def fetch_evidence(topic: str, max_results=5) -> List[Dict]:
    evidence = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(topic, max_results=max_results)
            if results:
                for r in results:
                    evidence.append({
                        "title": r.get("title", "Unknown"),
                        "snippet": r.get("body", "")[:300],
                        "source": r.get("href", "")
                    })
    except Exception as e:
        print(f"Search Error: {e}")
    return evidence

def evidence_diversity_penalty(evidence: List[Dict]) -> int:
    """Penalizes if all evidence comes from the same domain."""
    domains = set()
    for e in evidence:
        src = e.get("source", "")
        if "//" in src:
            try:
                domains.add(src.split("//")[1].split("/")[0])
            except: pass
    if len(domains) <= 1 and len(evidence) > 1:
        return -20
    return 0

def knowledge_sufficiency(evidence_count: int, precedent_count: int) -> bool:
    """Epistemic brake: Do we know enough to even judge?"""
    if evidence_count == 0 and precedent_count == 0:
        return False
    return True

def verdict_stability(confidence: int) -> str:
    if confidence >= 80: return "Stable"
    if confidence >= 50: return "Moderate"
    return "Fragile"

# =========================
# LLM AGENTS
# =========================

def run_agent(system_prompt: str, context: str) -> str:
    """Generic runner for Prosecutor, Defense, and Judge."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Using correct role priming: System instruction is separate in new API, 
        # but for compatibility with standard generate_content, we pass it as the first user part 
        # or use system_instruction if available. Here we use a safe User-Model pattern.
        
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": f"SYSTEM INSTRUCTION: {system_prompt}\n\nTASK CONTEXT: {context}"}
            ],
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# API ENDPOINTS
# =========================

class CourtroomRequest(BaseModel):
    topic: str = Field(..., min_length=5)

class CourtroomResponse(BaseModel):
    verdict: str
    confidence: int
    stability: str
    reasoning: Dict

@app.post("/courtroom", response_model=CourtroomResponse)
def run_courtroom_adversarial(req: CourtroomRequest):
    print(f"--- Session: {req.topic} ---")

    # 1. Gather Info
    past_memory = retrieve_memory(req.topic)
    evidence = fetch_evidence(req.topic)
    
    evidence_text = str(evidence) if evidence else "No external evidence found."
    memory_text = str(past_memory) if past_memory else "No precedents."

    # 2. Epistemic Gate
    if not knowledge_sufficiency(len(evidence), len(past_memory)):
        return {
            "verdict": "Dismissed",
            "confidence": 0,
            "stability": "None",
            "reasoning": {"error": "Insufficient data to form a court session."}
        }

    # 3. ADVERSARIAL PHASE (Prosecutor vs Defense)
    print("Running Prosecutor...")
    prosecutor_arg = run_agent(PROSECUTOR_PROMPT, f"Claim: {req.topic}\nEvidence: {evidence_text}")
    
    print("Running Defense...")
    defense_arg = run_agent(DEFENSE_PROMPT, f"Claim: {req.topic}\nEvidence: {evidence_text}")

    # 4. JUDGEMENT PHASE
    print("Running Judge...")
    judge_context = f"""
    CASE TOPIC: {req.topic}
    
    --- PROSECUTION ARGUMENT ---
    {prosecutor_arg}
    
    --- DEFENSE ARGUMENT ---
    {defense_arg}
    
    --- RAW EVIDENCE ---
    {evidence_text}
    
    --- PAST PRECEDENTS ---
    {memory_text}
    """
    
    judge_output = run_agent(JUDGE_PROMPT, judge_context)

    # 5. Parsing & Scoring
    verdict = "Inconclusive"
    model_conf = 0
    
    for line in judge_output.split('\n'):
        clean_line = line.replace("*", "").strip()
        if clean_line.startswith("Verdict:"):
            verdict = clean_line.split("Verdict:")[1].strip()
        if clean_line.startswith("Confidence:"):
            try:
                model_conf = int(float(clean_line.split("Confidence:")[1].strip().replace("%", "")))
            except: pass

    # 6. Deterministic Adjustments
    div_penalty = evidence_diversity_penalty(evidence)
    final_conf = max(0, min(model_conf + div_penalty, 100))
    
    stability = verdict_stability(final_conf)

    # 7. Persistence
    summary = f"Topic: {req.topic} | Verdict: {verdict} | Stability: {stability}"
    store_memory(summary)
    
    cursor.execute(
        "INSERT INTO cases (topic, verdict, confidence, summary) VALUES (?, ?, ?, ?)",
        (req.topic, verdict, final_conf, judge_output)
    )
    conn.commit()

    return {
        "verdict": verdict,
        "confidence": final_conf,
        "stability": stability,
        "reasoning": {
            "prosecutor_argument": prosecutor_arg,
            "defense_argument": defense_arg,
            "judge_analysis": judge_output,
            "penalties": {
                "diversity_penalty": div_penalty
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

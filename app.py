# app.py
import os
import re
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------- App config ----------
st.set_page_config(page_title="Interview Chatbot", page_icon="💬")
st.title("Interview Chatbot — TinyLlama (CPU)")

hf_token = st.secrets["HUGGINGFACE_API_KEY"]

# ---------- Setup UI state ----------
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

def complete_setup():
    st.session_state.setup_complete = True

# Setup form
if not st.session_state.setup_complete:
    st.subheader("Personal information")
    st.session_state["name"] = st.text_input("Name", value=st.session_state.get("name", ""), placeholder="Enter your name")
    st.session_state["experience"] = st.text_area("Experience", value=st.session_state.get("experience", ""), placeholder="Describe your experience")
    st.session_state["skills"] = st.text_area("Skills", value=st.session_state.get("skills", ""), placeholder="List your skills")

    st.write(f"**Your Name**: {st.session_state['name']}")
    st.write(f"**Experience**: {st.session_state['experience']}")
    st.write(f"**Skills**: {st.session_state['skills']}")

    st.subheader("Company and Position")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["level"] = st.radio("Choose level",
                                             options=["Junior", "Mid-level", "Senior"],
                                             index=["Junior", "Mid-level", "Senior"].index(st.session_state.get("level", "Junior")))
    with col2:
        st.session_state["position"] = st.selectbox("Choose a position",
                                                   ["Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"],
                                                   index=["Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"].index(st.session_state.get("position", "Data Scientist")))
    st.session_state["company"] = st.selectbox("Choose a Company",
                                               ["Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify"],
                                               index=["Amazon", "Meta", "Udemy", "365 Company", "Nestle", "LinkedIn", "Spotify"].index(st.session_state.get("company", "Amazon")))

    st.write(f"**Applying as**: {st.session_state['level']} {st.session_state['position']} at {st.session_state['company']}")

    if st.button("Start Interview", on_click=complete_setup):
        st.success("Setup complete — starting interview.")

# ---------- MODEL LOADER (cached) ----------
@st.cache_resource
def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    hf_token = st.secrets["HUGGINGFACE_API_KEY"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.to("cpu")
    model.eval()
    return tokenizer, model

# Load once
tokenizer, model = load_model()

# ---------- Sanitizers & Helpers ----------
def sanitize_user_text(text: str) -> str:
    "Trim and remove stray role labels from user text."
    if not text:
        return ""
    text = re.sub(r'(?i)\b(User|Assistant)\s*:\s*', '', text)
    return text.strip()

def sanitize_assistant_text(text: str) -> str:
    """
    Remove role labels and trailing user lines from assistant text.
    """
    if not text:
        return ""
    # If assistant produced 'Assistant:' take only what's after the last occurrence
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1]
    # If assistant somehow included 'User:' cut off anything after the first 'User:'
    if "User:" in text:
        text = text.split("User:")[0]
    text = re.sub(r'(?i)\b(User|Assistant)\s*:\s*', '', text)
    return text.strip()

def remove_role_lines(text: str, candidate_name: str = None) -> str:
    """
    Remove entire lines that start with role labels like 'User:', 'Assistant:', or '<CandidateName>:'.
    """
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = []
    cand = candidate_name or ""
    for ln in lines:
        # remove if line starts with User: or Assistant:
        if re.match(r'^\s*(User|Assistant)\s*:\s*', ln, flags=re.I):
            continue
        # remove if line starts with candidate name (e.g., "Mark:")
        if cand and re.match(r'^\s*' + re.escape(cand) + r'\s*:\s*', ln, flags=re.I):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

def build_prompt(messages, candidate_name="", candidate_experience="", candidate_skills="", position_str=""):
    """
    Safer prompt builder:
    - Provides an explicit CANDIDATE PROFILE block (used as context).
    - Adds a minimal sanitized history (only recent assistant replies + latest user input) WITHOUT role labels.
    - Ends with very explicit instructions: only output assistant content, no role labels, no invented user lines.
    """
    system = messages[0]["content"] if messages else "You are a helpful HR interviewer assistant."
    # Candidate profile block
    profile = (
        "=== CANDIDATE PROFILE ===\n"
        f"Name: {candidate_name or '(not provided)'}\n"
        f"Experience: {candidate_experience or '(not provided)'}\n"
        f"Skills: {candidate_skills or '(not provided)'}\n"
        f"Applied for: {position_str or '(not provided)'}\n"
        "=== END PROFILE ===\n\n"
    )

    # Build a short sanitized history: only keep up to last 6 turns to avoid over-conditioning
    history_lines = []
    recent = messages[-12:] if messages else []
    for m in recent:
        if m["role"] == "user":
            clean = sanitize_user_text(m["content"])
            if clean:
                history_lines.append(f"Latest user input: {clean}")
        elif m["role"] == "assistant":
            clean = sanitize_assistant_text(m["content"])
            if clean:
                history_lines.append(f"Previous assistant reply: {clean}")

    history = ""
    if history_lines:
        history = "=== SANITIZED HISTORY (most recent first) ===\n" + "\n".join(history_lines) + "\n=== END HISTORY ===\n\n"

    # Strict instruction block
    instructions = (
        "SYSTEM INSTRUCTIONS (READ CAREFULLY):\n"
        "- You are an HR interviewer. Use the CANDIDATE PROFILE above as your context.\n"
        "- Ask interview questions, evaluate answers, and give constructive feedback.\n"
        "- IMPORTANT: Output ONLY the assistant's reply. Do NOT output any role labels such as 'User:', 'Assistant:', the candidate's name followed by a colon, or simulate the user's lines.\n"
        "- Do NOT invent or repeat user's inputs verbatim as new user turns. Respond concisely and directly.\n\n"
        "### RESPONSE (assistant only):\n"
    )

    prompt = f"SYSTEM: {system}\n\n{profile}{history}{instructions}"
    return prompt

def count_tokens(text: str) -> int:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0

# ---------- Robust sync generator with cleaning ----------
def generate_response_sync(prompt: str, max_new_tokens: int = 200,
                           temperature: float = 0.35, top_p: float = 0.85,
                           repetition_penalty: float = 1.25, no_repeat_ngram_size: int = 4,
                           candidate_name: str = "") -> str:
    """
    Generate synchronously and aggressively clean role-label echoes and fabricated user lines.
    Returns assistant-only text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    # Keep only generated tokens (slice prompt length)
    gen_ids = out[0][input_ids.shape[-1]:]

    if gen_ids.numel() == 0:
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    else:
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Defensive cleaning: strip common role echoes
    # If model re-inserted 'Assistant:' take what's after the last one
    if "Assistant:" in decoded:
        decoded = decoded.split("Assistant:")[-1]
    # If model inserted 'User:' cut everything after first 'User:'
    if "User:" in decoded:
        decoded = decoded.split("User:")[0]

    # Remove lines that begin with role labels or candidate name
    decoded = remove_role_lines(decoded, candidate_name=candidate_name)

    # Remove any remaining inline role tokens
    decoded = re.sub(r'(?i)\b(User|Assistant)\s*:\s*', '', decoded)

    # Finally strip any repeated whitespace/newlines at start
    decoded = decoded.strip()

    # Safety: if decoded is empty, return a fallback
    if not decoded:
        return "Thank you — could you please elaborate?"  # short fallback if model produced nothing useful

    return decoded

# ---------- Interview stage ----------
if st.session_state.setup_complete:
    st.info("Start by introducing yourself.", icon="👋")

    # initialize messages (system prompt includes context + strict instruction)
    if "messages" not in st.session_state:
        system_msg = (
            "You are an HR executive interviewing a candidate. Keep answers professional and do not output role labels."
        )
        st.session_state.messages = [{"role": "system", "content": system_msg}]

    # display previous messages (skip system)
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Sidebar controls
    st.sidebar.subheader("Generation settings")
    max_tokens = st.sidebar.number_input("Max response tokens", min_value=32, max_value=1024, value=200)
    temp = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.35)
    top_p = st.sidebar.slider("Top-p (nucleus)", min_value=0.1, max_value=1.0, value=0.85)
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset conversation"):
        st.session_state.pop("messages", None)
        st.experimental_rerun()

    # Chat input handler
    if prompt := st.chat_input("Your answer."):
        user_text = sanitize_user_text(prompt)
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Build prompt (pass candidate profile explicitly)
        candidate_name = st.session_state.get("name", "")
        candidate_experience = st.session_state.get("experience", "")
        candidate_skills = st.session_state.get("skills", "")
        position_str = f"{st.session_state.get('level','')} {st.session_state.get('position','')} at {st.session_state.get('company','')}"

        full_prompt = build_prompt(
            st.session_state.messages,
            candidate_name=candidate_name,
            candidate_experience=candidate_experience,
            candidate_skills=candidate_skills,
            position_str=position_str
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            assistant_reply = generate_response_sync(
                full_prompt,
                max_new_tokens=int(max_tokens),
                temperature=float(temp),
                top_p=float(top_p),
                repetition_penalty=1.25,
                no_repeat_ngram_size=4,
                candidate_name=candidate_name
            )
            placeholder.markdown(assistant_reply)

        # Store sanitized assistant reply
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

        # Token usage display
        st.sidebar.subheader("Token Usage")
        st.sidebar.write(f"Prompt tokens: {count_tokens(full_prompt)}")
        st.sidebar.write(f"Response tokens: {count_tokens(assistant_reply)}")
        st.sidebar.write(f"Total tokens: {count_tokens(full_prompt) + count_tokens(assistant_reply)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 — CPU friendly. If model download requires auth, set HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`.")

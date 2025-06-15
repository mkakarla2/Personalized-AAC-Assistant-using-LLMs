import streamlit as st
import torch
import time
import os
import faiss
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib 
import json

st.set_page_config(layout="wide")
st.title("Personalized AAC Assistant")

MERGED_MODEL_PATH = "/kaggle/input/finetuned/"
RAG_INDICES_DIR = "/kaggle/working/rag_indices" 
EMBEDDING_MODEL_ID = 'all-MiniLM-L6-v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SEQ_LENGTH = 2048

@st.cache_data(show_spinner=False)
def load_user_profiles(path: str = "user_profiles.json"):
    """Load the JSON file of user profiles and return as a dict."""
    if not os.path.exists(path):
        st.error(f"Could not find user profiles file at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

USER_PROFILES = load_user_profiles("/kaggle/input/user-profiles/user_profiles.json")

@st.cache_resource
def load_models_and_embedder(merged_model_path):
    print(f"Loading models from path: {merged_model_path}")
    model, tokenizer, embedding_model = None, None, None
    try: 
        if not os.path.exists(merged_model_path): raise FileNotFoundError(merged_model_path)
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        model = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token; print(f"Set pad token: {tokenizer.pad_token}")
        print("LLM and tokenizer loaded."); 
    except Exception as e: st.error(f"LLM/Tokenizer load error: {e}"); print(f"LLM/Tokenizer load error: {e}"); import traceback; traceback.print_exc()
    try: 
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
        print("Embedding model loaded."); 
    except Exception as e: st.error(f"Embedding model load error: {e}"); print(f"Embedding model load error: {e}"); import traceback; traceback.print_exc()
    if model and tokenizer and embedding_model: print("All models loaded."); return tokenizer, model, embedding_model
    else: print("Model loading failed."); return None, None, None


if 'cached_rag_indices' not in st.session_state:
    st.session_state.cached_rag_indices = {} 

def get_or_build_rag_index(persona_key: str,
                           chunk_size: int,
                           chunk_overlap: int,
                           _embedding_model: SentenceTransformer):
    if not _embedding_model:
        st.error("Embedding model unavailable for RAG index building.")
        return None, None

    settings_hash = hashlib.md5(f"{persona_key}-{chunk_size}-{chunk_overlap}".encode()).hexdigest()
    cache_key = f"rag_{settings_hash}"

    if cache_key in st.session_state.cached_rag_indices:
        cached_data = st.session_state.cached_rag_indices[cache_key]
        return cached_data['chunks'], cached_data['index']

    st.write(f"Building RAG index for {persona_key} (Chunk: {chunk_size}, Overlap: {chunk_overlap})...")
    print(f"Building RAG index for {persona_key} with settings key: {cache_key}")

    docs_path = os.path.join(RAG_INDICES_DIR, f"docs_{persona_key}.pkl")
    if not os.path.exists(docs_path):
        st.warning(f"No personal data file found for '{persona_key}' at {docs_path}")
        return [], None 

    try:
        with open(docs_path, 'rb') as f:
            raw_docs = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load raw docs for {persona_key}: {e}")
        return [], None

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = []
    for doc in raw_docs:
        if isinstance(doc, str) and doc.strip():
            chunks.extend(splitter.split_text(doc))
    chunks = [c for c in chunks if c.strip()] 

    if not chunks:
        print(f"No valid chunks produced for {persona_key} with current settings.")
        st.session_state.cached_rag_indices[cache_key] = {'chunks': [], 'index': None}
        return [], None

    try:
        print(f"Embedding {len(chunks)} chunks for {cache_key}...")
        embeddings = _embedding_model.encode(chunks, device=DEVICE, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        print("Embedding complete.")
    except Exception as e:
        st.error(f"Failed to embed chunks for {persona_key}: {e}")
        return [], None

    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f"FAISS index built ({index.ntotal} vectors) for {cache_key}.")
    except Exception as e:
        st.error(f"Failed to build FAISS index for {persona_key}: {e}")
        return chunks, None 

    st.session_state.cached_rag_indices[cache_key] = {'chunks': chunks, 'index': index}
    st.write(f"RAG index built and cached for {persona_key}.")
    return chunks, index

def retrieve_relevant_narratives(query_text: str,persona_key: str,k: int,_embedding_model: SentenceTransformer,chunk_size: int, chunk_overlap: int):
    if not _embedding_model:
        st.error("Embedding model not available for retrieval.")
        return []
    chunks, index = get_or_build_rag_index(persona_key, chunk_size, chunk_overlap, _embedding_model)
    if not chunks or index is None:
        print(f"No valid chunks or index available for retrieval for {persona_key}.")
        return []
    try:
        q_emb, = _embedding_model.encode([query_text], device=DEVICE, convert_to_numpy=True)
        q_emb = q_emb.astype(np.float32)[None, :] 
        distances, indices = index.search(q_emb, k)
        results = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)] 
        print(f"Retrieved {len(results)} docs for query '{query_text}' ({persona_key}).")
        return results

    except Exception as search_err:
        st.error(f"Error during FAISS search for {persona_key}: {search_err}")
        print(f"Error during FAISS search for {persona_key}: {search_err}")
        return []

def clean_decode(tokenizer, token_ids):
    raw = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    b = raw.encode("latin1", errors="ignore"); u = b.decode("utf8", errors="ignore")
    return " ".join(u.split())

def format_inference_prompt(persona_key: str, user_intent: str, influencing_prompt: str, conversation_history: list[dict], retrieved_context: list[str] | None = None):
    if not user_intent: 
        print("Warning: empty user_intent."); 
        return None
    profile = USER_PROFILES.get(persona_key, USER_PROFILES["Default"]); 
    name = profile.get("name", "Assistant"); 
    summary = profile.get("persona_summary", ""); 
    style = profile.get("communication_style", "")
    
    system_prompt = f"You are {name}. {summary} Communication guideline: {style}."
    if retrieved_context: 
        context_str = "\n".join([f"- {ctx}" for ctx in retrieved_context]); 
        system_prompt += f"\n\nRelevant context:\n{context_str}"

    prompt_parts = ["<|begin_of_text|>"]; 
    prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")

    for turn in conversation_history:
        role, content = turn.get("role"), turn.get("content")
        if role and content and role.lower() in ["user", "assistant"]: 
            prompt_parts.append(f"<|start_header_id|>{role.lower()}<|end_header_id|>\n\n{content}<|eot_id|>")


    final_user_message = user_intent
    if influencing_prompt:
        final_user_message += f"\n(When responding, please consider this direction: {influencing_prompt})"
    prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{final_user_message}<|eot_id|>")
    prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_intent}<|eot_id|>")
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n"); 
    full_prompt = "".join(prompt_parts)

    print("\n--- Generated Prompt ---"); 
    print(full_prompt); 
    print("----------------------\n")
    return full_prompt

@torch.inference_mode()
def generate_multiple_responses(prompt: str, _model, _tokenizer, num_responses: int) -> list[str]:
    print(f"Generating {num_responses} responses..."); st.write("Generating responses...")
    start_time = time.time(); outputs_decoded = []
    try: primary_device = _model.device
    except Exception: primary_device = DEVICE
    print(f"Using device: {primary_device}")
    inputs = _tokenizer(prompt, return_tensors="pt").to(primary_device)
    generation_params = {"max_new_tokens": st.session_state.gen_max, "min_new_tokens": st.session_state.gen_min, "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "num_return_sequences": num_responses, "eos_token_id": _tokenizer.eos_token_id, "pad_token_id": _tokenizer.pad_token_id, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3, "early_stopping": True}
    try:
        outputs = _model.generate(**inputs, **generation_params); input_length = inputs['input_ids'].shape[1]
        for i in range(num_responses): response_ids = outputs[i][input_length:]; response = clean_decode(_tokenizer, response_ids); outputs_decoded.append(response.strip())
    except Exception as e: st.error(f"Generation error: {e}"); print(f"LLM generation error: {e}"); return [f"Error: {e}"] * num_responses
    end_time = time.time(); print(f"LLM gen took {end_time - start_time:.2f}s."); st.write(f"Generation took {end_time - start_time:.2f}s.")
    return outputs_decoded

tokenizer, model, embedding_model = load_models_and_embedder(MERGED_MODEL_PATH)
if not tokenizer or not model or not embedding_model: st.error("Model/Embedder loading failed."); st.stop()
if "messages" not in st.session_state: st.session_state.messages = {}
if 'debug_retrieved_docs' not in st.session_state: st.session_state.debug_retrieved_docs = None
if 'debug_generated_prompt' not in st.session_state: st.session_state.debug_generated_prompt = None
if 'show_debug_panel' not in st.session_state: st.session_state.show_debug_panel = False
if 'num_responses_to_generate' not in st.session_state: st.session_state.num_responses_to_generate = 2

st.sidebar.header("User Selection")
available_users = list(USER_PROFILES.keys())
if "Default" in available_users: available_users.remove("Default")
if 'selected_user' not in st.session_state or st.session_state.selected_user not in available_users:
    st.session_state.selected_user = available_users[0] if available_users else None
if st.session_state.selected_user:
    selected_user = st.sidebar.selectbox("Select User", available_users, index=available_users.index(st.session_state.selected_user), key="user_select_widget")
    st.session_state.selected_user = selected_user
    current_profile = USER_PROFILES[selected_user]
    st.sidebar.subheader(f"Current User: {current_profile['name']}")
    st.sidebar.caption(f"Style: {current_profile['communication_style']}")
    st.sidebar.caption(f"Summary: {current_profile['persona_summary']}")
else: st.sidebar.warning("No user profiles."); st.stop()

def update_influence_prompt(text_to_add, key_suffix):
    """Appends text to the influence prompt text input."""
    influence_key = f"influence_{st.session_state.selected_user}_{key_suffix}"
    current_value = st.session_state.get(influence_key, "")
    if current_value:
        new_value = f"{current_value} {text_to_add}, "
    else:
        new_value = f"{text_to_add}, "
    st.session_state[influence_key] = new_value

def display_main_content(container_key_suffix):
    with chat_container:
        current_chat_history = st.session_state.messages[selected_user]
        for message in current_chat_history:
            with st.chat_message(message["role"]): st.markdown(message["content"])

    st.subheader("Your Input")
    input_key = f"input_{selected_user}_{container_key_suffix}" 
    user_intent_input = st.text_input(f"Enter input for {selected_user}:", key=input_key, value="")
    col1, col2 = st.columns([4, 1])
    influence_key = f"influence_{selected_user}_{container_key_suffix}"
    user_direction = col1.text_input(
        f"Add Direction (Entered by {selected_user}):",
        key=influence_key,
        placeholder="e.g., agree, disagree, happy, sad, etc.",)

    num_responses_options = [1, 2, 3, 4, 5]
    default_idx = num_responses_options.index(st.session_state.num_responses_to_generate)
    st.session_state.num_responses_to_generate = col2.selectbox(f"Response Count (Selected by {selected_user}):",options=num_responses_options,index=default_idx,key=f"num_responses_select_{container_key_suffix}",)
    preset_directions = ["Agree", "Disagree", "Maybe", "Happy", "Sad", "Empathetic", "Formal", "Informal", "Humorous", "Explain more", "Concise response", "Detailed response", "Provide alternatives", "Ask a question", "Express concern",]
    buttons_per_row = 8
    num_rows = (len(preset_directions) + buttons_per_row - 1) // buttons_per_row 

    button_index = 0
    for r in range(num_rows):
        start_idx = r * buttons_per_row
        end_idx = min(start_idx + buttons_per_row, len(preset_directions))
        row_presets = preset_directions[start_idx:end_idx]
        cols = st.columns(len(row_presets))
        for i, direction in enumerate(row_presets):
            button_key = f"preset_{direction}_{r}_{container_key_suffix}" 
            cols[i].button(
                direction,
                key=button_key,
                on_click=update_influence_prompt,
                args=(direction, container_key_suffix)
            )
            button_index += 1 
        if r < num_rows - 1:
             st.write("") 
    st.markdown("---")

    if st.button("Generate Suggestions", key=f"generate_{selected_user}_{container_key_suffix}"): # Unique key
        current_input_value = st.session_state.get(input_key, "")
        current_influence_value = st.session_state.get(influence_key, "")
        if current_input_value:
            display_intent = f"(Intent: {current_input_value}" + (f" / Direction: {current_influence_value}" if current_influence_value else "") + ")"
            st.session_state.messages[selected_user].append({"role": "user", "content": display_intent})

            history_for_prompt = [msg for msg in st.session_state.messages[selected_user][:-1] if not msg["content"].startswith("(Intent:")]
            retrieved_docs = retrieve_relevant_narratives(current_input_value, selected_user, k=1, _embedding_model=embedding_model, chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
            full_prompt = format_inference_prompt(selected_user, current_input_value, current_influence_value, history_for_prompt, retrieved_docs)

            st.session_state.debug_retrieved_docs = retrieved_docs
            st.session_state.debug_generated_prompt = full_prompt

            if full_prompt:
                with st.spinner("Generating suggestions..."):
                    num_to_gen = st.session_state.num_responses_to_generate
                    suggested_responses = generate_multiple_responses(full_prompt, model, tokenizer, num_to_gen)
                st.session_state.suggestions = suggested_responses
                st.session_state.show_suggestions = True
            else:
                st.error("Failed to format prompt."); st.session_state.suggestions = []; st.session_state.show_suggestions = False
            st.rerun()
        else: st.warning("Please enter an intent.")

    if st.session_state.get('show_suggestions', False):
        st.subheader("Choose a Response:")
        suggestions = st.session_state.get('suggestions', [])
        if suggestions:
            if any("Error:" in str(s) for s in suggestions): st.error(f"Gen error: {suggestions[0]}"); st.session_state.show_suggestions = False
            else:
                cols = st.columns(len(suggestions))
                for i, response in enumerate(suggestions):
                    if cols[i].button(response, key=f"select_{selected_user}_{i}_{container_key_suffix}"): # Unique key
                        st.session_state.messages[selected_user].append({"role": "assistant", "content": response})
                        st.session_state.show_suggestions = False; st.session_state.suggestions = []
                        st.session_state.debug_retrieved_docs = None; st.session_state.debug_generated_prompt = None
                        st.rerun()
        else:
            st.write("No suggestions generated.")
            st.session_state.show_suggestions = False
            st.session_state.debug_retrieved_docs = None; st.session_state.debug_generated_prompt = None

def display_debug_content():
    st.subheader("Debug Info")
    retrieved = st.session_state.get("debug_retrieved_docs")
    if retrieved is not None: st.write("**Retrieved Docs:**"); st.write(retrieved if retrieved else "None")
    prompt_text = st.session_state.get("debug_generated_prompt")
    if prompt_text is not None: st.write("**Generated Prompt:**"); st.text_area("Prompt", prompt_text, height=300, key="debug_prompt_area_persist_unique") # Unique key

st.sidebar.markdown("---")  
st.sidebar.subheader("Chunking Settings")
if "chunk_size" not in st.session_state: st.session_state.chunk_size = 512
if "chunk_overlap" not in st.session_state: st.session_state.chunk_overlap = 50
st.session_state.chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, st.session_state.chunk_size, 50, key="chunk_size_slider")
st.session_state.chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, st.session_state.chunk_overlap, 10, key="chunk_overlap_slider")

st.sidebar.subheader("Generation Length")
if "gen_min" not in st.session_state: st.session_state.gen_min = 30
if "gen_max" not in st.session_state: st.session_state.gen_max = 250 # Increased default
st.session_state.gen_min = st.sidebar.slider("Min Response Length", 10, 200, st.session_state.gen_min, 10, key="gen_min_slider")
st.session_state.gen_max = st.sidebar.slider("Max Response Length", 50, 400, st.session_state.gen_max, 10, key="gen_max_slider") # Increased range

st.sidebar.markdown("---")  
st.sidebar.subheader("Display Options")
debug_button_text = "Hide Debug Info" if st.session_state.show_debug_panel else "Show Debug Info"
if st.sidebar.button(debug_button_text, key="toggle_debug_panel"):
    st.session_state.show_debug_panel = not st.session_state.show_debug_panel
    st.rerun() 

if selected_user not in st.session_state.messages: st.session_state.messages[selected_user] = []

st.subheader("Conversation Window") 

if st.session_state.show_debug_panel:
    main_col, debug_col = st.columns([3, 1])

    with main_col:
        chat_container = st.container(height=400)
        display_main_content("debug_shown") 
    with debug_col:
        display_debug_content()

else:
    chat_container = st.container(height=400) 
    display_main_content("debug_hidden")
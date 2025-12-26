import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from google import genai 

# Import specialized modules
from modules.medical import get_medical_response
from modules.research import search_arxiv
from modules.knowledge import handle_knowledge_update

# 1. Setup Environment and Client
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 2. Page Configuration
st.set_page_config(page_title="GenAI Enterprise Assistant", layout="wide")

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar - Mission Control
st.sidebar.title("🤖 Mission Control")

# --- NEW: SYSTEM HEALTH (TASK 1 DATABASE VERIFICATION) ---
st.sidebar.markdown("### 🛠️ System Health")
db_path = "faiss_index"
if os.path.exists(db_path):
    st.sidebar.success("✅ Knowledge Base: Connected")
else:
    st.sidebar.warning("⚠️ Knowledge Base: Offline")

st.sidebar.markdown("---")

task_mode = st.sidebar.selectbox("Select Task Mode", [
    "Task 1: Knowledge Update",
    "Task 2: Multi-modal (Text & Image)",
    "Task 3: Medical Q&A",
    "Task 4: Research Expert"
])

# Sidebar Reset Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# 4. Main Title
st.title("🚀 GenAI Enterprise Assistant")
st.markdown(f"**Status:** `Active` | **Mode:** `{task_mode}`")

# 5. Dashboard Tabs
tab_chat, tab_data, tab_logs = st.tabs(["💬 Assistant", "📊 Data Preview", "⚙️ System Logs"])

with tab_chat:
    # --- TASK-SPECIFIC SIDEBAR FILE UPLOADERS ---
    uploaded_image = None
    uploaded_kb = None

    if "Task 2" in task_mode:
        uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.sidebar.image(uploaded_image, caption="Current Input")

    elif "Task 1" in task_mode:
        uploaded_kb = st.sidebar.file_uploader("Upload .txt Knowledge", type=["txt"])
        
        # --- NEW: INDEXING BUTTON (TASK 1 REQUIREMENT) ---
        if uploaded_kb:
            if st.sidebar.button("⚙️ Process & Index Knowledge"):
                with st.spinner("Breaking text into chunks & updating Vector DB..."):
                    # We pass the file to knowledge.py to be embedded
                    handle_knowledge_update("Initial Indexing", uploaded_kb)
                    st.sidebar.success("Database Successfully Updated!")
                    st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ... (tab_data and tab_logs code remains the same as your previous version) ...

# --- THE LOGIC ROUTER ---
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Processing..."):
        # Route 1: Knowledge Update (RAG Implementation)
        if "Task 1" in task_mode:
            if os.path.exists(db_path):
                # We call the function with the prompt; it retrieves context from the DB
                final_response = handle_knowledge_update(prompt, None)
            else:
                final_response = "⚠️ No Knowledge Base found. Please upload a `.txt` file and click 'Process' in the sidebar."

        # Route 2: Medical
        elif "Task 3" in task_mode:
            final_response = get_medical_response(prompt)
        
        # Route 3: Research
        elif "Task 4" in task_mode:
            final_response = search_arxiv(prompt)

        # Route 4: Multi-modal, Sentiment, and Language (Tasks 2, 5, 6)
        else:
            # Unified instruction for automated analysis (Task 5 & 6)
            system_instruction = "Detect user's language and sentiment. Respond in the same language. Analyze images if provided."
            if uploaded_image:
                img = Image.open(uploaded_image)
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", 
                    contents=[system_instruction, prompt, img]
                )
            else:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", 
                    contents=[system_instruction, prompt]
                )
            final_response = response.text
        
        # Display and save assistant response
        with st.chat_message("assistant"):
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
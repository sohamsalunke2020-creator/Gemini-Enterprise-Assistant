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

# --- STEP 1: INITIALIZE SESSION STATE (FIXES YOUR ERROR) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Sidebar - Mission Control
st.sidebar.title("🤖 Mission Control")
st.sidebar.markdown("---")

task_mode = st.sidebar.selectbox("Select Task Mode", [
    "Task 1: Knowledge Update",
    "Task 2: Multi-modal (Text & Image)",
    "Task 3: Medical Q&A",
    "Task 4: Research Expert"
])

# Sidebar Reset Button (A Pro Feature)
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
        if uploaded_kb:
            st.sidebar.success("Knowledge loaded!")

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with tab_data:
    st.subheader("📁 Backend Reference")
    if "Task 3" in task_mode:
        try:
            df = pd.read_csv("data/medquad.csv")
            st.info("Displaying first 5 rows of MedQuAD Dataset")
            st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not load dataset: {e}")
    else:
        st.write("This tab displays datasets used for Medical Q&A. Switch to Task 3 to view.")

with tab_logs:
    st.subheader("⚙️ System Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model", value="Gemini 3 Flash")
    with col2:
        st.metric(label="Status", value="Healthy")
    st.code(f"Engine: gemini-3-flash-preview\nSDK: Google GenAI 1.0.0\nSession ID: {id(st.session_state)}")

# --- THE LOGIC ROUTER ---
if prompt := st.chat_input("Ask me anything..."):
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Processing..."):
        # Route 1: Knowledge Update
        if "Task 1" in task_mode:
            if uploaded_kb:
                final_response = handle_knowledge_update(prompt, uploaded_kb)
            else:
                final_response = "⚠️ Please upload a `.txt` file in the sidebar."

        # Route 2: Medical
        elif "Task 3" in task_mode:
            final_response = get_medical_response(prompt)
        
        # Route 3: Research
        elif "Task 4" in task_mode:
            final_response = search_arxiv(prompt)

        # Route 4: Multi-modal (Tasks 2, 5, 6)
        else:
            system_instruction = "Detect language, detect sentiment, and answer deeply."
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
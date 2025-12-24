import os
import pandas as pd
from dotenv import load_dotenv
from google import genai  # Unified SDK

# 1. Load environment variables specifically for this module scope
load_dotenv()

# 2. Initialize the client using the environment variable
# The SDK can also pick up 'GOOGLE_API_KEY' automatically if it's set in the environment
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_medical_response(user_query):
    """
    Retrieves relevant medical context from MedQuAD and generates an expert response.
    """
    # Load dataset
    try:
        df = pd.read_csv("data/medquad.csv")
    except FileNotFoundError:
        return "Error: Medical database (medquad.csv) not found in the 'data' folder."

    # Simple Keyword Retrieval logic
    keywords = user_query.lower().split()
    relevant_rows = df[df['Question'].str.contains('|'.join(keywords), case=False, na=False)].head(3)
    
    context = "\n".join(relevant_rows['Answer'].values) if not relevant_rows.empty else "No direct context found in the MedQuAD dataset."

    # Generate Content using the latest 2025 Gemini 3 model
    # We include a system instruction style prompt for Task 3 requirements
    prompt_text = (
        f"You are a specialized Medical Assistant. Use the following context to answer the user query.\n\n"
        f"Context from MedQuAD: {context}\n\n"
        f"User Query: {user_query}\n\n"
        f"Important: If the context doesn't have the answer, use your medical knowledge but state it. "
        f"Always end with a disclaimer advising the user to consult a professional doctor."
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt_text
    )
    
    return response.text
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def handle_knowledge_update(user_query, uploaded_file):
    """
    Satisfies Task 1: Incorporates information from a user-uploaded document.
    """
    # Read the content of the uploaded TXT file
    new_info = uploaded_file.read().decode("utf-8")
    
    # Custom instruction to prioritize the uploaded document
    prompt_text = (
        f"You are a Knowledge Assistant. A new document has been uploaded with the following content:\n\n"
        f"DOCUMENT CONTENT:\n{new_info}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"Answer the question strictly using the provided document content. If the answer isn't there, say you don't know."
    )
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt_text
    )
    return response.text
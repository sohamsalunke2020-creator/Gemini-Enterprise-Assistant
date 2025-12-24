import os
import arxiv
from dotenv import load_dotenv
from google import genai  # Unified SDK for 2025

# 1. Load environment variables for this specific module
load_dotenv()

# 2. Initialize the client using your environment variable
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def search_arxiv(user_query):
    """
    Fetches real scientific papers from arXiv and summarizes them using Gemini 3.
    """
    # Search for top 3 papers in Computer Science/AI
    search = arxiv.Search(
        query=f"cat:cs.AI AND {user_query}",
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results_text = ""
    # Process the results into a single string for context
    for result in search.results():
        results_text += f"Title: {result.title}\nSummary: {result.summary}\n---\n"
    
    # If no papers were found, provide a fallback message
    if not results_text:
        results_text = "No specific research papers found on arXiv for this topic."

    # 3. Use Gemini 3 to explain these papers simply
    prompt_text = (
        f"You are a Scientific Research Assistant. Using the following summaries from arXiv, "
        f"answer the user's question in simple, easy-to-understand language: {user_query}\n\n"
        f"Context from arXiv:\n{results_text}"
    )
    
    # NEW: Updated model call for 2025 standards
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt_text
    )
    
    return response.text
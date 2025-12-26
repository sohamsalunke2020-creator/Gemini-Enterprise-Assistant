import os
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Create a permanent folder for your database
DB_PATH = "faiss_index"

def handle_knowledge_update(user_query, uploaded_file):
    """
    Satisfies Task 1: Implements a Vector Database (FAISS) mechanism 
    to dynamically expand and query the chatbot's knowledge base.
    """
    # 1. Read and Process the File
    text_content = uploaded_file.read().decode("utf-8")
    
    # 2. Chunking: Break large text into smaller pieces
    # (Requirement: 'Mechanism to update the database')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text_content)
    
    # 3. Setup Embeddings (Converts text to numbers/vectors)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # 4. Create/Update the Vector Database
    if not os.path.exists(DB_PATH):
        # Create new DB if it doesn't exist
        vector_db = FAISS.from_texts(chunks, embeddings)
    else:
        # Load existing and add new information
        vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_db.add_texts(chunks)
    
    # Save the updated database to your HP laptop disk
    vector_db.save_local(DB_PATH)
    
    # 5. Retrieval: Find ONLY the relevant chunks for the user's question
    relevant_docs = vector_db.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 6. Generate Response using the retrieved context
    prompt_text = (
        f"You are a Knowledge Assistant. Use the following retrieved context to answer the question.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"Answer strictly using the context. If not found, say you don't know."
    )
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt_text
    )
    
    return response.text
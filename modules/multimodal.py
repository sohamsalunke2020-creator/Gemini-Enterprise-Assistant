import google.generativeai as genai
import PIL.Image

def handle_multimodal_query(user_text, image_path):
    # Load the Gemini 1.5 model (handles both text and images)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Open the image file
    img = PIL.Image.open(image_path)
    
    # Send both to Gemini
    response = model.generate_content([user_text, img])
    
    return response.text
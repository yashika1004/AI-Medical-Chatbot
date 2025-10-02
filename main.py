import base64
import requests
import io
import os
import logging
from PIL import Image
from dotenv import load_dotenv

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (like GROQ_API_KEY) from .env file
load_dotenv()

# Groq API endpoint for chat completions
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set the supported vision model
# NOTE: The models 'llama-3.2-11b-vision-preview' and 'llama-3.2-90b-vision-preview' are decommissioned.
# We are using the currently available, powerful Groq Vision model.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY is not set in the .env file")

# --- Function to Process Image and Query ---
def process_image(image_path, query):
    """
    Encodes an image, sends it with a query to the Groq Vision API, 
    and returns the model's response.
    """
    try:
        # 1. Read and Encode Image
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            # Base64 encode the image for the API call
            encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # 2. Basic Image Validation (using PIL)
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()  # Verifies image integrity
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}
        
        # 3. Construct API Messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    # The image is embedded directly into the content with a base64 data URI
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        # 4. Make API Request
        try:
            response = requests.post(
                GROQ_API_URL, 
                json={
                    "model": VISION_MODEL, 
                    "messages": messages, 
                    "max_tokens": 1000
                },
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout = 30
            )
            
            # 5. Process Response
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Processed response from {VISION_MODEL} API : {answer[:100]}...")
                return {VISION_MODEL: answer}
            else:
                logger.error(f"Error from Groq API ({VISION_MODEL}) : {response.status_code} - {response.text}")
                return {"error": f"Error from Groq API: {response.status_code}"}

        except requests.exceptions.Timeout:
            logger.error("Request timed out after 30 seconds.")
            return {"error": "Request timed out."}
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during API request: {str(e)}")
            return {"error": f"Network error: {str(e)}"}

    except FileNotFoundError:
        logger.error(f"Image file not found at: {image_path}")
        return {"error": f"Image file not found: {image_path}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred : {str(e)}")
        return {"error": f"An unexpected error occurred : {str(e)}"}

# --- Execution Block ---
if __name__ == "__main__":
    # The image path from your previous session
    image_path = "test2.jpg"
    query = "what are the encoders in this picture? Be precise and descriptive, given that this is a medical image."
    
    logger.info(f"Starting image processing for: {image_path}")
    
    result = process_image(image_path, query)
    
    # Print the final result from the API call
    print("\n--- Groq Vision API Response ---")
    
    # Check if the result is a dictionary and print neatly
    if isinstance(result, dict) and 'error' not in result:
        print(f"Model: {VISION_MODEL}")
        print(f"Answer:\n{result[VISION_MODEL]}")
    else:
        print(f"Operation failed with error: {result.get('error', 'Unknown error')}")
    print("--------------------------------\n")
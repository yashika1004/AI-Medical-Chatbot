import base64
import requests
import io
import os
import logging
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse

# --- Configuration & Setup ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# IMPORTANT: Jinja2Templates requires a directory named 'templates'
templates = Jinja2Templates(directory="templates")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set the currently supported Groq Vision model (FIXED MODEL NAME)
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" 

if not GROQ_API_KEY:
    # This will prevent the server from starting if the API key is missing
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    """Process uploaded image and query using Groq Vision API."""
    
    logger.info(f"Received query: {query}")
    
    try:
        # 1. Read and Encode Image
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
        # Base64 encode the image
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # 2. Image Validation
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # 3. Construct API Messages (using the Base64 data URI)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    # Note: Using image/jpeg as the generic mime type for simplicity
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        # 4. Make API Request
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": VISION_MODEL, # ONLY calling the supported model
                "messages": messages,
                "max_tokens": 1000
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        # 5. Process Groq API Response
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response successfully from {VISION_MODEL}")
            
            # Return a dictionary with the model name as the key (e.g., {"meta-llama/...": "Answer"})
            return JSONResponse(status_code=200, content={VISION_MODEL: answer})
        else:
            # Handle Groq API errors (like 400 or 404)
            error_detail = response.json().get("error", {}).get("message", "Unknown Groq API Error")
            logger.error(f"Groq API Error {response.status_code}: {error_detail}")
            raise HTTPException(status_code=response.status_code, detail=f"Groq API Error: {error_detail}")

    except HTTPException as he:
        # Pass through expected HTTP errors (400, 404, etc.)
        raise he
    except Exception as e:
        logger.error(f"An unexpected server error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Execution Block ---
if __name__ == "__main__":
    import uvicorn
    # Make sure the 'templates' directory exists for Jinja2
    os.makedirs('templates', exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)

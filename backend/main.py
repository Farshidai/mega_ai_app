import os
import requests
import pypdf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json

# --- AI Configuration ---
# In a real application, you would get this from your environment.
# Canvas provides this automatically, so we leave it as an empty string.
API_KEY = ""
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# Initialize the app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Helper Function ---
def get_ai_summary(spec_text: str) -> str:
    """Sends the spec text to the Gemini API and gets a summary."""
    prompt = f"""
    You are an expert construction estimator analyzing a specification document.
    Based ONLY on the text provided below, extract and format the following information in markdown:

    1.  **CSI Code and Scope Confirmation:** Identify the primary CSI sections and briefly describe their scope.
    2.  **Products and Manufacturers:** List any products and their manufacturers specifically named as the "Basis of Design".
    3.  **Key Specifications and Takeoff Data:** Extract specific, quantifiable values like material thickness, dimensions, and performance ratings (e.g., wind load, UL ratings).

    Here is the specification text:
    ---
    {spec_text}
    ---
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        
        # Safely extract the text from the response
        candidate = result.get("candidates", [{}])[0]
        content_part = candidate.get("content", {}).get("parts", [{}])[0]
        summary_text = content_part.get("text", "Error: Could not parse AI response.")
        
        return summary_text
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return "Error: Failed to communicate with the AI model."
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Full response: {result}")
        return "Error: Could not understand the response from the AI model."


# --- Unified Project Submission Endpoint ---
@app.post("/submitproject/")
async def submit_project(
    scope_of_work: str = Form(...),
    spec_file: Optional[UploadFile] = File(None),
    blueprint_file: Optional[UploadFile] = File(None)
):
    print("--- New Project Submission Received ---")
    ai_summary = "No specification file was provided for analysis."

    if spec_file:
        print(f"Processing specification file: {spec_file.filename}")
        try:
            # Read the content of the uploaded PDF file
            pdf_reader = pypdf.PdfReader(spec_file.file)
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()

            if extracted_text.strip():
                # If we got text, send it to the AI for summary
                print("Extracted text, sending to AI for summarization...")
                ai_summary = get_ai_summary(extracted_text)
            else:
                ai_summary = "Could not extract text from the provided PDF."

        except Exception as e:
            print(f"Error processing PDF file: {e}")
            ai_summary = "Error: Failed to read the specification PDF file."
    
    print("--- AI Summary Generated ---")
    print(ai_summary)
    print("----------------------------")

    return {
        "status": "Project submitted successfully!",
        "ai_summary": ai_summary # We now include the summary in the response
    }

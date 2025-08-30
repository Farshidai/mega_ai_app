import os
import requests
import pypdf
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import uuid
from dotenv import load_dotenv
import time # Import the time library for sleep

# --- Securely Load API Key ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# --- File-based Database ---
DB_FILE = "projects.json"
projects_db: Dict[str, Dict] = {}

class PriceUpdateRequest(BaseModel):
    scope_id: str
    item_index: int
    unit_price: float

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(projects_db, f, indent=4)

def load_db():
    global projects_db
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                projects_db = json.load(f)
                print(f"--- Loaded {len(projects_db)} projects from {DB_FILE} ---")
        except json.JSONDecodeError:
            print(f"--- WARNING: {DB_FILE} is corrupted. Starting with an empty database. ---")
            projects_db = {}
    else:
        print("--- No database file found. Starting with an empty database. ---")


# Initialize the app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    load_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Helper Functions with Retry Logic ---
def _call_gemini_api_with_retry(payload: Dict) -> requests.Response:
    """Internal function to call Gemini API with retry logic."""
    headers = {'Content-Type': 'application/json'}
    retries = 3
    backoff_factor = 1  # Start with 1 second

    for attempt in range(retries):
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code >= 500:
                print(f"--- GEMINI API server error (Status: {response.status_code}). Retrying in {backoff_factor}s... ---")
                time.sleep(backoff_factor)
                backoff_factor *= 2
                continue
            return response
        except requests.exceptions.RequestException as e:
            print(f"--- RequestException on attempt {attempt + 1}: {e}. Retrying in {backoff_factor}s... ---")
            if attempt < retries - 1:
                time.sleep(backoff_factor)
                backoff_factor *= 2
            else:
                raise e
    return response


def get_ai_spec_summary(spec_text: str) -> dict:
    prompt = """
    You are an expert construction estimator creating a detailed "Initial Data Extraction Report".
    Analyze the following specification document text and extract the key information into a structured JSON object.

    Follow this exact JSON schema:
    {
      "project_name": "...",
      "spec_section": "...",
      "addendum": "...",
      "material_systems": [
        {
          "system_name": "Example: Metal Wall Panels",
          "attributes": {
            "Basis of Design": "...",
            "Material": { "Type": "...", "Thickness/Gauge": "..." },
            "Finish": { "System": "...", "Composition": "..." },
            "Fabrication": { "Standard": "..." }
          }
        }
      ]
    }
    **CRITICAL INSTRUCTIONS:**
    1.  Focus ONLY on the main material systems. Do NOT create separate systems for minor accessories.
    2.  Extract specific, quantifiable data like thickness, gauge, and dimensions.
    3.  If a value is not found, omit the key.
    Here is the specification text:
    ---
    """ + spec_text
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": { "responseMimeType": "application/json" }
    }
    
    try:
        response = _call_gemini_api_with_retry(payload)
        if response.status_code != 200:
            print(f"--- ERROR FROM GEMINI API (Status Code: {response.status_code}) ---")
            print(response.text)
            return {"error": f"The AI model returned an error (Code: {response.status_code})."}
        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(response_text)
    except Exception as e:
        print(f"--- FATAL ERROR during AI spec summary: {e} ---")
        return {"error": "Failed to get a valid structured summary from the AI model."}


def get_ai_estimate_analysis(estimate_data: List[Dict]) -> dict:
    prompt = f"""
    You are a senior estimator reviewing a priced takeoff.
    Analyze the following list of line items and generate a structured JSON summary.

    Follow this exact JSON schema:
    {{
      "overall_summary": "A brief, one-sentence summary of the total bid.",
      "cost_breakdown": [
          {{ "category": "Example: Flashing & Trim", "total_cost": 15250.75 }},
          {{ "category": "Example: Wall Panels", "total_cost": 85320.00 }}
      ],
      "grand_total": 100570.75,
      "potential_risks": [
          "Example: Verify lead times for custom panels.",
          "Example: Labor for radius coping may be underestimated."
      ]
    }}

    **CRITICAL INSTRUCTIONS:**
    1.  The provided data is already filtered. Analyze ALL of it.
    2.  Categorize the items logically based on their description (e.g., group all flashing, coping, and trim together).
    3.  Calculate the sum of `total` for each category.
    4.  Calculate the `grand_total` by summing the totals of all items.
    5.  Identify 1-3 potential risks or points of interest for an estimator based on the item descriptions and costs.

    Here is the priced takeoff data:
    ---
    {json.dumps(estimate_data, indent=2)}
    ---
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": { "responseMimeType": "application/json" }
    }
    
    try:
        response = _call_gemini_api_with_retry(payload)
        if response.status_code != 200:
            print(f"--- ERROR FROM GEMINI API (Status Code: {response.status_code}) ---")
            print(response.text)
            return {"error": f"The AI model returned an error (Code: {response.status_code})."}
        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(response_text)
    except Exception as e:
        print(f"--- FATAL ERROR during AI estimate analysis: {e} ---")
        return {"error": "Failed to get a valid analysis from the AI model."}


# --- Project Endpoints ---
@app.get("/projects")
async def get_projects():
    return [{"id": pid, "project_name": pdata.get("project_name")} for pid, pdata in projects_db.items()]

@app.get("/project/{project_id}")
async def get_project_details(project_id: str):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects_db[project_id]

@app.post("/project")
async def create_project(project_name: str = Form(...)):
    project_id = str(uuid.uuid4())
    projects_db[project_id] = {"id": project_id, "project_name": project_name, "scopes": []}
    save_db()
    print(f"--- Project Created --- ID: {project_id}, Name: {project_name}")
    return projects_db[project_id]

@app.post("/project/{project_id}/scope")
async def add_scope_to_project(project_id: str, scope_of_work: str = Form(...), spec_file: UploadFile = File(...)):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    ai_summary = {"error": "Could not process spec."}
    try:
        pdf_reader = pypdf.PdfReader(spec_file.file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        if text.strip(): ai_summary = get_ai_spec_summary(text)
    except Exception as e:
        ai_summary = {"error": f"Failed to read PDF: {e}"}
    
    new_scope = {
        "scope_id": str(uuid.uuid4()), "scope_of_work": scope_of_work,
        "spec_filename": spec_file.filename, "ai_summary": ai_summary,
        "estimate_data": [], "estimate_analysis": {}
    }
    projects_db[project_id]["scopes"].append(new_scope)
    save_db()
    return projects_db[project_id]

@app.delete("/project/{project_id}")
async def delete_project(project_id: str):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    del projects_db[project_id]
    save_db()
    return {"status": "Project deleted successfully"}

@app.post("/project/{project_id}/estimate")
async def upload_estimate(project_id: str, scope_id: str = Form(...), estimate_file: UploadFile = File(...)):
    scope, _ = find_scope(project_id, scope_id)
    try:
        df = pd.read_excel(estimate_file.file)
        df.columns = [col.strip() for col in df.columns]
        required = ['Description', 'Quantity', 'Size']
        if not all(col in df.columns for col in required):
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join([c for c in required if c not in df.columns])}")
        
        df = df[required].copy()
        df.fillna({'Description': '', 'Size': '', 'Quantity': 0}, inplace=True)
        df = df[df['Description'] != '']
        df.rename(columns={'Size': 'UOM'}, inplace=True)
        
        scope["estimate_data"] = df.to_dict(orient='records')
        save_db()
        return projects_db[project_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read Excel file: {e}")

@app.post("/project/{project_id}/updateprice")
async def update_price(project_id: str, request: PriceUpdateRequest):
    scope, _ = find_scope(project_id, request.scope_id)
    item_index = request.item_index
    if not (0 <= item_index < len(scope["estimate_data"])):
        raise HTTPException(status_code=400, detail="Item index out of range")

    item = scope["estimate_data"][item_index]
    item["unitPrice"] = request.unit_price
    item["total"] = item.get("Quantity", 0) * request.unit_price
    save_db()
    return item

@app.post("/project/{project_id}/analyze_estimate")
async def analyze_estimate(project_id: str, scope_id: str = Form(...)):
    scope, _ = find_scope(project_id, scope_id)
    
    priced_items = [item for item in scope.get("estimate_data", []) if item.get("Quantity", 0) > 0 and "total" in item]
    if not priced_items:
        raise HTTPException(status_code=400, detail="No priced items with quantities found to analyze.")

    analysis = get_ai_estimate_analysis(priced_items)
    scope["estimate_analysis"] = analysis
    save_db()
    return projects_db[project_id]

def find_scope(project_id: str, scope_id: str) -> (Dict, int):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    for i, scope in enumerate(projects_db[project_id].get("scopes", [])):
        if scope.get("scope_id") == scope_id:
            return scope, i
    raise HTTPException(status_code=404, detail="Scope not found")


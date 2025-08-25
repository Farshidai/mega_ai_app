import os
import requests
import pypdf
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import uuid

# --- AI Configuration ---
API_KEY = "AIzaSyAxld93RME5WttQkVJMw9Rb6fnkAwAuJ_M" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# --- File-based Database ---
DB_FILE = "projects.json"
projects_db: Dict[str, Dict] = {}

# --- Pydantic Models ---
class PriceUpdateRequest(BaseModel):
    scope_id: str
    item_index: int
    unit_price: float

# --- Database Persistence Functions ---
def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(projects_db, f, indent=4)

def load_db():
    global projects_db
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            projects_db = json.load(f)
            print(f"--- Loaded {len(projects_db)} projects from {DB_FILE} ---")

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

# --- AI Helper Function ---
def get_ai_summary(spec_text: str) -> dict:
    prompt = f"""
    You are an expert construction estimator creating a detailed "Initial Data Extraction Report".
    Analyze the following specification document text and extract the key information into a structured JSON object.

    Follow this exact JSON schema:
    {{
      "project_name": "...",
      "spec_section": "...",
      "addendum": "...",
      "material_systems": [
        {{
          "system_name": "Example: Metal Soffit Panels",
          "attributes": {{
            "General Type": "...",
            "Basis of Design": "...",
            "Material": {{
                "Panels and Accessories": "...",
                "Fasteners (Exterior)": "..."
            }},
            "Finish": {{
                "System": "...",
                "Composition": "..."
            }},
            "Fabrication": {{
                "Standard": "...",
                "Seams (Aluminum)": "..."
            }}
          }}
        }}
      ]
    }}
    **CRITICAL INSTRUCTIONS:**
    1.  Focus ONLY on the main material systems. Do NOT create separate systems for minor accessories.
    2.  Populate the "attributes" object. If an attribute has sub-points (like "Material" or "Finish"), create a nested object for them.
    3.  If a value is not found, omit the key. If an entire nested object is empty, omit it.
    Here is the specification text:
    ---
    {spec_text}
    ---
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": { "responseMimeType": "application/json" }
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        if response.status_code != 200: 
            print(f"--- ERROR FROM GEMINI API (Status Code: {response.status_code}) ---")
            print(response.text)
            return {"error": f"The AI model returned an error (Code: {response.status_code})."}
        response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return json.loads(response_text)
    except Exception as e:
        return {"error": "Failed to get a valid structured summary from the AI model."}

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
        if text.strip(): ai_summary = get_ai_summary(text)
    except Exception as e:
        ai_summary = {"error": f"Failed to read PDF: {e}"}
    
    new_scope = {
        "scope_id": str(uuid.uuid4()), "scope_of_work": scope_of_work,
        "spec_filename": spec_file.filename, "ai_summary": ai_summary, "estimate_data": []
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
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    scope_index = -1
    for i, scope in enumerate(projects_db[project_id]["scopes"]):
        if scope["scope_id"] == scope_id:
            scope_index = i
            break
    if scope_index == -1:
        raise HTTPException(status_code=404, detail="Scope not found")

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
        
        projects_db[project_id]["scopes"][scope_index]["estimate_data"] = df.to_dict(orient='records')
        save_db()
        return projects_db[project_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read Excel file: {e}")

@app.post("/project/{project_id}/updateprice")
async def update_price(project_id: str, request: PriceUpdateRequest):
    if project_id not in projects_db:
        raise HTTPException(status_code=404, detail="Project not found")
    
    scope_index = -1
    for i, scope in enumerate(projects_db[project_id]["scopes"]):
        if scope["scope_id"] == request.scope_id:
            scope_index = i
            break
    if scope_index == -1:
        raise HTTPException(status_code=404, detail="Scope not found")

    item_index = request.item_index
    if not (0 <= item_index < len(projects_db[project_id]["scopes"][scope_index]["estimate_data"])):
        raise HTTPException(status_code=400, detail="Item index out of range")

    item = projects_db[project_id]["scopes"][scope_index]["estimate_data"][item_index]
    item["unitPrice"] = request.unit_price
    item["total"] = item.get("Quantity", 0) * request.unit_price
    save_db()
    return item
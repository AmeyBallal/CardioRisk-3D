import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# ==========================================
# SETUP PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Map to the unzipped kaggle folder you just downloaded
MODELS_DIR = os.path.join(BASE_DIR, '..', 'cardiorisk_outputs', 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, '..', 'cardiorisk_outputs', 'artifacts')

app = FastAPI(title="CardioRisk ML API", description="API serving cardiovascular risk predictions for 3D visualization.")

# Enable CORS so our React frontend can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models
pipelines = {}
feature_schema = {}

@app.on_event("startup")
def load_assets():
    global pipelines, feature_schema
    print("Loading ML pipelines into memory...")
    try:
        pipelines['CAD'] = joblib.load(os.path.join(MODELS_DIR, 'CAD_pipeline.joblib'))
        pipelines['LAD'] = joblib.load(os.path.join(MODELS_DIR, 'LAD_pipeline.joblib'))
        pipelines['LCX'] = joblib.load(os.path.join(MODELS_DIR, 'LCX_pipeline.joblib'))
        pipelines['RCA'] = joblib.load(os.path.join(MODELS_DIR, 'RCA_pipeline.joblib'))
        
        with open(os.path.join(ARTIFACTS_DIR, 'raw_feature_schema.json'), 'r') as f:
            feature_schema = json.load(f)
            
        print("✅ Models and schemas loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        print("Please ensure you unzipped cardiorisk_outputs correctly.")

class PatientData(BaseModel):
    # A dictionary mapping feature names (e.g., 'Age', 'Sex') to their values
    features: Dict[str, Any]

@app.post("/predict")
def predict_risk(data: PatientData):
    """
    Accepts raw patient data, passes it through the preprocessor pipelines, 
    and returns risk probabilities for CAD, LAD, LCX, and RCA.
    """
    try:
        # 1. Convert incoming JSON dictionary into a Pandas DataFrame (1 row)
        df = pd.DataFrame([data.features])
        
        # 2. Safety Check: Ensure all 55 features expected by the model exist.
        for col, details in feature_schema.items():
            if col not in df.columns or pd.isna(df[col]).all():
                # If it's a categorical/binary field, assume '0.0' or normal/baseline
                if "allowed_values" in details:
                    # For VHD/BBB string categories, use the most common normal ('N' or 0)
                    if "N" in details["allowed_values"]:
                        df[col] = "N"
                    else:
                        df[col] = 0.0
                else:
                    # If it's numeric (Age, BP), leave as None so SimpleImputer uses the Median!
                    df[col] = None
                    
        # Enforce exact column order
        df = df[list(feature_schema.keys())]
        
        # 3. Generate predictions
        results = {}
        for target, pipeline in pipelines.items():
            prob = pipeline.predict_proba(df)[0][1]
            results[target] = round(float(prob) * 100, 2)
            
        return {
            "status": "success",
            "risk_scores": results,
            "message": "Predictions generated successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/schema")
def get_schema():
    """
    Returns the feature schema. 
    The React Frontend will use this to automatically build the clinical input form!
    """
    return feature_schema

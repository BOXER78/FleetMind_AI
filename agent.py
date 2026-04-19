import os
import joblib
import pandas as pd
import json
import logging
from typing import Dict, Any, Union
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- STEP 1: VALIDATION LAYER ---

def validate_input(data: dict) -> dict:
    """
    Validates input data strictly to prevent unrealistic values.
    Returns error dict if validation fails, else empty dict.
    """
    try:
        mileage = float(data.get("Mileage", 0))
        vehicle_age = float(data.get("Vehicle_Age", 0))
        
        # Check no negative values for common numerical inputs
        for key in ["Mileage", "Vehicle_Age", "Reported_Issues", "Engine_Size", "Odometer_Reading"]:
            val = data.get(key)
            if val is not None and float(val) < 0:
                logger.error(f"Validation failed: Negative value for {key}")
                return {
                    "error": "Invalid input data",
                    "message": "Please check vehicle parameters (Negative values not allowed)"
                }
                
        if vehicle_age > 30:
            logger.error("Validation failed: Vehicle age > 30")
            return {
                "error": "Invalid input data",
                "message": "Please check vehicle parameters (Vehicle age > 30)"
            }
            
        if mileage <= 0:
            logger.error("Validation failed: Mileage <= 0")
            return {
                "error": "Invalid input data",
                "message": "Please check vehicle parameters (Mileage must be > 0)"
            }
            
        return {}
    except ValueError:
        logger.error("Validation failed: Non-numeric value for numeric field")
        return {
            "error": "Invalid input data",
            "message": "Please check vehicle parameters (Invalid types)"
        }

# --- STEP 2: REUSE EXISTING SYSTEM ---

def load_ml_model(model_path: str = "fleetmind_model.pkl") -> Any:
    """
    Loads the trained ML pipeline without modifying it.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def get_ml_prediction(model: Any, input_data: Union[pd.DataFrame, Dict[str, Any]]) -> tuple[int, float]:
    """
    Performs prediction using the model and extracts prediction (0 or 1) and probability score.
    """
    # Ensure input is a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
        
    try:
        prediction_val = int(model.predict(input_data)[0])
        # Safely get probability if the model supports it
        if hasattr(model, "predict_proba"):
            probability_val = float(model.predict_proba(input_data)[0][1])
        else:
            probability_val = float(prediction_val) # Fallback if predict_proba is missing
            
        logger.info(f"Model prediction: {prediction_val}, Probability: {probability_val:.4f}")
        return prediction_val, probability_val
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

# --- STEP 2: PREPARE AGENT INPUT ---

def prepare_agent_input(input_dict: dict, prediction_val: int, probability_val: float) -> dict:
    """
    Constructs a structured dictionary taking only available data and model outcomes.
    Type validation is enforced.
    """
    # Safely extract values with fallbacks/type casting to avoid fabrication/errors
    vehicle_data = {
        "mileage": int(input_dict.get("Mileage", 0)),
        "vehicle_age": int(input_dict.get("Vehicle_Age", 0)),
        "reported_issues": int(input_dict.get("Reported_Issues", 0)),
        "days_since_last_service": int(input_dict.get("Days_Since_Last_Service", 0)),
        "prediction": int(prediction_val),
        "probability": float(probability_val)
    }
    logger.info(f"Prepared agent input: {vehicle_data}")
    return vehicle_data

# --- STEP 3 & 4: LLM INTEGRATION & AGENT PROMPT EXECUTION ---

def generate_maintenance_insights(vehicle_data: dict, model_name: str = "openai/gpt-oss-120b") -> dict:
    """
    Uses the Groq API interface to generate structured JSON output.
    """
    try:
        # Initialize the Groq client. It will automatically look for GROQ_API_KEY in your environment.
        # If you still want to route it to local Ollama, add: base_url="http://localhost:11434/v1" and api_key="ollama"
        client = Groq()
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise

    prompt = f"""
You are a senior AI engineer reviewing and correcting an AI agent used in a Fleet Maintenance Prediction System.

SYSTEM CONTEXT:
* ML model already outputs:
  * prediction (0 or 1)
  * probability (float)
* LLM is used ONLY to generate explanations and recommendations
* The system must be deterministic and production-safe

CRITICAL RULES:
1. CONFIDENCE FIELD (MANDATORY CHANGE)
* DO NOT generate confidence using natural language
* DO NOT rephrase or interpret probability
* Use EXACT probability value from input
* Format as percentage string (e.g., "{vehicle_data['probability'] * 100:.1f}%")

2. STRICT OUTPUT FORMAT
* Output MUST be valid JSON
* No extra keys
* No explanations outside JSON
* Base ALL reasoning ONLY on provided input
* Do NOT hallucinate
* Keep recommendations realistic
* Max 4 actions
* Keep output concise and structured
* If input is insufficient -> return "Insufficient data"
* No markdown

ALLOWED OUTPUT STRUCTURE ONLY:
{{
  "health_summary": "...",
  "risk_level": "LOW | MEDIUM | HIGH",
  "actions": ["...", "..."],
  "timeline": "...",
  "confidence": "{vehicle_data['probability'] * 100:.1f}%",
  "disclaimer": "..."
}}

INPUT:
{json.dumps(vehicle_data, indent=2)}
"""

    try:
        # ChatCompletion via Groq client interface
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an API that outputs ONLY raw JSON without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            # KEEP TEMPERATURE LOW (0.1 - 0.3) for consistent JSON output
            temperature=0.1, 
            # If supported by the model, ensure deterministic response format
            response_format={"type": "json_object"} 
        )
        
        raw_output = response.choices[0].message.content.strip()
        logger.info("Successfully generated insights from LLM.")
        return raw_output
        
    except Exception as e:
        logger.error(f"Error during LLM generation using {model_name}: {e}")
        # Return fallback json string if generation fails to avoid crashing
        fallback = {
            "health_summary": "LLM generation failed.",
            "risk_level": "HIGH" if vehicle_data['prediction'] == 1 else "LOW",
            "actions": ["Perform manual inspection"],
            "timeline": "ASAP",
            "confidence": f"{vehicle_data['probability'] * 100:.1f}%",
            "disclaimer": "Error generating advanced insights. Please review manually."
        }
        return json.dumps(fallback)
# --- STEP 5: OUTPUT HANDLING ---

def parse_agent_response(raw_output: str) -> dict:
    """
    Parses response safely using json.loads(), handling parsing errors.
    """
    try:
        # Strip any accidental markdown formatting the LLM might have returned
        clean_output = raw_output.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        parsed_json = json.loads(clean_output)
        
        # Basic validation of required keys
        required_keys = ["health_summary", "risk_level", "actions", "timeline", "confidence", "disclaimer"]
        for key in required_keys:
            if key not in parsed_json:
                parsed_json[key] = "N/A"
                
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing Error: {e}\nRaw Output was:\n{raw_output}")
        # Safe structured fallback report
        return {
            "health_summary": "Parsing Error: Could not determine health summary.",
            "risk_level": "UNKNOWN",
            "actions": ["Perform manual inspection due to AI parsing failure"],
            "timeline": "Immediate",
            "confidence": "Low",
            "disclaimer": "Generated response failed strict JSON formatting."
        }

# --- STEP 6: POST-PROCESSING ENFORCEMENT ---

def enforce_output_rules(report: dict, vehicle_data: dict) -> dict:
    """
    Hardens the LLM output to match deterministic constraints and structure rules.
    """
    # 1. Remove Invalid Fields
    allowed_keys = {"health_summary", "risk_level", "actions", "timeline", "confidence", "disclaimer"}
    final_report = {k: v for k, v in report.items() if k in allowed_keys}
    
    # Ensure all required keys exist safely
    for k in allowed_keys:
        if k not in final_report:
            final_report[k] = "N/A"
            
    # 2. Timeline Inconsistency Mapping
    raw_timeline = str(final_report.get("timeline", "")).lower()
    if any(k in raw_timeline for k in ["immedi", "asap", "urgent"]) or (
        "day" in raw_timeline and any(d in raw_timeline for d in ["0", "1", "2", "3"])):
        final_report["timeline"] = "Immediate (0–3 days)"
    elif any(k in raw_timeline for k in ["soon", "week", "7"]):
        final_report["timeline"] = "Soon (within 7 days)"
    else:
        final_report["timeline"] = "Monitor (within 30 days)"
        
    # 3. Deterministic Confidence
    # Format EXACTLY from ML Probability, discarding whatever the LLM said
    final_report["confidence"] = f"{float(vehicle_data.get('probability', 0)) * 100:.1f}%"
    
    # 4. Limit Actions to Max 4
    actions = final_report.get("actions", [])
    if isinstance(actions, list):
        final_report["actions"] = actions[:4]
    else:
        final_report["actions"] = [str(actions)]
        
    return final_report

# --- MAIN AGENT PIPELINE FOR STREAMLIT PLUG-IN ---

def run_agent_pipeline(input_data_dict: dict, model_path: str = "fleetmind_model.pkl") -> dict:
    """
    Orchestrates the entire agent layer from a Streamlit input dictionary.
    Returns the final structured json response as a python dictionary.
    """
    # 0. Validate Input
    validation_err = validate_input(input_data_dict)
    if validation_err:
        return validation_err
        
    try:
        # 1. Load Model
        ml_model = load_ml_model(model_path)
        
        # 2. Perform Prediction
        prediction_val, probability_val = get_ml_prediction(ml_model, input_data_dict)
        
        # 3. Prepare Agent Input
        vehicle_data = prepare_agent_input(input_data_dict, prediction_val, probability_val)
        
        # 4. Generate Insights via LLM
        raw_llm_response = generate_maintenance_insights(vehicle_data, model_name="openai/gpt-oss-120b")
        
        # 5. Parse Output
        parsed_report = parse_agent_response(raw_llm_response)
        
        # 6. Enforce Post-Processing Rules
        final_report = enforce_output_rules(parsed_report, vehicle_data)
        
        # Strict adherence to return purely the agent report without injected metadata
        return final_report
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {
            "health_summary": f"System Failure: {str(e)}",
            "risk_level": "UNKNOWN",
            "actions": ["Contact IT support"],
            "timeline": "N/A",
            "confidence": "N/A",
            "disclaimer": "The agent pipeline failed to execute completely."
        }

if __name__ == "__main__":
    # Example test usage
    sample_input = {
        "Vehicle_Model": "Truck",
        "Fuel_Type": "Diesel",
        "Transmission_Type": "Manual",
        "Owner_Type": "Second",
        "Mileage": 85000,
        "Reported_Issues": 3,
        "Vehicle_Age": 7,
        "Engine_Size": 4500,
        "Odometer_Reading": 91000,
        "Insurance_Premium": 700,
        "Accident_History": 1,
        "Fuel_Efficiency": 8.5,
        "Days_Since_Last_Service": 150,
        "Warranty_Days_Left": -100 # Expired
    }
    
    print("Running sample inference...")
    report = run_agent_pipeline(sample_input, "fleetmind_model.pkl")
    print("\n--- FINAL AGENT REPORT ---")
    print(json.dumps(report, indent=2))

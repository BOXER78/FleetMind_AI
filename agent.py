import os
import joblib
import pandas as pd
import json
import logging
from typing import Dict, Any, Union
from groq import Groq
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

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
    Strictly inference-only: Expects the model to already exist.
    """
    if not os.path.exists(model_path):
        logger.error("Model file not found. Please train the model separately.")
        raise FileNotFoundError("Model file not found. Please train the model separately.")
        
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ValueError("Model file missing or incompatible")

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

# --- STEP 3: RAG KNOWLEDGE BASE & RETRIEVER ---

def create_knowledge_base():
    """
    Step 1: Create a simple domain-specific knowledge base.
    """
    return [
        "Brake pads: Replace every 30,000–50,000 km. Heavy towing reduces lifespan by 20%.",
        "Engine oil: Replace every 10,000 km or 12 months, whichever comes first.",
        "Battery: Replace every 3–5 years. Cold weather reduces battery performance.",
        "Suspension: Inspect every 20,000 km for leaks or worn bushings.",
        "Tires: Rotate every 8,000–10,000 km to ensure even wear.",
        "Brake fluid: Flush and replace every 2 years.",
        "Air filter: Replace every 15,000–30,000 km.",
        "Transmission fluid: Check every 50,000 km for discoloration."
    ]

def build_vector_store():
    """
    Step 2: Vector store setup using Chroma and HuggingFaceEmbeddings.
    """
    rules = create_knowledge_base()
    documents = [Document(page_content=rule) for rule in rules]
    # Free embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # In-memory vector store for simplicity
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store

def get_retriever():
    """
    Step 3: Create a retriever for similarity search.
    """
    vector_store = build_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 3})

# --- STEP 4 & 5: LLM INTEGRATION & AGENT PROMPT EXECUTION ---

def generate_maintenance_insights(vehicle_data: dict, model_name: str = "openai/gpt-oss-120b") -> dict:
    """
    Uses the Groq API interface to generate structured JSON output.
    """
    try:
        client = Groq()
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise

    prompt = f"""
You are a senior AI engineer reviewing and correcting an AI agent used in a Fleet Maintenance Prediction System.

SYSTEM CONTEXT:
* ML model already outputs prediction and probability.
* RAG: Use the provided "maintenance rules" (RETRIEVED CONTEXT) to ground your recommendations.

CRITICAL RULES:
1. GROUNDING (MANDATORY)
* MUST use retrieved rules when giving recommendations. 
* DO NOT invent rules or hallucinate intervals.
* If no relevant rules found in context -> say "Limited knowledge available" in health_summary.
* Maintain strict JSON format.

2. CONFIDENCE FIELD
* Use EXACT probability value from input as percentage string.

3. STRICT OUTPUT FORMAT
* Max 4 actions.
* Limit recommendations to what is justified by context and vehicle data.
* Keep temperature low (0.1–0.3).

ALLOWED OUTPUT STRUCTURE:
{{
  "health_summary": "...",
  "risk_level": "LOW | MEDIUM | HIGH",
  "actions": ["...", "..."],
  "timeline": "...",
  "confidence": "{vehicle_data['probability'] * 100:.1f}%",
  "sources": "Knowledge base",
  "disclaimer": "..."
}}

RETRIEVED CONTEXT (MAINTENANCE RULES):
{vehicle_data.get('context', 'Limited knowledge available')}

VEHICLE INPUT DATA:
{json.dumps({k:v for k,v in vehicle_data.items() if k != 'context'}, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an API that outputs ONLY raw JSON without markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            temperature=0.1, 
            response_format={"type": "json_object"} 
        )
        
        raw_output = response.choices[0].message.content.strip()
        logger.info("Successfully generated insights from LLM.")
        return raw_output
        
    except Exception as e:
        logger.error(f"Error during LLM generation using {model_name}: {e}")
        fallback = {
            "health_summary": "LLM generation failed.",
            "risk_level": "HIGH" if vehicle_data['prediction'] == 1 else "LOW",
            "actions": ["Perform manual inspection"],
            "timeline": "ASAP",
            "confidence": f"{vehicle_data['probability'] * 100:.1f}%",
            "sources": "Knowledge base",
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
        required_keys = ["health_summary", "risk_level", "actions", "timeline", "confidence", "sources", "disclaimer"]
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
    allowed_keys = {"health_summary", "risk_level", "actions", "timeline", "confidence", "sources", "disclaimer"}
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

# --- LANGGRAPH IMPLEMENTATION ---

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    input_data: dict
    prediction: Optional[int]
    probability: Optional[float]
    vehicle_data: Optional[dict]
    context: Optional[str]
    report: Optional[dict]
    error: Optional[str]
    model_path: Optional[str]

def validate_node(state: AgentState):
    validation_err = validate_input(state["input_data"])
    if validation_err:
        return {"error": validation_err.get("message", "Validation failed")}
    return {}

def predict_node(state: AgentState):
    ml_model = load_ml_model(state.get("model_path", "fleetmind_model.pkl"))
    pred, prob = get_ml_prediction(ml_model, state["input_data"])
    return {"prediction": pred, "probability": prob}

def prepare_node(state: AgentState):
    vehicle_data = prepare_agent_input(state["input_data"], state["prediction"], state["probability"])
    return {"vehicle_data": vehicle_data}

def retrieve_node(state: AgentState):
    """
    Step 4: Retrieve node to fetch relevant maintenance rules.
    """
    logger.info("Retrieving maintenance rules...")
    v_data = state["vehicle_data"]
    # Create query string from vehicle data
    query = f"vehicle with {v_data['mileage']} km, {v_data['vehicle_age']} years old, {v_data['reported_issues']} reported issues."
    
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        context = "\n".join([f"- {doc.page_content}" for doc in docs])
        logger.info(f"Retrieved Context:\n{context}")
        return {"context": context}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"context": "Limited knowledge available"}

def generate_node(state: AgentState):
    """
    Step 5: Modified generate node to include context.
    """
    # Combine vehicle data and context
    input_to_llm = state["vehicle_data"].copy()
    input_to_llm["context"] = state.get("context", "Limited knowledge available")
    
    raw_llm_response = generate_maintenance_insights(input_to_llm, model_name="openai/gpt-oss-120b")
    parsed_report = parse_agent_response(raw_llm_response)
    return {"report": parsed_report}

def enforce_node(state: AgentState):
    final_report = enforce_output_rules(state["report"], state["vehicle_data"])
    return {"report": final_report}

def route_after_validation(state: AgentState):
    if state.get("error"):
        return "end"
    return "predict_node"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("validate_node", validate_node)
    workflow.add_node("predict_node", predict_node)
    workflow.add_node("prepare_node", prepare_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("enforce_node", enforce_node)
    
    workflow.set_entry_point("validate_node")
    
    workflow.add_conditional_edges(
        "validate_node",
        route_after_validation,
        {
            "end": END,
            "predict_node": "predict_node"
        }
    )
    
    workflow.add_edge("predict_node", "prepare_node")
    workflow.add_edge("prepare_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", "enforce_node")
    workflow.add_edge("enforce_node", END)
    
    return workflow.compile()

# Compile the graph instance
agent_graph = build_graph()

# --- MAIN AGENT PIPELINE FOR STREAMLIT PLUG-IN ---

def run_agent_pipeline(input_data_dict: dict, model_path: str = "fleetmind_model.pkl") -> dict:
    """
    Orchestrates the entire agent layer from a Streamlit input dictionary using LangGraph.
    Returns the final structured json response as a python dictionary.
    """
    try:
        initial_state = {
            "input_data": input_data_dict,
            "model_path": model_path
        }
        
        # Execute the graph
        result_state = agent_graph.invoke(initial_state)
        
        # Handle early termination (Validation Failure)
        if result_state.get("error"):
            return {
                "error": "Invalid input data",
                "message": result_state["error"]
            }
            
        # Success path
        if result_state.get("report"):
            return result_state["report"]
            
        raise ValueError("Pipeline completed but no report was generated.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        error_msg = str(e)
        if isinstance(e, FileNotFoundError) or "missing or incompatible" in error_msg:
            return {
                "error": "Model loading failed",
                "message": "Model file missing or incompatible"
            }
            
        return {
            "health_summary": f"System Failure: {error_msg}",
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

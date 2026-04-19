# FleetMind AI – Predictive Vehicle Maintenance System

## Overview

FleetMind AI is an end-to-end intelligent system that combines machine learning and large language models to predict vehicle maintenance risk and generate structured, actionable maintenance reports.

The system takes vehicle data as input, predicts the likelihood of maintenance requirements using a trained ML model, and enhances the output with an AI agent that provides human-readable insights and recommendations.

---

## Key Features

* Predictive maintenance using a trained machine learning model
* AI-powered insights using a large language model via Groq
* Deterministic and structured JSON output
* Input validation to ensure data integrity
* Post-processing layer to enforce consistency and prevent hallucinations
* Modular and scalable backend architecture
* Interactive user interface built with Streamlit
* LangGraph-based workflow for structured agent execution (in progress)

---

## System Architecture

```
User Input
   ↓
Validation Layer
   ↓
ML Model (Prediction)
   ↓
AI Agent (LLM via Groq)
   ↓
Post-Processing Enforcement
   ↓
Structured Report
   ↓
Streamlit UI
```

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn (ML model)
* Groq API
* LangChain
* LangGraph

---

## Project Structure

```
FleetMind_AI/
│
├── agent.py                # Core AI agent pipeline
├── app.py                  # Streamlit UI
├── fleetmind_model.pkl     # Trained ML model
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## How It Works

### 1. Input Processing

User inputs vehicle data through the Streamlit interface or CSV.

### 2. Validation Layer

Input data is validated to ensure:

* No negative values
* Reasonable ranges (e.g., vehicle age ≤ 30)

Invalid inputs return a structured error response.

### 3. ML Prediction

The trained model generates:

* Prediction (maintenance required or not)
* Probability score

### 4. Agent Input Preparation

Relevant fields are transformed into a structured `vehicle_data` object for the AI agent.

### 5. AI Agent (LLM)

The LLM:

* Summarizes vehicle condition
* Assesses risk
* Recommends actions
* Provides timeline

### 6. Post-Processing Enforcement

Ensures:

* Strict JSON format
* Fixed output schema
* Deterministic confidence from ML
* Valid timeline categories
* Maximum of 4 actions

---

## Output Format

```json
{
  "health_summary": "...",
  "risk_level": "LOW | MEDIUM | HIGH",
  "actions": ["...", "..."],
  "timeline": "...",
  "confidence": "XX.X%",
  "disclaimer": "..."
}
```

---

## Setup Instructions

### 1. Clone Repository

```
git clone <your-repo-url>
cd FleetMind_AI
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Application

```
streamlit run app.py
```

---

## Running the Agent Independently

You can test the agent pipeline directly:

```
python agent.py
```

This runs a sample inference for debugging and validation.

---

## LangGraph Integration

The system is being extended using LangGraph to model the pipeline as a state-driven workflow:

* Validate → Predict → Prepare → Generate → Enforce

This improves modularity, traceability, and control over execution.

---

## Future Enhancements

* Retrieval-Augmented Generation (RAG) for maintenance guidelines
* PDF report export
* Cloud deployment (Hugging Face Spaces / Render)
* Improved UI/UX and analytics dashboard

---

## Design Principles

* Deterministic outputs over generative variability
* Strict schema enforcement
* Separation of concerns (ML, Agent, UI)
* Safety and validation-first approach

---

## License

This project is intended for educational and demonstration purposes.

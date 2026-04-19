import streamlit as st
import pandas as pd
import joblib
import numpy as np
import base64
from agent import run_agent_pipeline
from pdf_generator import generate_pdf

# --- CONFIGURATION ---
st.set_page_config(
    page_title="FleetMind AI • Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ASSETS & HELPERS ---
LOGO_PATH = "/Users/kanishkranjan/.gemini/antigravity/brain/8d000d7c-c623-41d1-9826-56a61f59e586/fleetmind_liquid_chrome_logo_1771863946357.png"

def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

logo_base64 = get_base64(LOGO_PATH)

# --- EXPERIMENTAL: GRAINY TEXTURE ---
# Using a data URI for a subtle noise texture or SVG filter
GRAIN_FILTER = """
<svg viewBox="0 0 200 200" xmlns='http://www.w3.org/2000/svg'>
  <filter id='noiseFilter'>
    <feTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/>
  </filter>
  <rect width='100%' height='100%' filter='url(#noiseFilter)' opacity='0.05'/>
</svg>
"""
grain_base64 = base64.b64encode(GRAIN_FILTER.encode()).decode()

# --- CSS: BRUTALIST AGENCY STYLE ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&family=Inter:wght@400;700;900&display=swap');

    :root {{
        --bg-color: #f0f0f0;
        --border-color: #000000;
        --accent-color: #000000;
        --primary-text: #000000;
        --secondary-text: #333333;
    }}

    /* Reset & Background */
    .stApp {{
        background-color: var(--bg-color);
        background-image: url("data:image/svg+xml;base64,{grain_base64}");
    }}

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, p, div, span, label {{
        font-family: 'Inter', sans-serif !important;
        color: var(--primary-text) !important;
    }}

    h1, h2, h3, .stHeader, [data-testid="stHeader"] {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: -0.05em !important;
        color: black !important;
    }}

    /* Brutalist Grid Container */
    .brutalist-container {{
        border: 2px solid var(--border-color);
        padding: 0;
        margin: 2rem 0;
        background: white;
        box-shadow: 10px 10px 0px 0px rgba(0,0,0,1);
    }}

    .grid-row {{
        display: flex;
        border-bottom: 2px solid var(--border-color);
    }}

    .grid-row:last-child {{
        border-bottom: none;
    }}

    .grid-cell {{
        padding: 1.5rem;
        border-right: 2px solid var(--border-color);
        flex: 1;
    }}

    .grid-cell:last-child {{
        border-right: none;
    }}

    /* Header Styling */
    .agency-header {{
        padding: 6rem 2rem;
        border-bottom: 3px solid var(--border-color);
        background: #000000;
        text-align: center;
        position: relative;
        overflow: hidden;
    }}

    .agency-header::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            repeating-linear-gradient(
                45deg,
                transparent,
                transparent 20px,
                rgba(255,255,255,0.03) 20px,
                rgba(255,255,255,0.03) 22px
            );
        pointer-events: none;
    }}

    .agency-header::after {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: radial-gradient(circle, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 24px 24px;
        pointer-events: none;
    }}

    .hero-subtitle {{
        font-weight: 900;
        font-size: 1.5rem;
        letter-spacing: 0.2rem;
        margin-top: -1rem;
        color: white !important;
        background: transparent;
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border: 2px solid white;
        position: relative;
        z-index: 1;
    }}

    .hero-subtitle:hover {{
        background: white;
        color: black !important;
    }}

    .logo-img {{
        max-width: 900px;
        width: 100%;
        filter: contrast(1.2) brightness(1.1);
        margin-bottom: 2rem;
    }}

    .marquee {{
        background: var(--accent-color);
        color: white !important;
        padding: 0.5rem 0;
        font-size: 0.8rem;
        font-weight: 900;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        border-top: 2px solid black;
        border-bottom: 2px solid black;
    }}

    .marquee span, .marquee div {{
        color: white !important;
    }}

    /* Form Overrides */
    .stNumberInput, .stSelectbox {{
        margin-bottom: 1rem;
    }}

    label [data-testid="stWidgetLabel"] p {{
        font-weight: 900 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        color: black !important;
    }}

    /* Force sharp corners and readability on ALL inputs */
    input, select, div[data-baseweb="select"], div[data-baseweb="input"] {{
        border-radius: 0px !important;
        border: 2px solid black !important;
        font-weight: 700 !important;
        background-color: white !important;
        color: black !important;
    }}
    
    /* Target internal elements for rounded corners */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        border-radius: 0px !important;
        background-color: white !important;
        color: black !important;
    }}

    /* Selectbox selected value text */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div[class*="value"],
    div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="select"] * {{
        color: black !important;
        background-color: white !important;
    }}

    /* Selectbox dropdown arrow */
    div[data-baseweb="select"] svg {{
        fill: black !important;
        color: black !important;
    }}

    /* Dropdown menu / popover - aggressive dark mode override */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > div,
    div[data-baseweb="list"],
    div[data-baseweb="list"] > div,
    ul[role="listbox"],
    ul[role="listbox"] li,
    ul[role="listbox"] li *,
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] li *,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {{
        background-color: white !important;
        color: black !important;
        border-radius: 0px !important;
    }}

    /* Dropdown option hover & selected */
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"],
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li:focus,
    li[data-highlighted="true"],
    li[aria-selected="true"] {{
        background-color: #e0e0e0 !important;
        color: black !important;
    }}

    /* Force all option text */
    [data-baseweb="menu"] li span,
    [data-baseweb="menu"] li div,
    [data-baseweb="menu"] li p {{
        color: black !important;
    }}

    /* Number input internal */
    .stNumberInput input,
    .stNumberInput div[data-baseweb="input"] input {{
        background-color: white !important;
        color: black !important;
    }}

    /* Widget labels in dark mode */
    .stSelectbox label,
    .stNumberInput label,
    .stSelectbox label p,
    .stNumberInput label p {{
        color: black !important;
    }}

    /* Button Styling */
    .stButton>button,
    .stFormSubmitButton>button {{
        width: 100%;
        border-radius: 0 !important;
        background: white !important;
        background-color: white !important;
        color: black !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        padding: 1rem !important;
        border: 3px solid black !important;
        transition: all 0.1s ease !important;
        font-size: 1.2rem !important;
        margin-top: 1rem;
    }}

    .stButton>button *,
    .stFormSubmitButton>button *,
    .stButton>button p,
    .stFormSubmitButton>button p,
    .stButton>button span,
    .stFormSubmitButton>button span {{
        color: black !important;
    }}

    .stButton>button:hover,
    .stFormSubmitButton>button:hover {{
        background: black !important;
        background-color: black !important;
        color: white !important;
        transform: translate(-4px, -4px);
        box-shadow: 6px 6px 0px 0px rgba(0,0,0,1);
    }}

    .stButton>button:hover *,
    .stFormSubmitButton>button:hover *,
    .stButton>button:hover p,
    .stFormSubmitButton>button:hover p {{
        color: white !important;
    }}

    .stFormSubmitButton>button:focus,
    .stFormSubmitButton>button:active {{
        background: white !important;
        color: black !important;
        outline: none !important;
    }}

    /* Results section */
    .result-card {{
        padding: 2rem;
        text-align: center;
        border: 4px solid black;
        margin-top: 2rem;
    }}
    
    .status-healthy {{
        background: #00FF41;
        color: black;
    }}
    
    .status-risk {{
        background: #FF3131;
        color: white;
    }}

    .big-metric {{
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
        margin: 1rem 0;
    }}

    /* Hide redundant elements */
    header, footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# --- APP LAYOUT ---

# Top Marquee
st.markdown('<div class="marquee"> PREDICTIVE MAINTENANCE • FLEET ANALYSIS • AI DRIVEN • RISK ASSESSMENT • REAL-TIME DATA • PREDICTIVE MAINTENANCE • </div>', unsafe_allow_html=True)

# Header Section
st.markdown(f"""
<div class="brutalist-container">
    <div class="agency-header">
        <div class="hero-subtitle">FOR_DIGITAL_INSIDERS_&_FLEET_MANAGERS_</div>
    </div>
    <div class="grid-row">
        <div class="grid-cell" style="background: white; color: black;">
            <h3>ABOUT US_</h3>
            <p>We are the vanguard of algorithmic vehicle health. FleetMind AI leverages deep learning to anticipate mechanical failure before it occurs.</p>
        </div>
        <div class="grid-cell">
            <h3>SERVICES_</h3>
            <ul style="list-style: none; padding: 0; font-weight: 700;">
                <li>→ RISK MODELING</li>
                <li>→ ANOMALY DETECTION</li>
                <li>→ MAINTENANCE SCHEDULING</li>
            </ul>
        </div>
        <div class="grid-cell" style="background: #e0e0e0;">
             <h3>CONTACT_</h3>
             <p>AUTO_THINK@FLEETMIND.AI<br>+1 (800) AI-FLEET</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('fleetmind_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"FATAL_ERROR: {e}")
    st.stop()

# Input Section
st.markdown('<h2 style="font-size: 3rem; color: black !important;">VEHICLE_FEED_</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="brutalist-container">', unsafe_allow_html=True)
    with st.form("vehicle_form_brutalist"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("### RAW_SPECS")
            vehicle_model = st.selectbox("MODEL", ["Sedan", "SUV", "Truck", "Van", "Bus"])
            fuel_type = st.selectbox("FUEL", ["Petrol", "Diesel", "Electric", "Hybrid"])
            transmission = st.selectbox("GEARBOX", ["Automatic", "Manual"])
            owner_type = st.selectbox("OWNER_RANK", ["First", "Second", "Third", "Fourth & Above"])

        with c2:
            st.markdown("### TELEMETRY")
            mileage = st.number_input("MILEAGE (KM)", min_value=0, value=50000, step=1000)
            vehicle_age = st.number_input("AGE (YRS)", min_value=0, value=5, step=1)
            engine_size = st.number_input("ENGINE (CC)", min_value=500, max_value=10000, value=1500, step=100)
            odometer = st.number_input("ODOMETER (KM)", min_value=0, value=55000, step=1000)

        with c3:
            st.markdown("### HISTORY")
            reported_issues = st.number_input("ISSUES_LOG", min_value=0, value=1, step=1)
            insurance_premium = st.number_input("PREMIUM ($)", min_value=0, value=500, step=50)
            accident_history = st.number_input("INCIDENTS", min_value=0, value=0, step=1)
            fuel_efficiency = st.number_input("EFFICIENCY (KM/L)", min_value=0.0, value=15.0, step=0.5)

        st.markdown("---")
        cx1, cx2 = st.columns(2)
        with cx1:
            days_since_service = st.number_input("DAYS_SINCE_SVC", min_value=0, value=90, step=1)
        with cx2:
            warranty_days_left = st.number_input("WARRANTY_TTL", min_value=-3650, value=365, step=1)

        submitted = st.form_submit_button("PROCESS_PREDICTION ↗")
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Result
if submitted:
    input_data = pd.DataFrame([{
        "Vehicle_Model": vehicle_model,
        "Fuel_Type": fuel_type,
        "Transmission_Type": transmission,
        "Owner_Type": owner_type,
        "Mileage": mileage,
        "Reported_Issues": reported_issues,
        "Vehicle_Age": vehicle_age,
        "Engine_Size": engine_size,
        "Odometer_Reading": odometer,
        "Insurance_Premium": insurance_premium,
        "Accident_History": accident_history,
        "Fuel_Efficiency": fuel_efficiency,
        "Days_Since_Last_Service": days_since_service,
        "Warranty_Days_Left": warranty_days_left
    }])

    try:
        prediction = model.predict(input_data)[0]
        predict_proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card status-risk">
                <h1 style="color:white;">STATUS: CRITICAL_DANGER</h1>
                <div class="big-metric">{predict_proba*100:.0f}%</div>
                <p style="font-size: 1.5rem; font-weight: 900;">MAINTENANCE_REQUIRED_IMMEDIATELY</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card status-healthy">
                <h1>STATUS: OPERATIONAL</h1>
                <div class="big-metric">{predict_proba*100:.0f}%</div>
                <p style="font-size: 1.5rem; font-weight: 900;">NO_THREAT_DETECTED</p>
            </div>
            """, unsafe_allow_html=True)

        # Contribution Factors
        try:
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']

            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                num_features = input_data.select_dtypes(exclude="object").columns
                feature_names = np.concatenate([cat_features, num_features])
                X_transformed = preprocessor.transform(input_data)
                active_importances = X_transformed[0] * importances
                
                normalized_importances = np.abs(active_importances) / np.sum(np.abs(active_importances)) * 100 if np.sum(np.abs(active_importances)) > 0 else np.abs(active_importances)

                importance_df = pd.DataFrame({'Feature': feature_names, 'Impact': normalized_importances})
                top_factors = importance_df.sort_values(by='Impact', ascending=False).head(3).to_dict('records')

                st.markdown('<div class="brutalist-container" style="background:white; padding: 2rem;">', unsafe_allow_html=True)
                st.markdown("<h3>ROOT_CAUSES_</h3>", unsafe_allow_html=True)
                cols = st.columns(3)
                for i, factor in enumerate(top_factors):
                    with cols[i]:
                        name = factor['Feature'].replace('cat__', '').replace('remainder__', '').replace('_', ' ').upper()
                        st.markdown(f"""
                        <div style="border: 2px solid black; padding: 1rem; text-align: center;">
                            <div style="font-weight: 900; font-size: 0.8rem;">{name}</div>
                            <div style="font-size: 2rem; font-weight: 900;">{factor['Impact']:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass

        # AI Agent Insights (RAG Powered)
        st.markdown('<h2 style="font-size: 2.5rem; color: black !important; margin-top: 3rem;">AI_AGENT_INSIGHTS_ (RAG_POWERED)</h2>', unsafe_allow_html=True)
        
        with st.spinner("Analyzing maintenance knowledge base..."):
            input_dict = {
                "Vehicle_Model": vehicle_model,
                "Fuel_Type": fuel_type,
                "Transmission_Type": transmission,
                "Owner_Type": owner_type,
                "Mileage": mileage,
                "Reported_Issues": reported_issues,
                "Vehicle_Age": vehicle_age,
                "Engine_Size": engine_size,
                "Odometer_Reading": odometer,
                "Insurance_Premium": insurance_premium,
                "Accident_History": accident_history,
                "Fuel_Efficiency": fuel_efficiency,
                "Days_Since_Last_Service": days_since_service,
                "Warranty_Days_Left": warranty_days_left
            }
            report = run_agent_pipeline(input_dict)

        if "health_summary" in report:
            st.markdown(f"""
            <div class="brutalist-container" style="background:white; padding: 2rem; border-left: 10px solid black;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 2;">
                        <h3>HEALTH_SUMMARY_</h3>
                        <p style="font-size: 1.2rem; font-weight: 700;">{report['health_summary']}</p>
                    </div>
                    <div style="flex: 1; text-align: right; border-left: 2px solid black; padding-left: 1rem;">
                         <div style="font-weight: 900; font-size: 0.8rem;">DETERMINISTIC_CONFIDENCE</div>
                         <div style="font-size: 2.5rem; font-weight: 900;">{report['confidence']}</div>
                    </div>
                </div>
                <hr style="border: 1px solid black; margin: 1.5rem 0;">
                <div class="grid-row" style="border-bottom: none;">
                    <div class="grid-cell" style="border-right: 2px solid black;">
                        <h4>REQUIRED_ACTIONS_</h4>
                        <ul style="font-weight: 700;">
                            {''.join([f"<li>{action}</li>" for action in report.get('actions', [])])}
                        </ul>
                    </div>
                    <div class="grid-cell" style="border-right: 2px solid black;">
                        <h4>SERVICE_TIMELINE_</h4>
                        <div style="font-weight: 900; font-size: 1.5rem; color: #FF3131 !important;">{report['timeline']}</div>
                    </div>
                    <div class="grid-cell">
                        <h4>KNOWLEDGE_SOURCES_</h4>
                        <div style="font-weight: 700; opacity: 0.7;">{report.get('sources', 'N/A')}</div>
                        <div style="font-size: 0.7rem; margin-top: 1rem; font-style: italic;">{report.get('disclaimer', '')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- PDF EXPORT FEATURE ---
            try:
                pdf_path = generate_pdf(report)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="DOWNLOAD_MAINTENANCE_REPORT_PDF ↗",
                        data=f,
                        file_name="fleet_maintenance_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as pdf_err:
                st.warning(f"PDF_GENERATION_FAILED: {pdf_err}")
        else:
            st.error(f"AGENT_ERROR: {report.get('message', 'Unknown failure')}")

    except Exception as e:
        st.error(f"COMPUTE_ERROR: {str(e)}")

# Bottom Footer
st.markdown('<div class="marquee" style="margin-top: 5rem;"> SEARCH_BERG_INTEL • 2026 • ALL_RIGHTS_RESERVED • DATA_LENS • </div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; padding: 2rem; font-weight: 900; opacity: 0.5;'>VERSION_4.2.0 • CORE_KERNEL_OPTIMIZED</p>", unsafe_allow_html=True)

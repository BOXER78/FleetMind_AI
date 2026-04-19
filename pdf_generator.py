import os
from fpdf import FPDF
from datetime import datetime

def generate_pdf(report: dict) -> str:
    """
    Generates a professional PDF maintenance report from structured JSON data.
    """
    # Create PDF instance
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- HELPER: SANITIZE TEXT ---
    def sanitize(text):
        """Replaces common non-Latin-1 characters (en-dash, em-dash, etc.) with ASCII equivalents."""
        if not text:
            return "N/A"
        text = str(text)
        replacements = {
            "\u2013": "-", # en-dash
            "\u2014": "-", # em-dash
            "\u2018": "'", # left single quote
            "\u2019": "'", # right single quote
            "\u201c": '"', # left double quote
            "\u201d": '"', # right double quote
            "\u2022": "*", # bullet point
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text

    # --- HELPER: FALLBACK ---
    def get_val(key):
        return sanitize(report.get(key, "N/A"))

    # --- TITLE ---
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 20, "Fleet Maintenance Report", ln=True, align="C")
    
    # --- TIMESTAMP ---
    pdf.set_font("helvetica", "I", 10)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated On: {now}", ln=True, align="C")
    pdf.ln(10)
    
    # --- HEALTH SUMMARY ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "1. Health Summary", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.multi_cell(0, 8, get_val("health_summary"))
    pdf.ln(5)
    
    # --- RISK LEVEL ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Risk Assessment", ln=True)
    pdf.set_font("helvetica", "B", 12)
    risk_level = get_val("risk_level").upper()
    pdf.cell(0, 8, f"Status: {risk_level}", ln=True)
    pdf.ln(5)
    
    # --- RECOMMENDED ACTIONS ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Recommended Actions", ln=True)
    pdf.set_font("helvetica", "", 12)
    actions = report.get("actions", [])
    if isinstance(actions, list) and actions:
        for action in actions:
            pdf.cell(5) # Indent
            pdf.cell(0, 8, f"- {sanitize(action)}", ln=True)
    else:
        pdf.cell(5)
        pdf.cell(0, 8, "- No specific actions required", ln=True)
    pdf.ln(5)
    
    # --- TIMELINE ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "4. Service Timeline", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, get_val("timeline"), ln=True)
    pdf.ln(5)
    
    # --- CONFIDENCE / PROBABILITY ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "5. AI Confidence Score", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, f"Probability Score: {get_val('confidence')}", ln=True)
    pdf.ln(5)
    
    # --- SOURCES ---
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "6. Knowledge Sources", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 8, get_val("sources"), ln=True)
    pdf.ln(5)
    
    # --- DISCLAIMER ---
    pdf.set_font("helvetica", "I", 10)
    pdf.multi_cell(0, 6, f"Disclaimer: {get_val('disclaimer')}")
    
    # --- FOOTER ---
    pdf.set_y(-25)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 10, "FLEETMIND AI - CORE_KERNEL_OPTIMIZED", align="C")

    # Output path
    output_path = "maintenance_report.pdf"
    pdf.output(output_path)
    
    return output_path

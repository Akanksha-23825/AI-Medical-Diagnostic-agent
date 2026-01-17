import os
import concurrent.futures
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from google import genai 

# ---------------- CONFIG ---------------- #
client = genai.Client()
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# For local testing only (Ensure this path exists if you run locally)
medical_report_path_placeholder = os.path.join(
    "Medical Reports",
    "Medical Report - Laura Garcia - Rheumatoid Arthritis.txt"
)

# ---------------- UTILITY FUNCTIONS ---------------- #

def load_medical_report(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def save_report(file_name, content, header=""):
    try:
        path = os.path.join(RESULTS_DIR, file_name)
        with open(path, 'w', encoding='utf-8') as f:
            if header:
                f.write(f"--- {header} ---\n\n")
            f.write(content)
    except Exception as e:
        print(f"Error saving file {file_name}: {e}")


def run_specialist_agent(agent_class, medical_report, base_filename):
    print(f"{agent_class.__name__} is running...")
    agent = agent_class(medical_report)
    report_content = agent.run()

    if report_content:
        filename = f"{base_filename.replace('.txt', '')}_{agent_class.__name__.lower()}_report.txt"
        save_report(filename, report_content, agent_class.__name__)

    return report_content


# ---------------- PATIENT SUMMARY (NEW + CRITICAL for concise output) ---------------- #

def generate_patient_summary(full_mdt_report):
    """Generates a short, non-technical summary of the MDT report."""
    print("Generating patient summary...")
    prompt = f"""
You are a medical assistant.

Create a SHORT and CLEAR patient-friendly medical report summary from the information provided below.

Rules:
- Use simple, non-technical language.
- Maximum 8 bullet points.
- No medical jargon.
- No doctor or specialist names.
- Focus ONLY on the confirmed diagnosis and clear next steps/treatment.

Medical information:
{full_mdt_report}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# ---------------- MAIN PIPELINE ---------------- #

def run_multi_agent_analysis(input_medical_report_path: str):

    # 1. Load report
    medical_report = load_medical_report(input_medical_report_path)
    if not medical_report:
        return None

    base_filename = os.path.basename(input_medical_report_path)

    # 2. Run specialist agents concurrently
    specialist_agents = [Cardiologist, Psychologist, Pulmonologist]
    specialist_outputs = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {
            executor.submit(run_specialist_agent, agent, medical_report, base_filename): agent.__name__.lower()
            for agent in specialist_agents
        }

        for future in concurrent.futures.as_completed(future_map):
            agent_name = future_map[future]
            try:
                specialist_outputs[agent_name] = future.result()
            except Exception as e:
                specialist_outputs[agent_name] = f"Error generating {agent_name} report: {e}"

    cardiologist_report = specialist_outputs.get("cardiologist", "")
    psychologist_report = specialist_outputs.get("psychologist", "")
    pulmonologist_report = specialist_outputs.get("pulmonologist", "")

    # 3. MDT Agent
    print("MultidisciplinaryTeam is running...")
    mdt_agent = MultidisciplinaryTeam(
        cardiologist_report=cardiologist_report,
        psychologist_report=psychologist_report,
        pulmonologist_report=pulmonologist_report
    )

    final_mdt_report = mdt_agent.run()

    if not final_mdt_report:
        return None

    # 4. Save INTERNAL doctor report
    internal_filename = f"{base_filename.replace('.txt', '')}_internal_mdt_report.txt"
    save_report(internal_filename, final_mdt_report, "Multidisciplinary Team Report (Internal)")

    # 5. Generate PATIENT SUMMARY
    patient_summary = generate_patient_summary(final_mdt_report)

    patient_filename = f"{base_filename.replace('.txt', '')}_patient_summary.txt"
    save_report(patient_filename, patient_summary, "Patient Summary (Frontend)")

    # 6. RETURN DATA TO FLASK
    return {
        "final_report_path": os.path.join(RESULTS_DIR, patient_filename),
        "raw_mdt_report": final_mdt_report,       # doctor-only
        "patient_report": patient_summary,         # frontend
        "cardiologist_report": cardiologist_report,
        "pulmonologist_report": pulmonologist_report,
        "psychologist_report": psychologist_report
    }


# ---------------- LOCAL TESTING ---------------- #

if __name__ == "__main__":
    print("Running local test...")
    result = run_multi_agent_analysis(medical_report_path_placeholder)

    if result:
        print("\nSUCCESS")
        print("Patient summary preview:\n")
        print(result["patient_report"])
    else:
        print("\nFAILED")
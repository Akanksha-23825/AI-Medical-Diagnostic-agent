import os
import json
import glob
import concurrent.futures
from dotenv import load_dotenv
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
from google import genai

load_dotenv(dotenv_path='apikey.env', override=True)

client = genai.Client()
RESULTS_DIR = "results"
CACHE_DIR = "cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

medical_report_path_placeholder = os.path.join(
    "Medical Reports",
    "Medical Report - Laura Garcia - Rheumatoid Arthritis.txt"
)

def cache_path_for(filename):
    safe = filename.replace(".txt", "").replace(".pdf", "").replace(".doc", "").replace(".docx", "")
    return os.path.join(CACHE_DIR, f"{safe}.json")

def save_cache(filename, data):
    try:
        with open(cache_path_for(filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Cache save failed:", e)

def load_cache(filename):
    try:
        path = cache_path_for(filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Cache load failed:", e)
        return None

def looks_like_error(text):
    if not text:
        return True
    lower = text.lower()
    return "unable to" in lower or "error" in lower or "cannot" in lower or "quota" in lower

def rebuild_cache_from_results():
    bases = set()

    def base_from_filename(path):
        name = os.path.basename(path)
        for suffix in [
            "_patient_summary.txt",
            "_internal_mdt_report.txt",
            "_cardiologist_report.txt",
            "_psychologist_report.txt",
            "_pulmonologist_report.txt",
        ]:
            if name.endswith(suffix):
                return name.replace(suffix, "")
        return None

    def read_if_exists(path):
        return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

    for f in glob.glob(os.path.join(RESULTS_DIR, "*.txt")):
        base = base_from_filename(f)
        if base:
            bases.add(base)

    for base in bases:
        data = {
            "final_report_path": os.path.join(RESULTS_DIR, f"{base}_patient_summary.txt"),
            "raw_mdt_report_path": os.path.join(RESULTS_DIR, f"{base}_internal_mdt_report.txt"),
            "raw_mdt_report": read_if_exists(os.path.join(RESULTS_DIR, f"{base}_internal_mdt_report.txt")),
            "patient_report": read_if_exists(os.path.join(RESULTS_DIR, f"{base}_patient_summary.txt")),
            "immediate_steps": [],
            "followup_steps": [],
            "cardiologist_report": read_if_exists(os.path.join(RESULTS_DIR, f"{base}_cardiologist_report.txt")),
            "psychologist_report": read_if_exists(os.path.join(RESULTS_DIR, f"{base}_psychologist_report.txt")),
            "pulmonologist_report": read_if_exists(os.path.join(RESULTS_DIR, f"{base}_pulmonologist_report.txt")),
        }

        if not looks_like_error(data["patient_report"]):
            save_cache(base, data)

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

def generate_patient_summary(full_mdt_report):
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

def generate_steps_split(full_mdt_report):
    prompt = f"""
You are a clinical assistant.

From the information below, produce TWO separate bullet lists:

1) Immediate Steps (actions to do now or within days)
2) Follow-up Steps (actions for later weeks/months)

Rules:
- Each item must be short and action-focused.
- Immediate Steps: exactly 3 bullets.
- Follow-up Steps: 3 to 5 bullets.
- No medical jargon.
- Do not repeat items.
- Do not include headings, only bullets.

Medical info:
{full_mdt_report}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip().splitlines()
    immediate, followup = [], []
    current = "immediate"

    for line in text:
        clean = line.strip()
        if not clean:
            continue
        if "immediate" in clean.lower():
            current = "immediate"
            continue
        if "follow" in clean.lower():
            current = "followup"
            continue
        if clean.startswith("-") or clean.startswith("•"):
            clean = clean[1:].strip()
        if current == "immediate":
            immediate.append(clean)
        else:
            followup.append(clean)

    return immediate[:3], followup[:5]

def run_multi_agent_analysis(input_medical_report_path: str):
    medical_report = load_medical_report(input_medical_report_path)
    if not medical_report:
        return None

    base_filename = os.path.basename(input_medical_report_path)

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

    if (
        looks_like_error(cardiologist_report) or
        looks_like_error(psychologist_report) or
        looks_like_error(pulmonologist_report)
    ):
        cached = load_cache(base_filename)
        if cached and not looks_like_error(cached.get("patient_report", "")):
            print("API error detected — using cached results.")
            return cached

    print("MultidisciplinaryTeam is running...")
    mdt_agent = MultidisciplinaryTeam(
        cardiologist_report=cardiologist_report,
        psychologist_report=psychologist_report,
        pulmonologist_report=pulmonologist_report
    )

    final_mdt_report = mdt_agent.run()

    if not final_mdt_report or looks_like_error(final_mdt_report):
        cached = load_cache(base_filename)
        if cached and not looks_like_error(cached.get("patient_report", "")):
            print("MDT error detected — using cached results.")
            return cached
        return None

    internal_filename = f"{base_filename.replace('.txt', '')}_internal_mdt_report.txt"
    save_report(internal_filename, final_mdt_report, "Multidisciplinary Team Report (Internal)")

    patient_summary = generate_patient_summary(final_mdt_report)
    immediate_steps, followup_steps = generate_steps_split(final_mdt_report)

    if not followup_steps:
        summary_lines = [l.strip("-•* ").strip() for l in patient_summary.split("\n") if l.strip()]
        followup_steps = summary_lines[3:6]

    patient_filename = f"{base_filename.replace('.txt', '')}_patient_summary.txt"
    save_report(patient_filename, patient_summary, "Patient Summary (Frontend)")

    result_payload = {
        "final_report_path": os.path.join(RESULTS_DIR, patient_filename),
        "raw_mdt_report_path": os.path.join(RESULTS_DIR, internal_filename),
        "raw_mdt_report": final_mdt_report,
        "patient_report": patient_summary,
        "immediate_steps": immediate_steps,
        "followup_steps": followup_steps,
        "cardiologist_report": cardiologist_report,
        "pulmonologist_report": pulmonologist_report,
        "psychologist_report": psychologist_report
    }

    if not looks_like_error(result_payload["patient_report"]):
        save_cache(base_filename, result_payload)
        rebuild_cache_from_results()
    else:
        print("Skipping cache save (error-like response).")

    return result_payload

if __name__ == "__main__":
    print("Running local test...")
    result = run_multi_agent_analysis(medical_report_path_placeholder)

    if result:
        print("\nSUCCESS")
        print("Patient summary preview:\n")
        print(result["patient_report"])
    else:
        print("\nFAILED")

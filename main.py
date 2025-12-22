import os
import concurrent.futures
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam

# --- Configuration ---
# Create the results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# NOTE: The medical_report_path variable is now ONLY a placeholder for local testing
# It is NOT used when imported by app.py
# If you run main.py directly, it will use this path.
medical_report_path_placeholder = os.path.join(
    "Medical Reports", 
    "Medical Report - Laura Garcia - Rheumatoid Arthritis.txt"
)


def load_medical_report(file_path):
    """Loads the patient's medical report from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Medical report file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def save_report(file_name, content, agent_name=""):
    """Saves the generated report content to a file in the results directory."""
    try:
        path = os.path.join(RESULTS_DIR, file_name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"--- {agent_name} Report ---\n\n")
            f.write(content)
        # We comment out print statements to keep the Flask console clean
        # print(f"Report saved to {path}") 
    except Exception as e:
        print(f"Error saving file {file_name}: {e}")

def run_specialist_agent(agent_class, medical_report, report_filename):
    """Initializes and runs a single specialist agent."""
    agent = agent_class(medical_report)
    report_content = agent.run()
    
    # Save the individual specialist report
    if report_content:
        # Use the base filename from the medical report to keep results unique per patient
        specialist_filename = f"{report_filename.replace('.txt', '')}_{agent.__class__.__name__.lower()}_report.txt"
        save_report(specialist_filename, report_content, agent.__class__.__name__)
    
    return report_content

# --- NEW FUNCTION FOR WEB APP INTEGRATION ---

def run_multi_agent_analysis(input_medical_report_path: str):
    """
    Runs the entire multi-agent pipeline for a given medical report file.
    Returns the path to the final diagnosis file.
    """
    # 1. Load the initial report
    medical_report = load_medical_report(input_medical_report_path)
    if not medical_report:
        # Returning None if the report is not found/readable
        return None 

    # Extract the base filename for unique output file naming
    report_filename_base = os.path.basename(input_medical_report_path)

    # 2. Run Specialist Agents Concurrently
    specialist_agents = [Cardiologist, Psychologist, Pulmonologist]
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks to the executor
        future_to_agent = {
            executor.submit(run_specialist_agent, agent_class, medical_report, report_filename_base): agent_class.__name__
            for agent_class in specialist_agents
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            try:
                report = future.result()
                results[agent_name.lower()] = report if report else f"Error: {agent_name} report failed to generate."
            except Exception as exc:
                # print(f'{agent_name} generated an exception: {exc}') # Keep clean for web app
                results[agent_name.lower()] = f"Error: Exception during {agent_name} run: {exc}"

    # 3. Prepare reports for the Multidisciplinary Team (MDT)
    cardiologist_report = results.get('cardiologist', '')
    psychologist_report = results.get('psychologist', '')
    pulmonologist_report = results.get('pulmonologist', '')

    # 4. Run the Multidisciplinary Team Agent
    mdt_agent = MultidisciplinaryTeam(
        cardiologist_report=cardiologist_report,
        psychologist_report=psychologist_report,
        pulmonologist_report=pulmonologist_report
    )
    final_diagnosis = mdt_agent.run()

    # 5. Save the Final Diagnosis
    final_report_name = f"{report_filename_base.replace('.txt', '')}_final_diagnosis.txt"
    final_report_path = os.path.join(RESULTS_DIR, final_report_name)
    
    if final_diagnosis:
        # Renaming the existing save_report logic to save the final file
        try:
            with open(final_report_path, 'w', encoding='utf-8') as f:
                f.write(f"--- Multidisciplinary Team Report ---\n\n")
                f.write(final_diagnosis)
        except Exception as e:
            print(f"Error saving final diagnosis: {e}")
            return None
        
        return final_report_path
    
    return None

# --- Local Testing Execution Block (Optional) ---
# This block is for running main.py directly without the web app.
if __name__ == "__main__":
    print(f"--- Running Pipeline for Local Testing: {os.path.basename(medical_report_path_placeholder)} ---")
    final_path = run_multi_agent_analysis(medical_report_path_placeholder)
    if final_path:
        print(f"\nSUCCESS: Final report generated at {final_path}")
    else:
        print("\nFAILURE: Could not generate the final report.")
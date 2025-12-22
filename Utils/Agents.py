from langchain_core.prompts import PromptTemplate
import os
# --- NEW IMPORTS FOR AUTH AND MODERN API ---
from dotenv import load_dotenv
from google import genai
from google.genai import types 
import time 
from google.genai.errors import APIError
# --------------------------------------------

# Load environment variables at the top level
# CRITICAL: Use the GEMINI_API_KEY variable and load the apikey.env file
# load_dotenv(dotenv_path='apikey.env')

# Define Constants
MODEL_NAME = "gemini-2.5-flash" 
MAX_TOKENS = 4096

class Agent:
    """
    Base Agent class for doctors or multidisciplinary team.
    Handles prompt creation and execution with the Gemini API.
    """
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}
        # The prompt template is created *after* the role is set
        
        # CRITICAL FIX 1: API Key and Client Initialization
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Note: This error likely points to the root of the 403 issue if the key file is present
            raise ValueError("GEMINI_API_KEY is not set. Please check apikey.env and load_dotenv setup.")
            
        # Use the modern Client initialization
        self.client = genai.Client(api_key=api_key)
        
        # Call create_prompt_template here
        self.prompt_template = self.create_prompt_template()


    def create_prompt_template(self):
        
        # --- MULTIDISCIPLINARY TEAM PROMPT (Synthesis) ---
        if self.role == "MultidisciplinaryTeam":
            templates = """
                Act as a Multidisciplinary Team of healthcare professionals. You will receive specialized reports from the Cardiologist, Psychologist, and Pulmonologist based on the patient's case.

                Task: Review and synthesize ALL three specialist reports. Your output must be a holistic, integrated final report.

                1. **Final Diagnosis:** Provide a list of 3 possible health issues for the patient, prioritized by likelihood, based on the combined evidence. For each issue, provide a brief reason referencing the specialist reports.
                2. **Integrated Treatment Plan:** Provide a detailed, integrated, bulleted Treatment and Management Plan. This plan must combine recommendations for medication, therapy, and lifestyle changes derived from all three specialist assessments.

                Just return the final diagnosis list followed immediately by the Integrated Treatment Plan.

                Cardiologist Report: {cardiologist_report}
                Psychologist Report: {psychologist_report}
                Pulmonologist Report: {pulmonologist_report}
            """
            return PromptTemplate.from_template(templates)

        # --- SPECIALIST AGENTS PROMPTS (Generalization Fix) ---
        else: 
            templates = {
                # General Prompt for Cardiologist
                "Cardiologist": """
                Act as an expert Cardiologist. Your sole task is to analyze the provided medical report, focusing ONLY on the cardiovascular system (heart, blood vessels, and blood pressure).

                1. Assess for any signs of primary cardiac conditions (e.g., arrhythmia, hypertension) or cardiac risks associated with the patient's primary complaints (e.g., inflammation).
                2. Provide a provisional assessment of the cardiac state and identify potential issues.
                3. List 3 specific, relevant next diagnostic steps, such as specific cardiac tests (e.g., ECG, echocardiogram, Holter monitor).
                Medical Report: {medical_report}
                """,
                
                # General Prompt for Psychologist
                "Psychologist": """
                Act as an expert Psychologist. Your sole task is to analyze the provided medical report, focusing ONLY on the mental and behavioral health status of the patient, including stress, anxiety, mood, and cognitive issues.

                1. Assess for any signs of primary psychological disorders, including generalized anxiety, panic disorder, depression, or somatization.
                2. Provide a provisional assessment of the patient's mental health status and identify significant stressors or behavioral risk factors.
                3. List 3 specific, relevant next diagnostic steps, such as specific psychological tests or therapy recommendations (e.g., CBT referral).
                Medical Report: {medical_report}
                """,
                
                # General Prompt for Pulmonologist
                "Pulmonologist": """
                Act as an expert Pulmonologist. Your sole task is to analyze the provided medical report, focusing ONLY on the respiratory system (lungs, airways, breathing).

                1. Assess for any signs of primary pulmonary conditions, including asthma, COPD, infection, or respiratory manifestations of systemic disease.
                2. Provide a provisional assessment of the respiratory state and identify potential issues like hyperventilation or abnormal gas exchange.
                3. List 3 specific, relevant next diagnostic steps, such as specific lung function tests (e.g., spirometry) or imaging (e.g., chest X-ray).
                Medical Report: {medical_report}
                """
            }
            return PromptTemplate.from_template(templates[self.role])




    def run(self):
        print(f"{self.role} is running...")
        
        prompt = self.prompt_template.format(
        medical_report=self.medical_report or '',
        cardiologist_report=self.extra_info.get('cardiologist_report', ''),
        psychologist_report=self.extra_info.get('psychologist_report', ''),
        pulmonologist_report=self.extra_info.get('pulmonologist_report', '')
)

        MAX_RETRIES = 3
        RETRY_DELAY_SECONDS = 15 # Wait 5 seconds between retries

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=MODEL_NAME, 
                    contents=[prompt], 
                    config=types.GenerateContentConfig(
                        max_output_tokens=MAX_TOKENS, 
                        temperature=0.2 
                    )
                )
                return response.text # Success! Return the response.
                
            except APIError as e:
                # Check specifically for 503 errors
                if "503 UNAVAILABLE" in str(e) and attempt < MAX_RETRIES - 1:
                    print(f"[{self.role}] Attempt {attempt + 1} failed (503 UNAVAILABLE). Retrying in {RETRY_DELAY_SECONDS} seconds...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue # Go to the next attempt
                else:
                    # If it's a permanent error (like 403 or 400) or last retry failed, raise the error
                    print(f"Error occurred in {self.role} after {attempt + 1} attempts:", e)
                    return f"Error: {e}"
            
            except Exception as e:
                # Catch other, non-API-specific errors
                print(f"Non-API Error occurred in {self.role}:", e)
                return f"Error: {e}"
        
        # Should be unreachable, but good practice
        return "Error: Failed to generate content after all retries."

# Define specialized agent classes (remain unchanged)
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from google.genai.errors import APIError

# Load environment variables
load_dotenv(dotenv_path='apikey.env', override=True)

MODEL_NAME = "gemini-2.5-flash-lite"
MAX_TOKENS = 4096

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please check apikey.env.")

        self.client = genai.Client(api_key=api_key)
        self.prompt_template = self.create_prompt_template()

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            templates = """
Act as a Multidisciplinary Team of healthcare professionals. 
Review and synthesize the reports from the Cardiologist, Psychologist, and Pulmonologist.

Task: Create a BALANCED summary of the findings. 
It should be detailed enough to explain the 'Why', but clear enough not to overwhelm the patient.

1. **Primary Observations:** Provide the top 3 prioritized health insights. For each, give a clear reason referencing the specialist reports in 1-2 sentences.
2. **Essential Care Plan:** Provide a bulleted list of the most important next steps (medication, lifestyle, or tests). Focus only on actionable items.

Style Guidelines:
- Use professional but supportive language.
- Avoid extremely dense medical jargon where a simpler term suffices.
- Keep the total length under 300 words.

Cardiologist Report: {cardiologist_report}
Psychologist Report: {psychologist_report}
Pulmonologist Report: {pulmonologist_report}
"""
            return PromptTemplate.from_template(templates)

        else:
            templates = {
                "Cardiologist": """
Act as an expert Cardiologist.

Rules:
- Start with "Key Findings:" and provide EXACTLY 3 bullet points.
- No introductions, no role descriptions.
- Each bullet must be a specific clinical finding or risk.
- Use plain language where possible.

Medical Report: {medical_report}
""",
                "Psychologist": """
Act as an expert Psychologist.

Rules:
- Start with "Key Findings:" and provide EXACTLY 3 bullet points.
- No introductions, no role descriptions.
- Each bullet must be a specific psychological finding or risk.
- Use plain language where possible.

Medical Report: {medical_report}
""",
                "Pulmonologist": """
Act as an expert Pulmonologist.

Rules:
- Start with "Key Findings:" and provide EXACTLY 3 bullet points.
- No introductions, no role descriptions.
- Each bullet must be a specific respiratory finding or risk.
- Use plain language where possible.

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
        RETRY_DELAY_SECONDS = 15

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
                return response.text

            except APIError as e:
                if "503 UNAVAILABLE" in str(e) and attempt < MAX_RETRIES - 1:
                    print(f"[{self.role}] Attempt {attempt + 1} failed (503 UNAVAILABLE). Retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    print(f"Error occurred in {self.role} after {attempt + 1} attempts:", e)
                    return f"Error: {e}"

            except Exception as e:
                print(f"Non-API Error occurred in {self.role}:", e)
                return f"Error: {e}"

        return "Error: Failed to generate content after all retries."


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

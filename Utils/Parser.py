# --- Utils/Parser.py ---

import os
import json
from google import genai
from google.genai import types
from google.genai.errors import APIError
# from dotenv import load_dotenv # Assuming this is loaded in app.py or main.py

class StructuredParser:
    def __init__(self):
        # Initialize client using the GEMINI_API_KEY environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # If the API key is missing, parsing will fail.
            print("WARNING: GEMINI_API_KEY is not set. StructuredParser cannot run.")
            self.client = None
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-lite" # Use the high-quota model
        self.max_tokens = 4096

    def parse_mdt_report_to_json(self, raw_mdt_report, analysis_results):
        """
        Sends the raw MDT text to Gemini with strict instructions to return a JSON object,
        and integrates the specialist reports from the analysis_results dictionary.
        """
        if not self.client:
            return None # Fail gracefully if API key is missing
            
        # 1. Define the desired JSON structure (Schema enforcement)
        json_schema = {
            "type": "object",
            "properties": {
                "final_diagnosis": {
                    "type": "array",
                    "description": "A list of the top 3 prioritized diagnoses, each with a name and a brief supporting reason (e.g., [{\"name\": \"Diagnosis 1\", \"reason\": \"Supporting evidence\"}]).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The disease or condition name."},
                            "reason": {"type": "string", "description": "The supporting medical reason from specialist reports."}
                        },
                        "required": ["name", "reason"]
                    }
                },
                "treatment_plan": {
                    "type": "string",
                    "description": "The full, integrated treatment and management plan derived from the MDT report, formatted as a detailed, multi-line string with bullet points."
                },
                # Note: We include specialist reports here, but the parser's job is only to extract 
                # the diagnosis and treatment plan from the MDT summary text. We will manually 
                # inject the raw specialist texts back into the final dict for simplicity.
                # However, the prompt needs to include the original specialist texts to ensure 
                # the MDT synthesis is robust.
            },
            "required": ["final_diagnosis", "treatment_plan"]
        }

        # 2. Prepare the combined prompt content
        # Include all raw texts to give the parser context, even if it only extracts MDT synthesis components.
        full_content = f"""
        Analyze the following raw Multidisciplinary Team Report, which is a synthesis of specialist reports, and extract the content strictly into the provided JSON schema.

        Your task is ONLY to extract the final diagnosis and the integrated treatment plan from the synthesis text below.

        RAW MDT SYNTHESIS REPORT:
        ---
        {raw_mdt_report}
        ---
        
        Specialist Reports for Context (DO NOT include this text in the JSON output, use it only to understand the structure of the diagnosis and plan):
        - Cardiologist: {analysis_results.get('cardiologist_report', 'N/A')}
        - Pulmonologist: {analysis_results.get('pulmonologist_report', 'N/A')}
        - Psychologist: {analysis_results.get('psychologist_report', 'N/A')}
        """
        
        # 3. Configure and run the request
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=json_schema,
            max_output_tokens=self.max_tokens,
            temperature=0.0 # Use low temperature for reliable, deterministic parsing
        )
        
        try:
            print("Running StructuredParser to convert MDT text to JSON...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[full_content],
                config=config
            )
            
            # The response text will be a clean JSON string
            parsed_json = json.loads(response.text)
            
            # 4. Integrate the raw specialist reports into the final dictionary
            # This avoids having the LLM re-output the raw reports, which is inefficient.
            parsed_json['specialist_reports'] = {
                'Cardiologist': analysis_results.get('cardiologist_report', 'No report available.'),
                'Pulmonologist': analysis_results.get('pulmonologist_report', 'No report available.'),
                'Psychologist': analysis_results.get('psychologist_report', 'No report available.')
            }
            
            return parsed_json
            
        except APIError as e:
            print(f"Error during JSON parsing API call: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from LLM output. Raw output: {response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during parsing: {e}")
            return None
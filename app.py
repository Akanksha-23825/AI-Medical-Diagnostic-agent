from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from main import run_multi_agent_analysis

# --- CONFIGURATION & ENVIRONMENT SETUP ---
load_dotenv(dotenv_path='apikey.env', override=True)

print("Loaded key:", "YES" if os.getenv("GEMINI_API_KEY") else "NO")
print("Configuration: Environment variables loaded.")

app = Flask(__name__)

# Define the folder where uploaded reports will be temporarily stored
UPLOAD_FOLDER = 'Medical Reports/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}


def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- ROUTES ---

@app.route("/", methods=["GET"])
def index():
    """Serves the file upload form (index.html)."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles file upload, runs the pipeline, and displays results."""

    # 1. Handle file retrieval and error checking
    if "file" not in request.files:
        return render_template("results.html", error="No file part in the request.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("results.html", error="No selected file.")

    if not allowed_file(file.filename):
        return render_template("results.html", error="Invalid file type. Please upload a TXT, PDF, DOC, or DOCX file.")

    # 2. Save the file
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f"--- Running pipeline for uploaded file: {filepath} ---")

    # 3. Run the Multi-Agent Pipeline
    results = run_multi_agent_analysis(filepath)

    if not results or "patient_report" not in results:
        return render_template(
            "results.html",
            error="Failed to generate diagnosis. Check console logs for errors."
        )

    # 4. Display outputs in template
    return render_template(
    "results.html",
    patient_report=results["patient_report"],
    patient_file=os.path.basename(results["final_report_path"]),
    mdt_file=os.path.basename(results["raw_mdt_report_path"]),
    cardiologist=results["cardiologist_report"],
    psychologist=results["psychologist_report"],
    pulmonologist=results["pulmonologist_report"],
    immediate_steps=results.get("immediate_steps", []),
    followup_steps=results.get("followup_steps", [])
)



@app.route("/download/<path:filename>")
def download(filename):
    safe_path = os.path.join("results", filename)
    return send_file(safe_path, as_attachment=True)


# --- EXECUTION BLOCK ---
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    app.run(debug=True)

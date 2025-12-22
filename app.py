# app.py

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Import your core pipeline function from main.py
from main import run_multi_agent_analysis 

# --- CONFIGURATION ---
app = Flask(__name__)

# Define the folder where uploaded reports will be temporarily stored
UPLOAD_FOLDER = 'Medical Reports/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# You can set a limit on file size if needed (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'} # Only allow these file types

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # If the user is just visiting the page (GET request)
    if request.method == 'GET':
        # 1. Render the HTML form for the user to upload a file
        return render_template('index.html')
        
    # If the user submits the form (POST request)
    if request.method == 'POST':
        # Check if the file part is in the request
        if 'file' not in request.files:
            # If no file is selected, redirect back to the upload page
            return redirect(request.url) 

        file = request.files['file']

        # If the user selects a file but leaves the name empty
        if file.filename == '':
            return redirect(request.url)

        # Process the file if it exists and is allowed
        if file and allowed_file(file.filename):
            # 1. Securely save the filename to prevent path traversal issues
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 2. Save the file
            file.save(filepath)
            
            # 3. Process the file using your backend pipeline
            print(f"--- Running pipeline for uploaded file: {filepath} ---")
            final_report_path = run_multi_agent_analysis(filepath)
            
            # 4. Handle results and redirect to the display route
            if final_report_path:
                # Extract just the filename to pass to the results page
                result_filename = os.path.basename(final_report_path)
                return redirect(url_for('display_results', filename=result_filename))
            else:
                return render_template('error.html', message="Pipeline failed to generate the final report.")
        
        return render_template('error.html', message="Invalid file type. Please upload a TXT, PDF, DOC, or DOCX file.")


# Placeholder for the results route (to be implemented in Phase 2)
@app.route('/results/<filename>')
def display_results(filename):
    """
    Reads the content of the final diagnosis file and renders it on the results page.
    """
    # Construct the full path to the results file
    # Note: We use "results" folder, which is where your pipeline saves reports
    filepath = os.path.join("results", filename)
    
    try:
        # Read the content of the final report
        with open(filepath, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # Pass the plain text report to the results.html template
        return render_template('results.html', 
                               filename=filename,
                               report_content=report_content)
        
    except FileNotFoundError:
        return render_template('error.html', 
                               message=f"Error: The final report file '{filename}' was not found.")
    except Exception as e:
        return render_template('error.html', 
                               message=f"An error occurred while reading the report: {e}")

# --- EXECUTION BLOCK ---
if __name__ == '__main__':
    # Ensure the upload folder exists when the app starts
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True) # Run the Flask development server
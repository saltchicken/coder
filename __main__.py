import os
import re
import argparse
import sys
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# 1. The Strict Formatting Contract
# We explicitly tell the model how to structure its output and forbid any conversational filler.
SYSTEM_INSTRUCTION = """
You are an expert autonomous coding agent. 
You must output perfectly formatted, complete code files.
Do not include any conversational text, explanations, or introductory filler. 

You MUST wrap every single file you generate using the following strict XML/Markdown hybrid structure:

<file name="relative/path/to/filename.ext">
```language
# Raw code goes here
```
</file>

If you generate multiple files, output them sequentially using this exact block structure.
"""

def extract_and_save_files(ai_response_text: str, output_dir: str = "generated_workspace"):
    """
    Parses the AI's XML/Markdown hybrid output and writes the files to disk.
    """
    # Regex Breakdown:
    # <file name="([^"]+)">  -> Captures the filename inside the quotes (Group 1)
    # \s*```[^\n]*\n         -> Matches optional whitespace, ```, any language tag, and the newline
    # (.*?)                  -> Captures the actual raw code block (Group 2), non-greedy
    # \n```\s*</file>        -> Matches the closing backticks and the closing XML tag
    pattern = re.compile(r'<file name="([^"]+)">\s*```[^\n]*\n(.*?)\n```\s*</file>', re.DOTALL)
    
    matches = pattern.findall(ai_response_text)
    
    if not matches:
        print("Error: No files matching the strict XML schema were found in the response.")
        print("Raw response:\n", ai_response_text)
        return

    print(f"Found {len(matches)} file(s). Writing to './{output_dir}'...")

    for filename, code in matches:
        # Construct the full path
        file_path = os.path.join(output_dir, filename)
        
        # Ensure the subdirectories exist (e.g., if AI outputs 'src/utils/math.py')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the clean, unescaped code to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        print(f" -> Saved: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Coding Agent")
    parser.add_argument("prompt", help="The coding task for the AI to complete")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT env var)")
    parser.add_argument("--outdir", default="generated_workspace", 
                        help="The folder where generated files will be saved")
    args = parser.parse_args()

    # 2. Initialize the Client (Vertex AI / ADC)
    try:
        client = genai.Client(
            vertexai=True,
            project=args.project,
            location="us-central1",
            http_options=HttpOptions(api_version="v1")
        )
    except Exception as e:
        print(f"Authentication Error: Could not initialize client. Are you logged into gcloud? \nDetails: {e}")
        sys.exit(1)

    # 3. Configure the model parameters
    config = types.GenerateContentConfig(
        temperature=0.1,               # Low temperature for strict structural adherence
        max_output_tokens=8192,        # Maximize context for large files
        system_instruction=SYSTEM_INSTRUCTION
    )

    print(f"Generating code... (Target: ./{args.outdir}/)")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=args.prompt,
            config=config
        )
    except Exception as e:
        print(f"API Error: Request to Vertex AI failed. \nDetails: {e}")
        sys.exit(1)

    # 4. Check for truncation before processing
    if response.candidates[0].finish_reason != 'STOP':
        print(f"WARNING: Generation did not finish normally! Reason: {response.candidates[0].finish_reason}")
        print("Attempting to parse whatever was generated so far...")
    
    # 5. Extract and save
    extract_and_save_files(response.text, output_dir=args.outdir)

if __name__ == "__main__":
    main()

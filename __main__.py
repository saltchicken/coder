import os
import re
import argparse
import sys
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# --- Pydantic Schema for Iterative Planning ---
class FilePlan(BaseModel):
    filepath: str
    purpose: str

class ProjectArchitecture(BaseModel):
    files: list[FilePlan]

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

def extract_and_save_files(ai_response_text: str, output_dir: str = "generated_workspace", auto_save: bool = False):
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

    print(f"\nFound {len(matches)} file(s):")
    for filename, _ in matches:
        print(f"  - {filename}")

    if not auto_save:
        while True:
            choice = input("\nOptions: [p]review content, [s]ave files, [a]bort: ").strip().lower()
            if choice == 'p':
                for filename, code in matches:
                    print(f"\n{'='*40}\nFILE: {filename}\n{'='*40}\n{code}\n")
            elif choice == 's':
                break
            elif choice == 'a':
                print("Aborted saving these files.")
                return
            else:
                print("Invalid choice. Please enter 'p', 's', or 'a'.")

    print(f"Writing to './{output_dir}'...")

    for filename, code in matches:
        # Construct the full path
        file_path = os.path.join(output_dir, filename)
        
        # Ensure the subdirectories exist (e.g., if AI outputs 'src/utils/math.py')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the clean, unescaped code to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        print(f" -> Saved: {file_path}")

def generate_iteratively(client, prompt: str, output_dir: str, auto_save: bool):
    """
    Two-phase generation: plans the architecture first, then generates each file individually.
    """
    print("\n[Phase 1] Planning Project Architecture...")
    
    arch_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=ProjectArchitecture,
        system_instruction="You are a senior software architect. Given a project request, output the necessary file structure. Provide the relative filepath and a brief purpose for each file."
    )
    
    try:
        arch_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=arch_config
        )
        architecture = arch_response.parsed
    except Exception as e:
        print(f"Failed to generate architecture: {e}")
        return

    print(f"[Phase 1 Complete] Planned {len(architecture.files)} files.\n")
    
    # Configuration for Phase 2 (Strict XML formatting for the code)
    code_config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        system_instruction=SYSTEM_INSTRUCTION
    )

    print("[Phase 2] Generating Files...")
    accumulated_context = ""
    
    for file_plan in architecture.files:
        print(f" -> Generating {file_plan.filepath}...")
        
        # Build a highly contextual prompt for this specific file
        file_prompt = (
            f"Overall Project Context: {prompt}\n\n"
            f"Task: Write the complete, runnable code for the file: '{file_plan.filepath}'.\n"
            f"Purpose of this file: {file_plan.purpose}\n"
        )
        
        if accumulated_context:
            file_prompt += f"\nCode already generated for this project that you can import/use:\n{accumulated_context}\n"
        
        try:
            file_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=file_prompt,
                config=code_config
            )
            
            if file_response.candidates[0].finish_reason != 'STOP':
                print(f"    WARNING: {file_plan.filepath} generation cut off! Reason: {file_response.candidates[0].finish_reason}")
            
            # Use our existing, robust extraction logic
            extract_and_save_files(file_response.text, output_dir=output_dir, auto_save=auto_save)
            
            # Extract the raw code to add to our running memory for the next file
            pattern = re.compile(r'<file name="([^"]+)">\s*```[^\n]*\n(.*?)\n```\s*</file>', re.DOTALL)
            matches = pattern.findall(file_response.text)
            for filename, code in matches:
                accumulated_context += f"\n--- {filename} ---\n{code}\n"
            
        except Exception as e:
            print(f"    Error generating {file_plan.filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Coding Agent")
    parser.add_argument("prompt", help="The coding task for the AI to complete")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                        help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT env var)")
    parser.add_argument("--outdir", default="generated_workspace", 
                        help="The folder where generated files will be saved")
    parser.add_argument("--iterative", action="store_true", 
                        help="Enable iterative generation for massive projects")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompts and auto-save files")
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
    
    if args.iterative:
        # Route to the new iterative pipeline
        generate_iteratively(client, args.prompt, args.outdir, auto_save=args.yes)
    else:
        # Standard Single-Shot pipeline
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
        extract_and_save_files(response.text, output_dir=args.outdir, auto_save=args.yes)

if __name__ == "__main__":
    main()

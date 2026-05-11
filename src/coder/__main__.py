import os
import re
import argparse
import sys
import logging
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# --- Logging Setup ---
logging.basicConfig(
    filename="coder_api_usage.log",
    level=logging.INFO,
    format="%(asctime)s - API CALL - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

API_CALL_COUNT = 0

def generate_with_tracking(client, model: str, contents: str, config, description: str):
    """Wrapper to track, log, and execute API calls."""
    global API_CALL_COUNT
    API_CALL_COUNT += 1
    log_msg = f"Call #{API_CALL_COUNT} | {description}"
    logging.info(log_msg)
    print(f"\n[API Tracker] {log_msg}...")
    
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

# --- Pydantic Schema for Iterative Planning ---
class FilePlan(BaseModel):
    filepath: str
    purpose: str
    action: Literal["create", "edit", "skip"] = Field(
        description="Must be 'create' for new files, 'edit' for modifying existing files using search/replace, or 'skip' if no changes are needed."
    )

class ProjectArchitecture(BaseModel):
    files: list[FilePlan]

# 1. The Strict Formatting Contract (Updated for partial edits)
SYSTEM_INSTRUCTION = """
You are an expert autonomous coding agent. 
You must output perfectly formatted code. Do not include conversational text, explanations, or introductory filler. 

CRITICAL INSTRUCTION: ONLY generate <file> blocks for files that need to be CREATED or MODIFIED.

NEVER generate lock files (package-lock.json, poetry.lock) or dependency directories.

You have TWO modes of outputting files depending on the action required:

MODE 1: CREATING NEW FILES (or completely rewriting)
Use action="write" and wrap the complete code in a markdown block.
<file name="path/to/new_file.py" action="write">
```python
# Complete, runnable code goes here
```
</file>

MODE 2: EDITING EXISTING FILES (Search and Replace)
Use action="edit" and provide <search> and <replace> blocks. 
The <search> block MUST contain the EXACT, identical lines from the original file so they can be accurately found and replaced. Include a few lines of context above and below the change.
<file name="path/to/existing_file.py" action="edit">
<search>
def old_function():
    # old logic
    pass
</search>
<replace>
def old_function():
    # new logic
    print("Updated")
</replace>
</file>

You can include multiple <search>/<replace> pairs within a single <file action="edit"> block to modify multiple parts of the same file.
"""

def extract_and_save_files(ai_response_text: str, output_dir: str = "generated_workspace", auto_save: bool = False):
    """
    Parses the AI's XML/Markdown hybrid output for both full files and partial edits.
    """
    # Regex to find <file name="..." action="...">...</file>
    file_pattern = re.compile(r'<file name="([^"]+)"(?:\s+action="([^"]+)")?>\s*(.*?)\s*</file>', re.DOTALL)
    matches = file_pattern.findall(ai_response_text)
    
    if not matches:
        print("Error: No valid <file> blocks found in the response.")
        print("Raw response:\n", ai_response_text)
        return

    proposed_changes = []

    # Parse out exactly what needs to happen
    for filename, action, content in matches:
        action = action.lower() if action else "write"
        
        if action == "edit":
            # Extract search and replace pairs
            edit_pattern = re.compile(r'<search>\n?(.*?)\n?</search>\s*<replace>\n?(.*?)\n?</replace>', re.DOTALL)
            edits = edit_pattern.findall(content)
            proposed_changes.append({
                'filename': filename,
                'action': 'edit',
                'edits': edits
            })
        else:
            # Extract full code block
            code_pattern = re.compile(r'```[^\n]*\n(.*?)\n```', re.DOTALL)
            code_match = code_pattern.search(content)
            code = code_match.group(1) if code_match else content.strip()
            proposed_changes.append({
                'filename': filename,
                'action': 'write',
                'content': code
            })

    print(f"\nFound {len(proposed_changes)} file(s) to process:")
    for change in proposed_changes:
        action_label = "EDIT" if change['action'] == "edit" else "WRITE"
        print(f"  - [{action_label}] {change['filename']}")

    if not auto_save:
        while True:
            try:
                choice = input("\nOptions: [p]review content, [s]ave files, [a]bort: ").strip().lower()
                if choice == 'p':
                    for change in proposed_changes:
                        print(f"\n{'='*50}\nFILE: {change['filename']} ({change['action'].upper()})\n{'='*50}")
                        if change['action'] == 'write':
                            print(change['content'])
                        elif change['action'] == 'edit':
                            for i, (search_text, replace_text) in enumerate(change['edits']):
                                print(f"--- Edit #{i+1} ---")
                                print(f"<<<< SEARCH\n{search_text}\n====")
                                print(f">>>> REPLACE\n{replace_text}\n")
                        print("\n")
                elif choice == 's':
                    break
                elif choice == 'a':
                    print("Aborted saving these files.")
                    return
                else:
                    print("Invalid choice. Please enter 'p', 's', or 'a'.")
            except EOFError:
                print("\nError: Interactive prompt failed because standard input was piped. Use '-y' to bypass prompts.")
                return

    print(f"\nApplying changes to './{output_dir}'...")

    for change in proposed_changes:
        file_path = os.path.join(output_dir, change['filename'])
        
        if change['action'] == 'write':
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(change['content'])
            print(f" -> Wrote full file: {file_path}")

        elif change['action'] == 'edit':
            if not os.path.exists(file_path):
                print(f" -> ERROR: Cannot edit '{change['filename']}' because it does not exist in {output_dir}.")
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                file_text = f.read()

            new_text = file_text
            success_count = 0
            
            for search_text, replace_text in change['edits']:
                if search_text in new_text:
                    new_text = new_text.replace(search_text, replace_text)
                    success_count += 1
                else:
                    print(f" -> WARNING: Could not find matching <search> block in {change['filename']}. Skipping edit.")
                    print(f"    [Snippet]: {search_text[:60].strip()}...")
            
            if success_count > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                print(f" -> Applied {success_count}/{len(change['edits'])} edits to: {file_path}")

def generate_iteratively(client, prompt: str, output_dir: str, auto_save: bool):
    """
    Two-phase generation: plans the architecture first, then generates or edits each file.
    """
    arch_config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=ProjectArchitecture,
        system_instruction=(
            "You are a senior software architect. Given a project request and context, plan the necessary file modifications. "
            "For each file, provide the filepath, purpose, and the 'action' required ('create', 'edit', or 'skip'). "
            "CRITICAL: Use 'skip' for any existing files in the context that do NOT need modifications. "
            "NEVER include lock files (e.g., package-lock.json) or dependency folders."
        )
    )
    
    try:
        arch_response = generate_with_tracking(
            client=client,
            model="gemini-2.5-flash",
            contents=prompt,
            config=arch_config,
            description="Phase 1: Planning Project Architecture"
        )
        architecture = arch_response.parsed
    except Exception as e:
        print(f"Failed to generate architecture: {e}")
        return

    print(f"[Phase 1 Complete] Planned {len(architecture.files)} files.\n")
    
    code_config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        system_instruction=SYSTEM_INSTRUCTION
    )

    print("[Phase 2] Executing File Changes...")
    accumulated_context = ""
    
    for file_plan in architecture.files:
        if file_plan.action == "skip":
            print(f"  - Skipping '{file_plan.filepath}' (No changes required)")
            continue

        file_prompt = (
            f"Overall Project Context and Instructions: \n{prompt}\n\n"
            f"Task: Execute changes for the file: '{file_plan.filepath}'.\n"
            f"Purpose of this file: {file_plan.purpose}\n"
            f"Action Required: {file_plan.action}\n\n"
            f"Reminder: If Action is 'edit', ONLY output <search> and <replace> blocks inside <file action=\"edit\">."
        )
        
        if accumulated_context:
            file_prompt += f"\nCode already generated/modified in this session:\n{accumulated_context}\n"
        
        try:
            file_response = generate_with_tracking(
                client=client,
                model="gemini-2.5-flash",
                contents=file_prompt,
                config=code_config,
                description=f"Phase 2: Generating {file_plan.filepath} ({file_plan.action})"
            )
            
            if file_response.candidates[0].finish_reason != 'STOP':
                print(f"    WARNING: {file_plan.filepath} generation cut off! Reason: {file_response.candidates[0].finish_reason}")
            
            extract_and_save_files(file_response.text, output_dir=output_dir, auto_save=auto_save)
            
            # Simple context accumulation (just adding raw response for context to next files)
            accumulated_context += f"\n--- Actions performed on {file_plan.filepath} ---\n{file_response.text}\n"
            
        except Exception as e:
            print(f"    Error processing {file_plan.filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Coding Agent")
    parser.add_argument("prompt", help="The coding task for the AI to complete")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "coder-470"),
                        help="Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT env var)")
    parser.add_argument("--outdir", default="generated_workspace", 
                        help="The folder where generated files will be saved")
    parser.add_argument("--iterative", action="store_true", 
                        help="Enable iterative generation for massive projects")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompts and auto-save files")
    parser.add_argument("--context-file", 
                        help="Path to an exported codebase context file (e.g., from the 'context' CLI tool)")
    args = parser.parse_args()

    context_data = ""
    
    if args.context_file:
        try:
            with open(args.context_file, "r", encoding="utf-8") as f:
                context_data = f.read()
            print(f"Loaded {len(context_data)} bytes of context from {args.context_file}")
        except Exception as e:
            print(f"Error reading context file: {e}")
            sys.exit(1)
            
    elif not sys.stdin.isatty():
        context_data = sys.stdin.read()
        print(f"Loaded {len(context_data)} bytes of context from standard input.")
        if not args.yes:
            print("Notice: Piped input detected. Automatically enabling auto-save (-y) to prevent interactive prompt crash.")
            args.yes = True

    final_prompt = args.prompt
    if context_data:
        final_prompt = f"Here is the context of the existing codebase:\n\n{context_data}\n\nTask Instructions:\n{args.prompt}"

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

    config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=8192,
        system_instruction=SYSTEM_INSTRUCTION
    )

    print(f"Generating code... (Target: ./{args.outdir}/)")
    
    if args.iterative:
        generate_iteratively(client, final_prompt, args.outdir, auto_save=args.yes)
    else:
        try:
            response = generate_with_tracking(
                client=client,
                model="gemini-2.5-flash",
                contents=final_prompt,
                config=config,
                description="Single-Shot File Generation"
            )
        except Exception as e:
            print(f"API Error: Request to Vertex AI failed. \nDetails: {e}")
            sys.exit(1)

        if response.candidates[0].finish_reason != 'STOP':
            print(f"WARNING: Generation did not finish normally! Reason: {response.candidates[0].finish_reason}")
            print("Attempting to parse whatever was generated so far...")
        
        extract_and_save_files(response.text, output_dir=args.outdir, auto_save=args.yes)

if __name__ == "__main__":
    main()

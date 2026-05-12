use anyhow::{bail, Context, Result};
use clap::Parser;
use colored::*;
use log::info;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use similar::TextDiff;
use simplelog::*;
use time::macros::format_description;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

// --- Global State ---
static API_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

// Regex caches for performance
static FILE_PATTERN: OnceLock<Regex> = OnceLock::new();
static EDIT_PATTERN: OnceLock<Regex> = OnceLock::new();
static CODE_PATTERN: OnceLock<Regex> = OnceLock::new();

const SYSTEM_INSTRUCTION: &str = r#"
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
"#;

// --- CLI Arguments ---
#[derive(Parser, Debug)]
#[command(version, about = "Autonomous AI Coding Agent")]
struct Args {
    /// The coding task for the AI to complete
    prompt: String,

    /// Google Cloud Project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
    #[arg(long, default_value = "coder-470")]
    project: String,

    /// The folder where generated files will be saved
    #[arg(long, default_value = "generated_workspace")]
    outdir: String,

    /// Enable iterative generation for massive projects
    #[arg(long)]
    iterative: bool,

    /// Skip confirmation prompts and auto-save files
    #[arg(short = 'y', long = "yes")]
    yes: bool,

    /// Path to an exported codebase context file
    #[arg(long)]
    context_file: Option<String>,
}

// --- Schemas for Iterative Phase ---
#[derive(Debug, Serialize, Deserialize)]
struct FilePlan {
    filepath: String,
    purpose: String,
    action: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProjectArchitecture {
    files: Vec<FilePlan>,
}

struct MemoryOp {
    filename: String,
    path: PathBuf,
    action: String,
    new_text: String,
    diff: String,
    msg: String,
}

// --- Utility Functions ---

/// Gets the local gcloud access token to authenticate with Vertex AI
fn get_gcloud_token() -> Result<String> {
    let output = Command::new("gcloud")
        .args(["auth", "print-access-token"])
        .output()
        .context("Failed to execute gcloud CLI. Is it installed?")?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("gcloud auth failed: {}", err);
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Generates a git-style colorized unified diff string
fn generate_colorized_diff(original_text: &str, new_text: &str, filename: &str) -> String {
    let diff = TextDiff::from_lines(original_text, new_text);
    let unified = diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{}", filename), &format!("b/{}", filename))
        .to_string();

    if unified.is_empty() {
        return "\x1b[90mNo changes detected.\x1b[0m".to_string();
    }

    let mut colorized_diff = String::new();
    for line in unified.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            colorized_diff.push_str(&format!("{}\n", line.green()));
        } else if line.starts_with('-') && !line.starts_with("---") {
            colorized_diff.push_str(&format!("{}\n", line.red()));
        } else if line.starts_with("@@") {
            colorized_diff.push_str(&format!("{}\n", line.cyan()));
        } else {
            colorized_diff.push_str(&format!("{}\n", line));
        }
    }
    colorized_diff
}

/// Gets user input bypassing stdin if it's piped
fn get_user_input(prompt_text: &str) -> Result<String> {
    print!("{}", prompt_text);
    io::stdout().flush()?;

    if io::stdin().is_terminal() {
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        return Ok(input.trim().to_string());
    }

    // Direct TTY reads for piped stdin contexts
    #[cfg(unix)]
    {
        let file = File::open("/dev/tty").context("Could not open terminal for input")?;
        let mut reader = BufReader::new(file);
        let mut input = String::new();
        reader.read_line(&mut input)?;
        Ok(input.trim().to_string())
    }
    #[cfg(windows)]
    {
        let file = File::open("CONIN$").context("Could not open terminal for input")?;
        let mut reader = BufReader::new(file);
        let mut input = String::new();
        reader.read_line(&mut input)?;
        Ok(input.trim().to_string())
    }
}

// --- Core API Interaction ---

async fn generate_with_tracking(
    client: &reqwest::Client,
    token: &str,
    project: &str,
    model: &str,
    payload: Value,
    description: &str,
) -> Result<Value> {
    let count = API_CALL_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
    let log_msg = format!("Call #{} | {}", count, description);
    
    info!("{}", log_msg);
    println!("\n[API Tracker] {}...", log_msg);

    let url = format!(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/{}/locations/us-central1/publishers/google/models/{}:generateContent",
        project, model
    );

    let res = client
        .post(&url)
        .bearer_auth(token)
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;

    let response_json: Value = res.json().await?;
    Ok(response_json)
}

// --- File Handling and Parsing ---

fn extract_and_save_files(ai_response_text: &str, output_dir: &str, auto_save: bool) -> Result<()> {
    let file_pattern = FILE_PATTERN.get_or_init(|| {
        Regex::new(r"(?s)<file name=\x22([^\x22]+)\x22(?:\s+action=\x22([^\x22]+)\x22)?>\s*(.*?)\s*</file>").unwrap()
    });

    let mut memory_operations = Vec::new();

    // Phase 1: Apply all changes in-memory first
    for cap in file_pattern.captures_iter(ai_response_text) {
        let filename = cap.get(1).map_or("", |m| m.as_str());
        let action = cap.get(2).map_or("write", |m| m.as_str()).to_lowercase();
        let content = cap.get(3).map_or("", |m| m.as_str());
        let file_path = Path::new(output_dir).join(filename);

        if action == "edit" {
            if !file_path.exists() {
                println!(" -> ERROR: Cannot edit '{}' because it does not exist in {}.", filename, output_dir);
                continue;
            }

            let original_text = fs::read_to_string(&file_path)?;
            let mut new_text = original_text.clone();

            let edit_pattern = EDIT_PATTERN.get_or_init(|| {
                Regex::new(r"(?s)<search>\n?(.*?)\n?</search>\s*<replace>\n?(.*?)\n?</replace>").unwrap()
            });

            let edits: Vec<_> = edit_pattern.captures_iter(content).collect();
            let mut success_count = 0;

            for edit_cap in &edits {
                let search_text = edit_cap.get(1).map_or("", |m| m.as_str());
                let replace_text = edit_cap.get(2).map_or("", |m| m.as_str());

                if new_text.contains(search_text) {
                    new_text = new_text.replace(search_text, replace_text);
                    success_count += 1;
                } else {
                    println!("\n -> \x1b[93mWARNING: Could not find matching <search> block in {}. Skipping this chunk.\x1b[0m", filename);
                    let snippet = search_text.chars().take(60).collect::<String>();
                    println!("    [Looked for]: {}...", snippet.trim());
                }
            }

            if success_count > 0 {
                let diff = generate_colorized_diff(&original_text, &new_text, filename);
                memory_operations.push(MemoryOp {
                    filename: filename.to_string(),
                    path: file_path,
                    action: "edit".to_string(),
                    new_text,
                    diff,
                    msg: format!("Applied {}/{} edits", success_count, edits.len()),
                });
            }
        } else {
            // Write action
            let code_pattern = CODE_PATTERN.get_or_init(|| Regex::new(r"(?s)```[^\n]*\n(.*?)```").unwrap());
            
            let new_text = if let Some(matched) = code_pattern.captures(content) {
                matched.get(1).map_or("", |m| m.as_str()).to_string()
            } else {
                content.trim().to_string()
            };

            let original_text = if file_path.exists() {
                fs::read_to_string(&file_path).unwrap_or_default()
            } else {
                String::new()
            };

            let diff = generate_colorized_diff(&original_text, &new_text, filename);
            memory_operations.push(MemoryOp {
                filename: filename.to_string(),
                path: file_path,
                action: "write".to_string(),
                new_text,
                diff,
                msg: "Wrote full file".to_string(),
            });
        }
    }

    if memory_operations.is_empty() {
        println!("No successful file operations to apply.");
        return Ok(());
    }

    println!("\nPrepared {} file(s) for modification.", memory_operations.len());

    // Phase 2: Interactive Prompt (with diffs)
    if !auto_save {
        loop {
            let choice = get_user_input("\nOptions: [p]review diffs, [s]ave files, [a]bort: ")?;
            let choice = choice.to_lowercase();

            match choice.as_str() {
                "p" => {
                    for op in &memory_operations {
                        println!("\n{}", "=".repeat(60));
                        println!("FILE: {} ({})", op.filename, op.action.to_uppercase());
                        println!("{}", "=".repeat(60));
                        println!("{}", op.diff);
                    }
                }
                "s" => break,
                "a" => {
                    println!("Aborted saving these files.");
                    return Ok(());
                }
                _ => println!("Invalid choice. Please enter 'p', 's', or 'a'."),
            }
        }
    }

    // Phase 3: Write to Disk
    println!("\nApplying changes to './{}'...", output_dir);
    for op in memory_operations {
        if let Some(parent) = op.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&op.path)?;
        file.write_all(op.new_text.as_bytes())?;
        println!(" -> {}: {}", op.msg, op.path.display());
    }

    Ok(())
}

async fn generate_iteratively(
    client: &reqwest::Client,
    token: &str,
    project: &str,
    prompt: &str,
    output_dir: &str,
    auto_save: bool,
) -> Result<()> {
    // Schema definition for Vertex AI Structured Output
    let schema_json = json!({
        "type": "OBJECT",
        "properties": {
            "files": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "filepath": {"type": "STRING"},
                        "purpose": {"type": "STRING"},
                        "action": {
                            "type": "STRING", 
                            "description": "Must be 'create' for new files, 'edit' for modifying, or 'skip'."
                        }
                    },
                    "required": ["filepath", "purpose", "action"]
                }
            }
        },
        "required": ["files"]
    });

    let arch_payload = json!({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{"text": "You are a senior software architect. Given a project request and context, plan the necessary file modifications. For each file, provide the filepath, purpose, and the 'action' required ('create', 'edit', or 'skip'). CRITICAL: Use 'skip' for any existing files in the context that do NOT need modifications. NEVER include lock files or dependency folders."}]
        },
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": schema_json
        }
    });

    let arch_response = generate_with_tracking(
        client, token, project, "gemini-2.5-flash", arch_payload, "Phase 1: Planning Project Architecture"
    ).await?;

    let text_out = arch_response["candidates"][0]["content"]["parts"][0]["text"].as_str().unwrap_or("{}");
    
    let architecture: ProjectArchitecture = match serde_json::from_str(text_out) {
        Ok(a) => a,
        Err(e) => bail!("Failed to parse architecture JSON from model: {}\nRaw: {}", e, text_out),
    };

    println!("[Phase 1 Complete] Planned {} files.\n", architecture.files.len());

    let mut accumulated_context = String::new();

    println!("[Phase 2] Executing File Changes...");
    
    for file_plan in architecture.files {
        if file_plan.action.to_lowercase() == "skip" {
            println!("  - Skipping '{}' (No changes required)", file_plan.filepath);
            continue;
        }

        let mut file_prompt = format!(
            "Overall Project Context and Instructions:\n{}\n\nTask: Execute changes for the file: '{}'.\nPurpose of this file: {}\nAction Required: {}\n\nReminder: If Action is 'edit', ONLY output <search> and <replace> blocks inside <file action=\"edit\">.",
            prompt, file_plan.filepath, file_plan.purpose, file_plan.action
        );

        if !accumulated_context.is_empty() {
            file_prompt.push_str(&format!("\nCode already generated/modified in this session:\n{}\n", accumulated_context));
        }

        let file_payload = json!({
            "contents": [{"role": "user", "parts": [{"text": file_prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192
            }
        });

        let file_desc = format!("Phase 2: Generating {} ({})", file_plan.filepath, file_plan.action);
        let file_resp = match generate_with_tracking(client, token, project, "gemini-2.5-flash", file_payload, &file_desc).await {
            Ok(r) => r,
            Err(e) => {
                println!("    Error processing {}: {}", file_plan.filepath, e);
                continue;
            }
        };

        let finish_reason = file_resp["candidates"][0]["finishReason"].as_str().unwrap_or("UNKNOWN");
        if finish_reason != "STOP" {
            println!("    WARNING: {} generation cut off! Reason: {}", file_plan.filepath, finish_reason);
        }

        let gen_text = file_resp["candidates"][0]["content"]["parts"][0]["text"].as_str().unwrap_or("");
        
        if let Err(e) = extract_and_save_files(gen_text, output_dir, auto_save) {
            println!("    Error applying changes to {}: {}", file_plan.filepath, e);
        }

        accumulated_context.push_str(&format!("\n--- Actions performed on {} ---\n{}\n", file_plan.filepath, gen_text));
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse arguments, overriding project ID from env if missing in CLI
    let mut args = Args::parse();
    if args.project == "coder-470" {
        if let Ok(env_proj) = env::var("GOOGLE_CLOUD_PROJECT") {
            args.project = env_proj;
        }
    }

    // Set up file logger matching the Python format
    WriteLogger::init(
        LevelFilter::Info,
        ConfigBuilder::new()
            .set_time_format_custom(format_description!("[year]-[month]-[day] [hour]:[minute]:[second]"))
            .build(),
        OpenOptions::new().append(true).create(true).open("coder_api_usage.log")?,
    )?;

    // Handle Input Context
    let mut context_data = String::new();
    
    if let Some(ctx_path) = &args.context_file {
        context_data = fs::read_to_string(ctx_path)
            .with_context(|| format!("Error reading context file: {}", ctx_path))?;
        println!("Loaded {} bytes of context from {}", context_data.len(), ctx_path);
    } else if !io::stdin().is_terminal() {
        io::stdin().read_to_string(&mut context_data)?;
        println!("Loaded {} bytes of context from standard input.", context_data.len());
    }

    let final_prompt = if !context_data.is_empty() {
        format!("Here is the context of the existing codebase:\n\n{}\n\nTask Instructions:\n{}", context_data, args.prompt)
    } else {
        args.prompt.clone()
    };

    // Authenticate and construct HTTP client
    let token = get_gcloud_token()?;
    let client = reqwest::Client::new();

    println!("Generating code... (Target: ./{}/)", args.outdir);

    if args.iterative {
        generate_iteratively(&client, &token, &args.project, &final_prompt, &args.outdir, args.yes).await?;
    } else {
        let payload = json!({
            "contents": [{"role": "user", "parts": [{"text": final_prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192
            }
        });

        let response = generate_with_tracking(
            &client,
            &token,
            &args.project,
            "gemini-2.5-flash",
            payload,
            "Single-Shot File Generation",
        ).await.context("API Error: Request to Vertex AI failed.")?;

        let finish_reason = response["candidates"][0]["finishReason"].as_str().unwrap_or("UNKNOWN");
        if finish_reason != "STOP" {
            println!("WARNING: Generation did not finish normally! Reason: {}", finish_reason);
            println!("Attempting to parse whatever was generated so far...");
        }

        let gen_text = response["candidates"][0]["content"]["parts"][0]["text"].as_str().unwrap_or("");
        extract_and_save_files(gen_text, &args.outdir, args.yes)?;
    }

    Ok(())
}

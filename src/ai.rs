use crate::AppResult;
use crate::git::diff::{get_diff_summary, get_file, get_patch};
use async_openai::Client;
use async_openai::config::{Config, OpenAIConfig};
use async_openai::types::chat::{ReasoningEffort, ResponseFormatJsonSchema};
use async_openai::types::evals::InputTextContent;
use async_openai::types::responses::{
    CreateResponseArgs, FunctionCallOutput, FunctionCallOutputItemParam, FunctionTool,
    InputContent, InputItem, InputMessage, InputParam, InputRole, Item, MessageItem, OutputItem,
    OutputMessageContent, Reasoning, RefusalContent, ResponseTextParam,
    TextResponseFormatConfiguration, Tool, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use git2::{Diff, Repository};
use log::{debug, error, trace, warn};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

pub fn get_lm_studio_client<S: AsRef<str>>(server: S, port: u16) -> Client<Box<dyn Config>> {
    let config = Box::new(OpenAIConfig::default().with_api_base(format!(
        "http://{}:{}/v1",
        server.as_ref(),
        port
    ))) as Box<dyn Config>;
    let client: Client<Box<dyn Config>> = Client::with_config(config);
    client
}

static COMMIT_MESSAGE_PROMPT: &str = r##"
You generate commit messages from structured change metadata. 
Your only job is to read the JSON describing the repository changes, optionally fetch more context using tools, and output a commit message as a JSON object of the form:

{
  "summary": "...",   // required, < 72 characters
  "body": "..."       // optional, omitted if unnecessary
}

You must not output anything except this JSON object unless calling a tool.

====================================
AVAILABLE TOOLS
====================================
You may call these tools when you need more context:

- get_file(path, start_line?, end_line?)
- get_patch(path, start_line?, end_line?)

Use tools sparingly and only to clarify ambiguous or incomplete information in the input.

====================================
INPUT FORMAT
====================================
You will receive a JSON object with the following structure:

{
  "unmodified": [paths...],
  "added":      [{"path":..., "patch":...}, ...],
  "deleted":    [paths...],
  "modified":   [{"path":..., "patch":...}, ...],
  "renamed":    [{"from":..., "to":...}, ...],
  "copied":     [{"from":..., "to":...}, ...],
  "untracked":  [{"path":..., "patch":...}, ...],
  "typechange": [paths...],
  "unreadable": [paths...],
  "conflicted": [paths...]
}

Any field may be empty.

====================================
OUTPUT REQUIREMENTS
====================================

Your final output must be a JSON object:

{
  "summary": "<short descriptive subject line>",
  "body": "<optional longer explanation>"
}

Rules for the summary:
- Must be fewer than 72 characters.
- Must accurately reflect what changed.
- Must NOT reference the JSON input or tools.

Rules for the body:
- Include only if useful.
- Summarize conceptual changes, not line-by-line diffs.
- Group related changes (e.g., "added module X", "updated config Y").
- Mention motivations ONLY when explicitly obvious in the patch or filenames.
- Omit the `body` field entirely if unnecessary.

====================================
HOW TO INTERPRET CHANGES
====================================

ADDED FILES:
- Summarize purpose using filename + content.
- If content is unclear, call get_file() to examine it.

DELETED FILES:
- Briefly describe what the removed file previously contained (if visible).

MODIFIED FILES:
- Summarize high-level changes, not diff hunks.
- For ambiguous patches, use get_patch() for more context.

RENAMED FILES:
- Indicate what moved from → to.
- Interpret meaning only if obvious (e.g., moving into `src/`).

COPIED FILES:
- Same as rename, but note it’s a copy.

UNTRACKED FILES:
- Treat as newly added unless diff suggests otherwise.

TYPE CHANGES:
- Summarize the type change (e.g., made executable).

UNREADABLE FILES:
- State that the file was changed or removed, without content analysis.

CONFLICTS:
- Mention unresolved conflicts if present.

====================================
STYLE
====================================

- Tone: technical, concise, straightforward.
- Do not editorialize.
- Do not guess intentions.
- Do not mention diffs, patches, or the JSON structure.
- Do not mention the tools or that you used them.
- Keep the writing useful to another engineer reading the commit later.

====================================
WHEN TO USE TOOLS
====================================

Use `get_file` or `get_patch` when:
- A new file’s purpose is unclear from partial content.
- A diff is too small or ambiguous to understand.
- A rename/copy needs validation of contents.
- A modified file shows only trivial context and requires more lines.

Request only the minimal information required.

====================================
ABSOLUTE PROHIBITIONS
====================================

- Do NOT hallucinate intent.
- Do NOT fabricate behavior not present in the change data.
- Do NOT output any text outside the final JSON object.
- Do NOT describe your reasoning process.
- Do NOT mention the diff or JSON schema.

====================================
EXAMPLE OF CORRECT OUTPUT SHAPE
====================================

{
  "summary": "Add initial PLL demodulator prototype",
  "body": "Introduce pll.rs with basic demodulation logic and update receiver pipeline to call it."
}
"##;

/// # commit_message
/// Generate a commit message based on the provided diff summary
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CommitMessage {
    /// Short summary of the commit
    pub summary: String,
    /// Optional detailed body of the commit message
    pub body: Option<String>,
}

impl Display for CommitMessage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary)?;
        if let Some(body) = &self.body {
            write!(f, "\n\n{}", body)?;
        }
        Ok(())
    }
}

/// # get_file
/// Retrieve a file or a segment of a file from the repository.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetFile {
    /// Path to the file
    pub path: PathBuf,
    /// Optional starting line of the file (for partial retrieval)
    pub start_line: Option<usize>,
    /// Optional ending line of the file (for partial retrieval)
    pub end_line: Option<usize>,
}

/// # get_patch
/// Retrieve a patch or a segment of a patch from the repository.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetPatch {
    /// Path to the file the patch applies to
    pub path: PathBuf,
    /// Optional starting line of the patch (for partial retrieval)
    pub start_line: Option<usize>,
    /// Optional ending line of the patch (for partial retrieval)
    pub end_line: Option<usize>,
}

pub async fn generate_commit_message<'c, 'd, C: Config>(
    client: &'c Client<C>,
    diff: &Diff<'d>,
    repo: &Repository,
) -> AppResult<CommitMessage> {
    // Kick off first turn.
    let mut input_items: Vec<InputItem> = vec![InputItem::Item(Item::Message(MessageItem::Input(
        InputMessage {
            content: vec![InputContent::InputText(InputTextContent {
                text: serde_json::to_string_pretty(&get_diff_summary(
                    repo.path().parent().unwrap(),
                    diff,
                )?)?,
            })],
            role: InputRole::User,
            status: None,
        },
    )))];

    loop {
        let request = CreateResponseArgs::default()
            .model("openai/gpt-oss-20b")
            .input(InputParam::Items(input_items.clone()))
            .background(false)
            .instructions(COMMIT_MESSAGE_PROMPT)
            .parallel_tool_calls(false)
            .reasoning(Reasoning {
                effort: Some(ReasoningEffort::Medium),
                summary: None,
            })
            .store(true)
            .stream(false)
            .temperature(0.05)
            .text(ResponseTextParam {
                format: TextResponseFormatConfiguration::JsonSchema(ResponseFormatJsonSchema {
                    description: Some("Commit message with summary and optional body".to_string()),
                    schema: Some(schema_for!(CommitMessage).as_value().to_owned()),
                    name: "commit_message".to_string(),
                    strict: Some(true),
                }),
                verbosity: None,
            })
            .tool_choice(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))
            .tools(vec![
                Tool::Function(FunctionTool {
                    name: "get_patch".to_string(),
                    description: Some("Retrieve a patch for a file".to_string()),
                    parameters: Some(schema_for!(GetPatch).as_value().to_owned()),
                    strict: Some(true),
                }),
                Tool::Function(FunctionTool {
                    name: "get_file".to_string(),
                    description: Some("Retrieve the contents of a file".to_string()),
                    parameters: Some(schema_for!(GetFile).as_value().to_owned()),
                    strict: Some(true),
                }),
            ])
            .top_logprobs(0)
            .top_p(0.1)
            .truncation(Truncation::Disabled)
            .build()?;

        let response = client.responses().create(request).await?;

        let function_calls: Vec<_> = response
            .output
            .iter()
            .filter_map(|item| {
                if let OutputItem::FunctionCall(fc) = item {
                    Some(fc.clone())
                } else {
                    None
                }
            })
            .collect();

        if function_calls.is_empty() {
            let mut response_content = String::new();
            for out in &response.output {
                if let OutputItem::Message(msg) = out {
                    for content in &msg.content {
                        match content {
                            OutputMessageContent::OutputText(text) => {
                                response_content.push_str(&text.text)
                            }
                            OutputMessageContent::Refusal(RefusalContent { refusal }) => {
                                error!("AI refused prompt: {}", refusal);
                            }
                        }
                    }
                }
            }
            return Ok(serde_json::from_str(&response_content)?);
        }

        for call in function_calls {
            let function_name = call.name.as_str();
            let arguments = &call.arguments;
            match function_name {
                "get_file" => {
                    debug!("Processing tool call: get_file with args {}", arguments);
                    let args: GetFile = serde_json::from_str(arguments)?;
                    let content = get_file(repo, diff, &args.path, args.start_line, args.end_line)
                        .unwrap_or_default();
                    trace!("Retrieved file content: {}", content);
                    input_items.push(InputItem::Item(Item::FunctionCall(call.clone())));
                    input_items.push(InputItem::Item(Item::FunctionCallOutput(
                        FunctionCallOutputItemParam {
                            call_id: call.call_id,
                            output: FunctionCallOutput::Text(content),
                            id: None,
                            status: None,
                        },
                    )));
                }
                "get_patch" => {
                    debug!("Processing tool call: get_patch with args {}", arguments);
                    let args: GetPatch = serde_json::from_str(arguments)?;
                    let content = get_patch(
                        diff,
                        &args.path,
                        args.start_line.map(|n| n as u32),
                        args.end_line.map(|n| n as u32),
                    )
                    .unwrap_or_default();
                    trace!("Retrieved patch content: {}", content);

                    input_items.push(InputItem::Item(Item::FunctionCall(call.clone())));
                    input_items.push(InputItem::Item(Item::FunctionCallOutput(
                        FunctionCallOutputItemParam {
                            call_id: call.call_id,
                            output: FunctionCallOutput::Text(content),
                            id: None,
                            status: None,
                        },
                    )));
                }
                _ => warn!("Unknown tool call: {}", function_name),
            }
        }
    }
}

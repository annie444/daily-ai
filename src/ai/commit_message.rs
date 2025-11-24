use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::evals::InputTextContent;
use async_openai::types::responses::{
    CreateResponse, FunctionCallOutput, FunctionCallOutputItemParam, FunctionTool,
    FunctionToolCall, InputContent, InputItem, InputMessage, InputParam, InputRole, Item,
    MessageItem, OutputItem, OutputMessageContent, Reasoning, ReasoningEffort, RefusalContent,
    ResponseFormatJsonSchema, ResponseTextParam, TextResponseFormatConfiguration, Tool,
    ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use git2::{Diff, Repository};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, trace, warn};

use super::ResponseCleaner;
use crate::AppResult;
use crate::git::diff::{get_diff_summary, get_file, get_patch};

static COMMIT_MESSAGE_PROMPT: &str = std::include_str!("commit_message_prompt.txt");

/// Commit message output from the model: summary plus optional body.
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

/// Generate a commit message using the model, optionally calling back into file/patch tools.
#[tracing::instrument(
    name = "Generating a commit message with LLM",
    level = "debug",
    skip(client, diff, repo)
)]
pub async fn generate_commit_message<'c, 'd, C: Config>(
    client: &'c Client<C>,
    diff: &Diff<'d>,
    repo: &Repository,
) -> AppResult<CommitMessage> {
    // Kick off first turn with diff summary and commit prompt.
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
    input_items.push(InputItem::Item(Item::Message(MessageItem::Input(
        InputMessage {
            content: vec![InputContent::InputText(InputTextContent {
                text: COMMIT_MESSAGE_PROMPT.to_string(),
            })],
            role: InputRole::System,
            status: None,
        },
    ))));
    let mut previous_response_id: Option<String> = None;
    let tools = vec![
        Tool::Function(FunctionTool {
            name: "get_patch".to_string(),
            description: Some("Retrieve a patch for a file".to_string()),
            parameters: Some(schema_for!(GetPatch).as_value().to_owned()),
            strict: None,
        }),
        Tool::Function(FunctionTool {
            name: "get_file".to_string(),
            description: Some("Retrieve the contents of a file".to_string()),
            parameters: Some(schema_for!(GetFile).as_value().to_owned()),
            strict: None,
        }),
    ];

    loop {
        let request = CreateResponse {
            model: Some("openai/gpt-oss-20b".to_string()),
            input: InputParam::Items(input_items.clone()),
            background: Some(false),
            instructions: Some(COMMIT_MESSAGE_PROMPT.to_string()),
            parallel_tool_calls: Some(false),
            reasoning: Some(Reasoning {
                effort: Some(ReasoningEffort::Medium),
                summary: None,
            }),
            store: Some(true),
            stream: Some(false),
            temperature: Some(0.05),
            text: Some(ResponseTextParam {
                format: TextResponseFormatConfiguration::JsonSchema(ResponseFormatJsonSchema {
                    description: Some("Commit message with summary and optional body".to_string()),
                    schema: Some(schema_for!(CommitMessage).as_value().to_owned()),
                    name: "commit_message".to_string(),
                    strict: None,
                }),
                verbosity: None,
            }),
            tool_choice: Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
            tools: Some(tools.clone()),
            top_logprobs: Some(0),
            top_p: Some(0.1),
            truncation: Some(Truncation::Disabled),
            previous_response_id: previous_response_id.clone(),
            ..Default::default()
        };

        let response = client.responses().create(request).await?;
        debug!("AI Response: {:?}", response);
        previous_response_id = Some(response.id.clone());

        let function_calls: Vec<FunctionToolCall> = response
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
            trace!("Raw response content: {}", response_content);
            let response_content = ResponseCleaner::new(&response_content).clean();
            trace!("Cleaned response content: {}", response_content);
            let jd = &mut serde_json::Deserializer::from_str(&response_content);
            match serde_path_to_error::deserialize(jd) {
                Ok(cm) => return Ok(cm),
                Err(e) => {
                    error!("Failed to deserialize commit message: {}", e);
                    error!("Response content was: {}", response_content);
                    error!("Failed to parse JSON at path: {}", e.path());
                    return Err(e.into_inner().into());
                }
            };
        }

        // Handle each tool call in order and feed results back into the conversation.
        for call in function_calls {
            let function_name = call.name.as_str();
            let arguments = &call.arguments;
            match function_name {
                "get_file" => {
                    debug!("Processing tool call: get_file with args {}", arguments);
                    let jd = &mut serde_json::Deserializer::from_str(arguments);
                    let args: GetFile = match serde_path_to_error::deserialize(jd) {
                        Ok(args) => {
                            trace!("Parsed `get_file` arguments: {:?}", args);
                            args
                        }
                        Err(e) => {
                            error!("Failed to parse `get_file` arguments: {}", e);
                            error!("Failed to parse JSON at path: {}", e.path());
                            return Err(e.into_inner().into());
                        }
                    };
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
                    let jd = &mut serde_json::Deserializer::from_str(arguments);
                    let args: GetPatch = match serde_path_to_error::deserialize(jd) {
                        Ok(args) => {
                            trace!("Parsed `get_patch` arguments: {:?}", args);
                            args
                        }
                        Err(e) => {
                            error!("Failed to parse `get_patch` arguments: {}", e);
                            error!("Failed to parse JSON at path: {}", e.path());
                            return Err(e.into_inner().into());
                        }
                    };
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

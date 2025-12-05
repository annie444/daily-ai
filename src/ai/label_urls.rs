use std::fmt::{Display, Formatter};

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::evals::InputTextContent;
use async_openai::types::responses::{
    CreateResponse, FunctionToolCall, InputContent, InputItem, InputMessage, InputParam, InputRole,
    Item, MessageItem, OutputItem, OutputMessageContent, Reasoning, ReasoningEffort,
    RefusalContent, ResponseFormatJsonSchema, ResponseTextParam, TextResponseFormatConfiguration,
    Tool, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, trace};

use super::ResponseCleaner;
use super::tools::{CustomTool, FetchUrl, unknown_tool};
use crate::AppResult;
use crate::safari::SafariHistoryItem;

static LABEL_URLS_PROMPT: &str = std::include_str!("label_urls_prompt.txt");

/// Label returned by the model for a cluster of URLs.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UrlLabel {
    /// Short label for the group of URLs.
    pub label: String,
}

impl Display for UrlLabel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// Label a cluster of URLs using the model; may call back into the `fetch_url` tool.
#[tracing::instrument(
    name = "Generating a label for a group of URLs",
    level = "debug",
    skip(client, urls)
)]
pub async fn label_url_cluster<C: Config>(
    client: &Client<C>,
    urls: &[SafariHistoryItem],
) -> AppResult<UrlLabel> {
    // Kick off first turn with the URL list and system prompt.
    let mut input_items: Vec<InputItem> = vec![InputItem::Item(Item::Message(MessageItem::Input(
        InputMessage {
            content: vec![InputContent::InputText(InputTextContent {
                text: serde_json::to_string_pretty(&urls)?,
            })],
            role: InputRole::User,
            status: None,
        },
    )))];
    input_items.push(InputItem::Item(Item::Message(MessageItem::Input(
        InputMessage {
            content: vec![InputContent::InputText(InputTextContent {
                text: LABEL_URLS_PROMPT.to_string(),
            })],
            role: InputRole::System,
            status: None,
        },
    ))));
    let tools = vec![Tool::Function(FetchUrl::definition())];
    let mut previous_response_id: Option<String> = None;

    loop {
        let request = CreateResponse {
            model: Some("openai/gpt-oss-20b".to_string()),
            input: InputParam::Items(input_items.clone()),
            background: Some(false),
            instructions: Some(LABEL_URLS_PROMPT.to_string()),
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
                    description: Some("Label for the given list of URLs".to_string()),
                    schema: Some(schema_for!(UrlLabel).as_value().to_owned()),
                    name: "url_label".to_string(),
                    strict: None,
                }),
                verbosity: None,
            }),
            tool_choice: Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
            tools: Some(tools.clone()),
            top_logprobs: Some(0),
            top_p: Some(0.1),
            truncation: Some(Truncation::Disabled),
            previous_response_id,
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
                    error!("Failed to deserialize url label message: {}", e);
                    error!("Response content was: {}", response_content);
                    error!("Failed to parse JSON at path: {}", e.path());
                    return Err(e.into_inner().into());
                }
            };
        }

        // Handle each tool call in sequence and feed results back to the model.
        for call in function_calls {
            match call.name.as_str() {
                name if name == FetchUrl::NAME => {
                    input_items.extend(FetchUrl::process(call, &()).await);
                }
                _ => input_items.extend(unknown_tool(call)),
            };
        }
    }
}

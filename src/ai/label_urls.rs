use crate::AppResult;
use crate::safari::SafariHistoryItem;
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
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use tracing::{debug, error, trace, warn};

static LABEL_URLS_PROMPT: &str = std::include_str!("label_urls_prompt.txt");

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

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FetchUrlParams {
    /// URL to fetch.
    pub url: String,
    /// Optional starting line number to fetch from.
    pub starting_line: Option<usize>,
    /// Optional maximum number of lines to fetch.
    pub max_lines: Option<usize>,
}

#[tracing::instrument(name = "Label URL groups", level = "debug", skip(client, urls))]
pub async fn label_url_cluster<C: Config>(
    client: &Client<C>,
    urls: &[SafariHistoryItem],
) -> AppResult<UrlLabel> {
    // Kick off first turn.
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
    let tools = vec![Tool::Function(FunctionTool {
        name: "fetch_url".to_string(),
        description: Some("Fetches the content of a URL.".to_string()),
        parameters: Some(schema_for!(FetchUrlParams).as_value().to_owned()),
        strict: None,
    })];
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
            // Attempt to deserialize the response content into proper JSON by filtering out any extraneous text.
            // This is a bit of a hack, but it helps with models that may include extra text around the JSON.
            // We track whether we're inside quotes or brackets to ensure we capture the full JSON object.
            // This assumes the JSON object is the outermost structure in the response.
            // This may need to be adjusted for more complex scenarios.
            // For now, we assume the response is a single JSON object.
            //
            // Example response:
            //
            // !{
            //   !"label": "Tech News and Articles"
            // }
            // We want to extract:
            // {
            //   "label": "Tech News and Articles"
            // }
            //
            // We do this by iterating through the characters and tracking our position.
            // When we encounter a '{', we start capturing until we find the matching '}'.
            // We also need to handle quotes to avoid prematurely ending the capture.
            // This is a simple state machine approach.
            trace!("Raw response content: {}", response_content);
            let mut in_quote = false;
            let mut in_bracket = false;
            let response_content = response_content
                .chars()
                .filter(|c| {
                    if c == &'"' && in_bracket {
                        in_quote = !in_quote;
                        true
                    } else if c == &'{' && !in_quote {
                        in_bracket = true;
                        true
                    } else if c == &'}' && !in_quote {
                        in_bracket = false;
                        true
                    } else {
                        (c == &':' && in_bracket) || in_quote
                    }
                })
                .collect::<String>();
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
        for call in function_calls {
            let function_name = call.name.as_str();
            let arguments = &call.arguments;
            match function_name {
                "fetch_url" => {
                    debug!("Processing tool call: fetch_url with args {}", arguments);
                    let jd = &mut serde_json::Deserializer::from_str(arguments);
                    let args: FetchUrlParams = match serde_path_to_error::deserialize(jd) {
                        Ok(args) => {
                            trace!("Parsed `fetch_url` arguments: {:?}", args);
                            args
                        }
                        Err(e) => {
                            error!("Failed to parse `get_file` arguments: {}", e);
                            error!("Failed to parse JSON at path: {}", e.path());
                            return Err(e.into_inner().into());
                        }
                    };
                    let content = get_url(&args.url, args.starting_line, args.max_lines).await?;
                    trace!(
                        "Retrieved url content: {}",
                        content[..std::cmp::min(100, content.len())].to_string()
                    );
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

#[tracing::instrument(name = "Process MCP Tool Calls", level = "trace")]
async fn get_url(
    url: &str,
    starting_line: Option<usize>,
    max_lines: Option<usize>,
) -> AppResult<String> {
    let resp = reqwest::get(url).await?;
    let ct = if let Some(content) = resp.headers().get("content-type") {
        content.to_str().unwrap_or_default().to_string()
    } else {
        "text/plain".to_string()
    };
    let mut body = match (starting_line, max_lines) {
        (Some(start), Some(max)) => resp
            .text()
            .await?
            .lines()
            .collect::<Vec<&str>>()
            .iter()
            .skip(start)
            .take(max)
            .cloned()
            .collect::<Vec<&str>>()
            .join("\n"),
        (Some(start), None) => resp
            .text()
            .await?
            .lines()
            .collect::<Vec<&str>>()
            .iter()
            .skip(start)
            .cloned()
            .collect::<Vec<&str>>()
            .join("\n"),
        (None, Some(max)) => resp
            .text()
            .await?
            .lines()
            .collect::<Vec<&str>>()
            .iter()
            .take(max)
            .cloned()
            .collect::<Vec<&str>>()
            .join("\n"),
        (None, None) => resp.text().await?,
    };
    if ct.to_lowercase().contains("text/html") {
        // Simple HTML stripping.
        body = html2md::parse_html(&body);
    }
    Ok(body)
}

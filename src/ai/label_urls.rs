use crate::AppResult;
use crate::safari::SafariHistoryItem;
use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::evals::InputTextContent;
use async_openai::types::responses::{
    CreateResponse, InputContent, InputItem, InputMessage, InputParam, InputRole, Item,
    MessageItem, OutputItem, OutputMessageContent, Reasoning, ReasoningEffort, RefusalContent,
    ResponseFormatJsonSchema, ResponseTextParam, TextResponseFormatConfiguration,
    ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use tracing::{debug, error};

static LABEL_URLS_PROMPT: &str = std::include_str!("label_urls_prompt.txt");

/// # url_label
/// Generated label for a group of URLs.
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

#[tracing::instrument(level = "debug", skip(client, urls))]
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
        top_logprobs: Some(0),
        top_p: Some(0.1),
        truncation: Some(Truncation::Disabled),
        ..Default::default()
    };

    let response = client.responses().create(request).await?;
    debug!("AI Response: {:?}", response);

    let mut response_content = String::new();
    for out in &response.output {
        if let OutputItem::Message(msg) = out {
            for content in &msg.content {
                match content {
                    OutputMessageContent::OutputText(text) => response_content.push_str(&text.text),
                    OutputMessageContent::Refusal(RefusalContent { refusal }) => {
                        error!("AI refused prompt: {}", refusal);
                    }
                }
            }
        }
    }
    let jd = &mut serde_json::Deserializer::from_str(&response_content);
    match serde_path_to_error::deserialize(jd) {
        Ok(cm) => return Ok(cm),
        Err(e) => {
            error!("Failed to deserialize url label: {}", e);
            error!("Response content was: {}", response_content);
            error!("Failed to parse JSON at path: {}", e.path());
            return Err(e.into_inner().into());
        }
    };
}

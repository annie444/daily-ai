use async_openai::types::responses::{
    FunctionCallOutput, FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, InputItem,
    Item, OutputStatus,
};
use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde_json::Value;
use tracing::{error, trace, warn};

use super::ResponseCleaner;
use crate::AppResult;

pub trait CustomTool:
    Serialize + for<'de> serde::Deserialize<'de> + JsonSchema + Send + Sync
{
    type Context<'a>: ?Sized;
    const NAME: &'static str;
    const DESCRIPTION: &'static str;

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String);

    fn parameters() -> Value {
        schema_for!(Self).as_value().to_owned()
    }

    fn definition() -> FunctionTool {
        FunctionTool {
            name: Self::NAME.to_string(),
            parameters: Some(Self::parameters()),
            description: Some(Self::DESCRIPTION.to_string()),
            strict: None,
        }
    }

    fn parse_output(output: &str) -> AppResult<Self> {
        trace!("Raw response content: {output}");
        let output = ResponseCleaner::new(output).clean();
        trace!("Cleaned response content: {output}");
        let jd = &mut serde_json::Deserializer::from_str(&output);
        match serde_path_to_error::deserialize(jd) {
            Ok(cm) => Ok(cm),
            Err(e) => {
                error!("Failed to deserialize args for {} message: {e}", Self::NAME);
                error!("Response content was: {output}");
                error!("Failed to parse JSON at path: {}", e.path());
                Err(e.into_inner().into())
            }
        }
    }

    async fn process(call: FunctionToolCall, context: &Self::Context<'_>) -> Vec<InputItem> {
        let mut items = vec![InputItem::Item(Item::FunctionCall(call.clone()))];
        let output = match Self::parse_output(&call.arguments) {
            Ok(parsed) => parsed,
            Err(e) => {
                let error_msg = format!("Error parsing output: {e}");
                error!(error_msg);
                items.push(InputItem::Item(Item::FunctionCallOutput(
                    FunctionCallOutputItemParam {
                        call_id: call.call_id,
                        output: FunctionCallOutput::Text(error_msg),
                        id: None,
                        status: Some(OutputStatus::Incomplete),
                    },
                )));
                return items;
            }
        };
        let (status, response) = output.call(context).await;
        items.push(InputItem::Item(Item::FunctionCallOutput(
            FunctionCallOutputItemParam {
                call_id: call.call_id,
                output: FunctionCallOutput::Text(response),
                id: None,
                status: Some(status),
            },
        )));
        items
    }
}

pub fn arbitrary_tool_error(call: FunctionToolCall, msg: &str) -> Vec<InputItem> {
    warn!(msg);
    let mut items = vec![InputItem::Item(Item::FunctionCall(call.clone()))];
    items.push(InputItem::Item(Item::FunctionCallOutput(
        FunctionCallOutputItemParam {
            call_id: call.call_id,
            output: FunctionCallOutput::Text(msg.to_string()),
            id: None,
            status: Some(OutputStatus::Incomplete),
        },
    )));
    items
}

pub fn unknown_tool(call: FunctionToolCall) -> Vec<InputItem> {
    let error_msg = format!("Unknown tool call: {}", &call.name);
    arbitrary_tool_error(call, &error_msg)
}

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct FetchUrl {
    /// URL to fetch.
    pub url: String,
    /// Optional starting line number to fetch from.
    #[serde(default)]
    pub starting_line: Option<usize>,
    /// Optional maximum number of lines to fetch.
    #[serde(default)]
    pub max_lines: Option<usize>,
}

impl CustomTool for FetchUrl {
    type Context<'a> = ();
    const NAME: &'static str = "fetch_url";
    const DESCRIPTION: &'static str = "Fetches the content of a URL.";

    async fn call(&self, _context: &Self::Context<'_>) -> (OutputStatus, String) {
        let resp = match reqwest::get(&self.url).await {
            Ok(r) => r,
            Err(e) => {
                let error_msg = format!("Failed to fetch URL {}: {e}", self.url);
                error!(error_msg);
                return (OutputStatus::Incomplete, error_msg);
            }
        };
        let ct = if let Some(content) = resp.headers().get("content-type") {
            content.to_str().unwrap_or_default().to_string()
        } else {
            "text/plain".to_string()
        };
        let resp_text = match resp.text().await {
            Ok(t) => {
                if ct.to_lowercase().contains("text/html") {
                    html2md::parse_html(&t)
                } else {
                    t
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to read response text from URL {}: {e}", self.url);
                error!(error_msg);
                return (OutputStatus::Incomplete, error_msg);
            }
        };
        let resp_vec = resp_text.lines().collect::<Vec<&str>>();
        (
            OutputStatus::Completed,
            match (self.starting_line, self.max_lines) {
                (Some(start), Some(max)) => resp_vec
                    .iter()
                    .skip(start)
                    .take(max)
                    .cloned()
                    .collect::<Vec<&str>>()
                    .join("\n"),
                (Some(start), None) => resp_vec
                    .iter()
                    .skip(start)
                    .cloned()
                    .collect::<Vec<&str>>()
                    .join("\n"),
                (None, Some(max)) => resp_vec
                    .iter()
                    .take(max)
                    .cloned()
                    .collect::<Vec<&str>>()
                    .join("\n"),
                (None, None) => resp_text,
            },
        )
    }
}

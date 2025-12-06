pub mod commit;
pub mod fetch;
pub mod summary;

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

    fn parse_output<S>(output: S) -> AppResult<Self>
    where
        S: AsRef<str> + std::fmt::Display + std::fmt::Debug,
    {
        trace!("Raw response content: {output}");
        let output = ResponseCleaner::new().clean(output);
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

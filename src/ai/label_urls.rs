use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::responses::{FunctionToolCall, InputItem, Tool};
use daily_ai_include_zstd::include_zstd;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::agent::Agent;
use super::tools::ToolRegistry;
use super::tools::fetch::FetchUrl;
use super::tools::{CustomTool, unknown_tool};
use crate::safari::SafariHistoryItem;
use crate::{AppResult, impl_query};

static LABEL_URLS_PROMPT: &[u8] = include_zstd!("src/ai/prompts/label_urls_prompt.md");

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

impl_query!(UrlLabel, LABEL_URLS_PROMPT);

pub struct LabelUrlRegistry;

impl ToolRegistry for LabelUrlRegistry {
    type Context<'a> = ();

    fn definitions() -> Vec<Tool> {
        vec![Tool::Function(FetchUrl::definition())]
    }

    async fn execute<'c>(call: FunctionToolCall, context: &Self::Context<'c>) -> Vec<InputItem> {
        match call.name.as_str() {
            name if name == FetchUrl::name() => FetchUrl::process(call, context).await,
            _ => unknown_tool(call),
        }
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
    let initial_user_message = serde_json::to_string_pretty(&urls)?;
    let agent = Agent::new(None);

    agent
        .run::<_, (), LabelUrlRegistry, UrlLabel>(
            client,
            &(),
            &initial_user_message,
            &HashMap::new(),
        )
        .await
}

use async_openai::types::responses::OutputStatus;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::error;

use super::CustomTool;

/// # fetch_url
/// Fetch content from a specified URL, with options to limit the number of lines retrieved.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

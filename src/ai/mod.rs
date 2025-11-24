pub mod commit_message;
pub mod label_urls;

use async_openai::Client;
use async_openai::config::{Config, OpenAIConfig};

/// Build an async-openai client pointed at an LM Studio-compatible server.
#[tracing::instrument(name = "Connecting to local LLM server", level = "debug")]
pub fn get_lm_studio_client<S: AsRef<str> + std::fmt::Debug>(
    server: S,
    port: u16,
) -> Client<Box<dyn Config>> {
    let config = Box::new(OpenAIConfig::default().with_api_base(format!(
        "http://{}:{}/v1",
        server.as_ref(),
        port
    ))) as Box<dyn Config>;

    Client::with_config(config)
}

/// A utility to clean up responses from language models to extract valid JSON.
pub(super) struct ResponseCleaner<'cleaner> {
    /// The raw response string from the language model.
    response: &'cleaner str,
    /// State tracking for parsing braces (`{` or `}`)
    in_braces: bool,
    /// State tracking for parsing brackets (`[` or `]`)
    in_brackets: bool,
    /// State tracking for parsing quotes (`"`)
    in_quotes: bool,
    /// State tracking for escape characters (`\`)
    is_escaped: bool,
}

impl<'cleaner> ResponseCleaner<'cleaner> {
    /// Create a new ResponseCleaner instance.
    pub fn new(response: &'cleaner str) -> Self {
        Self {
            response,
            in_braces: false,
            in_brackets: false,
            in_quotes: false,
            is_escaped: false,
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
    pub fn clean(&mut self) -> String {
        self.response
            .chars()
            .filter(|c| match c {
                '\\' if !self.is_escaped && (self.in_braces || self.in_brackets) => {
                    self.is_escaped = true;
                    true
                }
                '"' if !self.is_escaped && (self.in_braces || self.in_brackets) => {
                    self.in_quotes = !self.in_quotes;
                    true
                }
                '"' if self.is_escaped => {
                    self.is_escaped = false;
                    true
                }
                '[' if !self.in_quotes && !self.is_escaped => {
                    self.in_brackets = true;
                    true
                }
                '[' if self.is_escaped => {
                    self.is_escaped = false;
                    true
                }
                ']' if !self.in_quotes && !self.is_escaped => {
                    self.in_brackets = false;
                    true
                }
                ']' if self.is_escaped => {
                    self.is_escaped = false;
                    true
                }
                '{' if !self.in_quotes && !self.is_escaped => {
                    self.in_braces = true;
                    true
                }
                '{' if self.is_escaped => {
                    self.is_escaped = false;
                    true
                }
                '}' if !self.in_quotes && !self.is_escaped => {
                    self.in_braces = false;
                    true
                }
                '}' if self.is_escaped => {
                    self.is_escaped = false;
                    true
                }
                ':' if self.in_braces || self.in_quotes => {
                    self.is_escaped = false;
                    true
                }
                ',' if self.in_brackets || self.in_quotes || self.in_braces => {
                    self.is_escaped = false;
                    true
                }
                _ => {
                    self.is_escaped = false;
                    self.in_quotes
                }
            })
            .collect()
    }
}

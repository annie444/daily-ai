pub mod commit_message;
pub mod label_urls;
pub mod summary;
pub mod tools;

use async_openai::Client;
use async_openai::config::{Config, OpenAIConfig};
use tracing::info;

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
#[derive(Debug, PartialEq, Clone, Copy)]
enum Expectation {
    Value,
    Key,
    Colon,
    CommaOrEnd,
    Done,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Container {
    Object,
    Array,
}

pub(super) struct ResponseCleaner<'cleaner> {
    response: &'cleaner str,
    stack: Vec<Container>,
    expect: Expectation,
    in_quotes: bool,
    is_escaped: bool,
    in_number: bool,
    literal_buffer: String,
}

impl<'cleaner> ResponseCleaner<'cleaner> {
    pub fn new(response: &'cleaner str) -> Self {
        info!("Cleaning AI response: {response}");
        Self {
            response,
            stack: Vec::new(),
            expect: Expectation::Value,
            in_quotes: false,
            is_escaped: false,
            in_number: false,
            literal_buffer: String::new(),
        }
    }

    pub fn clean(&mut self) -> String {
        let mut output = String::with_capacity(self.response.len());
        
        for c in self.response.chars() {
            if self.is_escaped {
                self.is_escaped = false;
                if self.in_quotes {
                    output.push(c);
                }
                continue;
            }

            if self.in_quotes {
                if c == '\\' {
                    self.is_escaped = true;
                    output.push(c);
                } else if c == '"' {
                    self.in_quotes = false;
                    output.push(c);
                    
                    if self.expect == Expectation::Key {
                        self.expect = Expectation::Colon;
                    } else {
                        // Finished value
                        self.transition_after_value();
                    }
                } else {
                    output.push(c);
                }
                continue;
            }

            // Outside quotes

            // Handle Number termination
            if self.in_number {
                 match c {
                     '0'..='9' | '.' | '-' | '+' | 'e' | 'E' => {
                         output.push(c);
                         continue;
                     }
                     _ => {
                         self.in_number = false;
                         self.transition_after_value();
                     }
                 }
            }

            // Handle Literal termination
            if !self.literal_buffer.is_empty() {
                if c.is_ascii_alphabetic() {
                    self.literal_buffer.push(c);
                    continue;
                } else {
                     let valid = matches!(self.literal_buffer.as_str(), "true" | "false" | "null");
                     if valid && self.expect == Expectation::Value {
                         output.push_str(&self.literal_buffer);
                         self.transition_after_value();
                     }
                     self.literal_buffer.clear();
                }
            }
            
            // Check literal start
            if c.is_ascii_alphabetic() {
                self.literal_buffer.push(c);
                continue;
            }
            
            // Check number start
            if (c == '-' || c.is_ascii_digit()) && self.expect == Expectation::Value {
                self.in_number = true;
                output.push(c);
                continue;
            }

            // Structural chars
            match c {
                '{' => {
                    if self.expect == Expectation::Value {
                        self.stack.push(Container::Object);
                        self.expect = Expectation::Key;
                        output.push(c);
                    }
                }
                '[' => {
                    if self.expect == Expectation::Value {
                        self.stack.push(Container::Array);
                        self.expect = Expectation::Value;
                        output.push(c);
                    }
                }
                '}' => {
                    if let Some(Container::Object) = self.stack.last() {
                         if self.expect == Expectation::Key || self.expect == Expectation::CommaOrEnd {
                             self.stack.pop();
                             self.transition_after_value();
                             output.push(c);
                         }
                    }
                }
                ']' => {
                    if let Some(Container::Array) = self.stack.last() {
                         if self.expect == Expectation::Value || self.expect == Expectation::CommaOrEnd {
                             self.stack.pop();
                             self.transition_after_value();
                             output.push(c);
                         }
                    }
                }
                '"' => {
                    if self.expect == Expectation::Key || self.expect == Expectation::Value {
                        self.in_quotes = true;
                        output.push(c);
                    }
                }
                ':' => {
                    if self.expect == Expectation::Colon {
                        self.expect = Expectation::Value;
                        output.push(c);
                    }
                }
                ',' => {
                    if self.expect == Expectation::CommaOrEnd {
                        if let Some(container) = self.stack.last() {
                            match container {
                                Container::Object => self.expect = Expectation::Key,
                                Container::Array => self.expect = Expectation::Value,
                            }
                            output.push(c);
                        }
                    }
                }
                _ => {} // Discard noise
            }
        }
        
        // Final flush
        if !self.literal_buffer.is_empty() {
             let valid = matches!(self.literal_buffer.as_str(), "true" | "false" | "null");
             if valid && self.expect == Expectation::Value {
                 output.push_str(&self.literal_buffer);
             }
        }

        output
    }

    fn transition_after_value(&mut self) {
        self.expect = Expectation::CommaOrEnd;
        if self.stack.is_empty() {
            self.expect = Expectation::Done;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ResponseCleaner;

    #[test]
    fn cleans_simple_object_with_noise() {
        let resp = "random prefix {\"label\": other chars \"Tech\", should be 12 trimmed \"duration\": 1275.0} trailing";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(cleaned, "{\"label\":\"Tech\",\"duration\":1275.0}");
    }

    #[test]
    fn preserves_brackets_and_quotes_inside() {
        let resp = "### [{\"label\": \"A [bracket] test\", # \"with_num\": 1234!Ld}] ###";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(
            cleaned,
            "[{\"label\":\"A [bracket] test\",\"with_num\":1234}]"
        );
    }

    #[test]
    fn handles_escaped_quotes_inside_string() {
        let resp = "!! {\"Number\": 1567,   1.-   \"label\": \"He said \\\"hi\\\"\"} !!";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(
            cleaned,
            "{\"Number\":1567,\"label\":\"He said \\\"hi\\\"\"}"
        );
    }
}

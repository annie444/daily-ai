pub mod commit_message;
pub mod label_urls;
pub mod prompt;
pub mod query;
pub mod summary;
pub mod tools;

use tracing::info;

pub trait SchemaInfo: Sized {
    fn schema_value() -> serde_json::Value;
    fn title() -> String;
    fn description() -> String;
    fn schema() -> schemars::Schema;
}

impl<T> SchemaInfo for T
where
    T: for<'de> serde::Deserialize<'de> + schemars::JsonSchema,
{
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    fn title() -> String {
        Self::schema()
            .get("title")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    fn description() -> String {
        Self::schema()
            .get("description")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    fn schema_value() -> serde_json::Value {
        Self::schema().as_value().to_owned()
    }
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

pub(super) struct ResponseCleaner {
    stack: Vec<Container>,
    expect: Expectation,
    in_quotes: bool,
    is_escaped: bool,
    in_number: bool,
    literal_buffer: String,
}

impl ResponseCleaner {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            expect: Expectation::Value,
            in_quotes: false,
            is_escaped: false,
            in_number: false,
            literal_buffer: String::new(),
        }
    }

    fn reset(&mut self) {
        self.stack.clear();
        self.expect = Expectation::Value;
        self.in_quotes = false;
        self.is_escaped = false;
        self.in_number = false;
        self.literal_buffer.clear();
    }

    pub fn clean<S>(&mut self, response: S) -> String
    where
        S: AsRef<str> + std::fmt::Debug + std::fmt::Display,
    {
        info!("Cleaning AI response: {response}");
        let mut output = String::with_capacity(response.as_ref().len());

        for c in response.as_ref().chars() {
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
                    if let Some(Container::Object) = self.stack.last()
                        && (self.expect == Expectation::Key
                            || self.expect == Expectation::CommaOrEnd)
                    {
                        self.stack.pop();
                        self.transition_after_value();
                        output.push(c);
                    }
                }
                ']' => {
                    if let Some(Container::Array) = self.stack.last()
                        && (self.expect == Expectation::Value
                            || self.expect == Expectation::CommaOrEnd)
                    {
                        self.stack.pop();
                        self.transition_after_value();
                        output.push(c);
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
                    if self.expect == Expectation::CommaOrEnd
                        && let Some(container) = self.stack.last()
                    {
                        match container {
                            Container::Object => self.expect = Expectation::Key,
                            Container::Array => self.expect = Expectation::Value,
                        }
                        output.push(c);
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

        self.reset();
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
        let mut cleaner = ResponseCleaner::new();
        let cleaned = cleaner.clean(resp);
        assert_eq!(cleaned, "{\"label\":\"Tech\",\"duration\":1275.0}");
    }

    #[test]
    fn preserves_brackets_and_quotes_inside() {
        let resp = "### [{\"label\": \"A [bracket] test\", # \"with_num\": 1234!Ld}] ###";
        let mut cleaner = ResponseCleaner::new();
        let cleaned = cleaner.clean(resp);
        assert_eq!(
            cleaned,
            "[{\"label\":\"A [bracket] test\",\"with_num\":1234}]"
        );
    }

    #[test]
    fn handles_escaped_quotes_inside_string() {
        let resp = "!! {\"Number\": 1567,   1.-   \"label\": \"He said \\\"hi\\\"\"} !!";
        let mut cleaner = ResponseCleaner::new();
        let cleaned = cleaner.clean(resp);
        assert_eq!(
            cleaned,
            "{\"Number\":1567,\"label\":\"He said \\\"hi\\\"\"}"
        );
    }
}

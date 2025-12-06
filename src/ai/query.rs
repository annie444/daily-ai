use std::collections::HashMap;

use async_openai::types::responses::ResponseFormatJsonSchema;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use tracing::{error, trace};

use super::SchemaInfo;
use crate::AppResult;

pub trait Query: JsonSchema + Serialize + for<'de> Deserialize<'de> + SchemaInfo {
    const PROMPT: &'static [u8];

    fn response_format() -> ResponseFormatJsonSchema {
        ResponseFormatJsonSchema {
            description: Some(Self::description()),
            schema: Some(schema_for!(Self).as_value().to_owned()),
            name: Self::title(),
            strict: None,
        }
    }

    fn prompt(vars: &HashMap<&str, &str>) -> String {
        let prompt = unsafe {
            String::from_utf8_unchecked(
                zstd::decode_all(std::io::Cursor::new(Self::PROMPT)).unwrap_unchecked(),
            )
        };
        crate::ai::prompt::PromptTemplate::new(prompt).render(vars)
    }

    fn from_str(s: &str) -> AppResult<Self> {
        trace!("Raw content: {s}");

        // 1. Fast path: Try valid JSON first (e.g. from OpenAI Structured Outputs)
        if let Ok(res) = serde_json::from_str(s) {
            return Ok(res);
        }

        // 2. Fallback: Clean up the response (e.g. for local models or markdown blocks)
        let s = crate::ai::ResponseCleaner::new().clean(s);
        trace!("Cleaned content: {s}");
        let jd = &mut serde_json::Deserializer::from_str(&s);
        match serde_path_to_error::deserialize(jd) {
            Ok(res) => Ok(res),
            Err(e) => {
                error!("Failed to deserialize {}: {e}", Self::title());
                error!("Response content was: {s}");
                error!("Failed to parse JSON at path: {}", e.path());
                Err(e.into_inner().into())
            }
        }
    }
}

#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! impl_query {
    ($struct_name:ident, $prompt:ident) => {
        impl crate::ai::query::Query for $struct_name {
            const PROMPT: &'static [u8] = $prompt;
        }
    };
}

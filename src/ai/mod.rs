pub mod commit_message;

use async_openai::Client;
use async_openai::config::{Config, OpenAIConfig};

#[tracing::instrument(level = "debug")]
pub fn get_lm_studio_client<S: AsRef<str> + std::fmt::Debug>(
    server: S,
    port: u16,
) -> Client<Box<dyn Config>> {
    let config = Box::new(OpenAIConfig::default().with_api_base(format!(
        "http://{}:{}/v1",
        server.as_ref(),
        port
    ))) as Box<dyn Config>;
    let client: Client<Box<dyn Config>> = Client::with_config(config);
    client
}

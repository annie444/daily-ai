use std::collections::HashMap;

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::evals::InputTextContent;
use async_openai::types::responses::{
    CreateResponse, FunctionToolCall, InputContent, InputItem, InputMessage, InputParam, InputRole,
    Item, MessageItem, OutputItem, OutputMessageContent, Reasoning, ReasoningEffort,
    RefusalContent, ResponseTextParam, TextResponseFormatConfiguration, ToolChoiceOptions,
    ToolChoiceParam, Truncation,
};
use tracing::{debug, error};

use super::tools::ToolRegistry;
use crate::AppResult;
use crate::ai::query::Query;

pub struct Agent {
    model: String,
}

impl Agent {
    pub fn new(model: Option<String>) -> Self {
        Self {
            model: model.unwrap_or_else(|| "openai/gpt-oss-20b".to_string()),
        }
    }

    pub async fn run<'c, C: Config, Ctx, R, Q>(
        &self,
        client: &Client<C>,
        context: &Ctx,
        initial_user_message: &str,
        vars: &HashMap<&str, &str>,
    ) -> AppResult<Q>
    where
        R: ToolRegistry<Context<'c> = Ctx>,
        Q: Query,
    {
        let system_prompt = Q::prompt(vars);
        let mut input_items: Vec<InputItem> = vec![
            InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                content: vec![InputContent::InputText(InputTextContent {
                    text: initial_user_message.to_string(),
                })],
                role: InputRole::User,
                status: None,
            }))),
            InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                content: vec![InputContent::InputText(InputTextContent {
                    text: system_prompt.to_string(),
                })],
                role: InputRole::System,
                status: None,
            }))),
        ];

        let tools = R::definitions();
        let mut previous_response_id: Option<String> = None;

        loop {
            let request = CreateResponse {
                model: Some(self.model.clone()),
                input: InputParam::Items(input_items.clone()),
                background: Some(false),
                instructions: Some(system_prompt.to_string()),
                parallel_tool_calls: Some(false),
                reasoning: Some(Reasoning {
                    effort: Some(ReasoningEffort::High),
                    summary: None,
                }),
                store: Some(true),
                stream: Some(false),
                temperature: Some(0.05),
                text: Some(ResponseTextParam {
                    format: TextResponseFormatConfiguration::JsonSchema(Q::response_format()),
                    verbosity: None,
                }),
                tool_choice: if tools.is_empty() {
                    None
                } else {
                    Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto))
                },
                tools: if tools.is_empty() {
                    None
                } else {
                    Some(tools.clone())
                },
                top_logprobs: Some(0),
                top_p: Some(0.1),
                truncation: Some(Truncation::Disabled),
                previous_response_id: previous_response_id.clone(),
                ..Default::default()
            };

            let response = client.responses().create(request).await?;
            debug!("AI Response: {:?}", response);
            previous_response_id = Some(response.id.clone());

            let function_calls: Vec<FunctionToolCall> = response
                .output
                .iter()
                .filter_map(|item| {
                    if let OutputItem::FunctionCall(fc) = item {
                        Some(fc.clone())
                    } else {
                        None
                    }
                })
                .collect();

            if function_calls.is_empty() {
                let mut response_content = String::new();
                for out in &response.output {
                    if let OutputItem::Message(msg) = out {
                        for content in &msg.content {
                            match content {
                                OutputMessageContent::OutputText(text) => {
                                    response_content.push_str(&text.text)
                                }
                                OutputMessageContent::Refusal(RefusalContent { refusal }) => {
                                    error!("AI refused prompt: {}", refusal);
                                }
                            }
                        }
                    }
                }
                return Q::from_str(&response_content);
            }

            // Handle tool calls
            for call in function_calls {
                // Pass context clone (Arc clone) to execute
                let items = R::execute(call, context).await;
                input_items.extend(items);
            }
        }
    }
}

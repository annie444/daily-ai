use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::responses::{FunctionToolCall, InputItem, Tool};
use daily_ai_include_zstd::include_zstd;
use git2::{Diff, Repository};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::agent::Agent;
use super::tools::commit::{CommitContext, GetFile, GetPatch};
use super::tools::{CustomTool, ToolRegistry, unknown_tool};
use crate::git::diff::get_diff_summary;
use crate::{AppResult, impl_query};

static COMMIT_MESSAGE_PROMPT: &[u8] = include_zstd!("src/ai/prompts/commit_message_prompt.md");

/// Commit message output from the model: summary plus optional body.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CommitMessage {
    /// Short summary of the commit
    pub summary: String,
    /// Optional detailed body of the commit message
    pub body: Option<String>,
}

impl Display for CommitMessage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary)?;
        if let Some(body) = &self.body {
            write!(f, "\n\n{}", body)?;
        }
        Ok(())
    }
}

impl_query!(CommitMessage, COMMIT_MESSAGE_PROMPT);

pub struct CommitMessageRegistry;

impl ToolRegistry for CommitMessageRegistry {
    type Context<'a> = CommitContext<'a>;

    fn definitions() -> Vec<Tool> {
        vec![
            Tool::Function(GetFile::definition()),
            Tool::Function(GetPatch::definition()),
        ]
    }

    async fn execute(call: FunctionToolCall, context: &Self::Context<'_>) -> Vec<InputItem> {
        match call.name.as_str() {
            name if name == GetFile::name() => GetFile::process(call, context).await,
            name if name == GetPatch::name() => GetPatch::process(call, context).await,
            _ => unknown_tool(call),
        }
    }
}

/// Generate a commit message using the model, optionally calling back into file/patch tools.
#[tracing::instrument(
    name = "Generating a commit message with LLM",
    level = "debug",
    skip(client, diff, repo)
)]
pub async fn generate_commit_message<'c, 'd, C: Config>(
    client: &'c Client<C>,
    diff: &Diff<'d>,
    repo: &Repository,
) -> AppResult<CommitMessage> {
    // Kick off first turn with diff summary and commit prompt.
    let initial_user_message =
        serde_json::to_string_pretty(&get_diff_summary(repo.path().parent().unwrap(), diff)?)?;
    let agent = Agent::new(None);

    agent
        .run::<_, CommitContext, CommitMessageRegistry, CommitMessage>(
            client,
            &CommitContext { repo, diff },
            &initial_user_message,
            &HashMap::new(),
        )
        .await
}

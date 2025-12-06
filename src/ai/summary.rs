use std::collections::HashMap;
use std::path::PathBuf;

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::responses::{FunctionToolCall, InputItem, Tool};
use daily_ai_include_zstd::include_zstd;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::agent::Agent;
use super::tools::ToolRegistry;
use super::tools::fetch::FetchUrl;
use super::tools::summary::{
    GetBrowserHistory, GetCommitMessages, GetDiff, GetRepo, GetShellHistory,
};
use super::tools::{CustomTool, unknown_tool};
use crate::AppResult;
use crate::classify::UrlCluster;
use crate::context::Context;
use crate::git::CommitMeta;
use crate::impl_query;
use crate::shell::ShellHistoryEntry;

static SUMMARY_PROMPT: &[u8] = include_zstd!("src/ai/prompts/full_summary/summary_prompt.md");
static HIGHLIGHTS_PROMPT: &[u8] = include_zstd!("src/ai/prompts/full_summary/highlights_prompt.md");
static TIME_BREAKDOWN_PROMPT: &[u8] =
    include_zstd!("src/ai/prompts/full_summary/time_breakdown_prompt.md");
static COMMON_GROUPS_PROMPT: &[u8] =
    include_zstd!("src/ai/prompts/full_summary/common_groups_prompt.md");
static REPO_SUMMARIES_PROMPT: &[u8] =
    include_zstd!("src/ai/prompts/full_summary/repo_summaries_prompt.md");
static SHELL_OVERVIEW_PROMPT: &[u8] =
    include_zstd!("src/ai/prompts/full_summary/shell_overview_prompt.md");

/// # common_groups
/// Identify common projects or categories of work the changes belong to.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CommonGroupsQuery {
    /// List of common groups
    pub common_groups: Vec<String>,
    /// Any specific notes
    #[serde(default)]
    pub notes: Vec<String>,
}

impl_query!(CommonGroupsQuery, COMMON_GROUPS_PROMPT);

/// # summary
/// Generate a comprehensive summary of the work done based on the provided context.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SummaryQuery {
    /// The summary
    pub summary: String,
    /// Any specific notes
    #[serde(default)]
    pub notes: Vec<String>,
}

impl_query!(SummaryQuery, SUMMARY_PROMPT);

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Highlight {
    /// A highlight title
    pub title: String,
    /// A highlight summary
    pub summary: String,
}

/// # highlights
/// Highlights of the work done.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct HighlightsQuery {
    /// List of highlights
    pub highlights: Vec<Highlight>,
    /// Any specific notes
    #[serde(default)]
    pub notes: Vec<String>,
}

impl_query!(HighlightsQuery, HIGHLIGHTS_PROMPT);

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RepoSummary {
    /// The repository path
    pub repo: PathBuf,
    /// The summary
    pub summary: String,
}

/// # repo_summaries
/// Summaries of changes made per repository.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RepoSummaryQuery {
    /// List of repo summaries
    pub repo_summaries: Vec<RepoSummary>,
    /// Any specific notes
    #[serde(default)]
    pub notes: Vec<String>,
}

impl_query!(RepoSummaryQuery, REPO_SUMMARIES_PROMPT);

/// # shell_overview
/// Summaries of shell history and operations performed.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ShellOverviewQuery {
    /// Overview of shell history
    pub shell_overview: String,
    /// Any specific notes
    #[serde(default)]
    pub notes: Vec<String>,
}

impl_query!(ShellOverviewQuery, SHELL_OVERVIEW_PROMPT);

/// # time_breakdown
/// Breakdown of time spent on different tasks.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct TimeBreakdownQuery {
    /// The time breakdown
    pub time_breakdown: Vec<String>,
    /// Any specific notes
    #[serde(default)]
    pub notes: Vec<String>,
}

impl_query!(TimeBreakdownQuery, TIME_BREAKDOWN_PROMPT);

/// # work_summary
/// Collection of summaries and highlights about the work done.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default)]
pub struct WorkSummary {
    /// Summary of changes made. Should be a concise couple of paragraphs.
    pub summary: String,
    /// Highlights of the changes made.
    #[serde(default)]
    pub highlights: Vec<String>,
    /// Breakdown of time spent on different tasks.
    #[serde(default)]
    pub time_breakdown: Vec<String>,
    /// Common projects or categories of work the changes belong to.
    #[serde(default)]
    pub common_groups: Vec<String>,
    /// Summary of commits made per project or module.
    #[serde(default)]
    pub repo_summaries: Vec<String>,
    /// Overview of shell operations performed. Should be a concise paragraph or two.
    #[serde(default)]
    pub shell_overview: String,
    /// Any notes, observations, recommendations, warnings, or cautions about the work done.
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinifiedContext {
    pub shell_history: Vec<ShellHistoryEntry>,
    pub safari_history: Vec<UrlCluster>,
    pub commit_history: Vec<MinifiedGitRepoHistory>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinifiedGitRepoHistory {
    pub repo: PathBuf,
    pub commits: Vec<CommitMeta>,
}

impl From<&Context> for MinifiedContext {
    fn from(ctx: &Context) -> Self {
        let commit_history = ctx
            .commit_history
            .iter()
            .take(10.min(ctx.commit_history.len()))
            .map(|repo_hist| MinifiedGitRepoHistory {
                repo: repo_hist.diff.repo_path.clone(),
                commits: repo_hist.commits.clone(),
            })
            .collect();
        let safari_history = ctx
            .safari_history
            .iter()
            .map(|cluster| {
                let max_urls = 10.min(cluster.urls.len());
                UrlCluster {
                    label: cluster.label.clone(),
                    urls: cluster.urls[..max_urls].to_vec(),
                }
            })
            .collect();
        let shell_hist_len = 10.min(ctx.shell_history.len());
        MinifiedContext {
            shell_history: ctx.shell_history[..shell_hist_len].to_vec(),
            safari_history,
            commit_history,
            notes: vec![],
        }
    }
}

pub enum QueryType {
    Summary,
    Highlights,
    RepoSummary,
    ShellOverview,
    TimeBreakdown,
    CommonGroups,
}

pub enum QueryResponse {
    Summary(SummaryQuery),
    Highlights(HighlightsQuery),
    RepoSummary(RepoSummaryQuery),
    ShellOverview(ShellOverviewQuery),
    TimeBreakdown(TimeBreakdownQuery),
    CommonGroups(CommonGroupsQuery),
}

impl QueryResponse {
    pub fn extract_notes(&self) -> Vec<String> {
        match self {
            QueryResponse::Summary(q) => q.notes.clone(),
            QueryResponse::Highlights(q) => q.notes.clone(),
            QueryResponse::RepoSummary(q) => q.notes.clone(),
            QueryResponse::ShellOverview(q) => q.notes.clone(),
            QueryResponse::TimeBreakdown(q) => q.notes.clone(),
            QueryResponse::CommonGroups(q) => q.notes.clone(),
        }
    }

    pub fn update_work_summary(&self, ws: &mut WorkSummary) {
        match self {
            QueryResponse::Summary(q) => {
                ws.summary = q.summary.clone();
            }
            QueryResponse::Highlights(q) => {
                ws.highlights = q
                    .highlights
                    .iter()
                    .map(|h| format!("{}: {}", h.title, h.summary))
                    .collect();
            }
            QueryResponse::RepoSummary(q) => {
                ws.repo_summaries = q
                    .repo_summaries
                    .iter()
                    .map(|rs| {
                        let parts = rs.repo.to_string_lossy().to_string();
                        let parts_len = parts.split('/').count();
                        let repo_name = if parts_len >= 2 {
                            format!(
                                "{}/{}",
                                parts.split('/').nth_back(1).unwrap(),
                                parts.split('/').next_back().unwrap()
                            )
                        } else {
                            parts
                        };
                        format!("Repo {}: {}", repo_name, rs.summary)
                    })
                    .collect();
            }
            QueryResponse::ShellOverview(q) => {
                ws.shell_overview = q.shell_overview.clone();
            }
            QueryResponse::TimeBreakdown(q) => {
                ws.time_breakdown = q.time_breakdown.clone();
            }
            QueryResponse::CommonGroups(q) => {
                ws.common_groups = q.common_groups.clone();
            }
        }
    }
}

struct SummaryToolRegistry;

impl ToolRegistry for SummaryToolRegistry {
    type Context<'a> = Context;

    fn definitions() -> Vec<Tool> {
        vec![
            Tool::Function(FetchUrl::definition()),
            Tool::Function(GetDiff::definition()),
            Tool::Function(GetRepo::definition()),
            Tool::Function(GetCommitMessages::definition()),
            Tool::Function(GetBrowserHistory::definition()),
            Tool::Function(GetShellHistory::definition()),
        ]
    }

    async fn execute<'c>(call: FunctionToolCall, context: &Self::Context<'c>) -> Vec<InputItem> {
        match call.name.as_str() {
            name if name == FetchUrl::name() => FetchUrl::process(call, &()).await,
            name if name == GetDiff::name() => {
                GetDiff::process(call, &context.commit_history).await
            }
            name if name == GetRepo::name() => {
                GetRepo::process(call, &context.commit_history).await
            }
            name if name == GetCommitMessages::name() => {
                GetCommitMessages::process(call, &context.commit_history).await
            }
            name if name == GetBrowserHistory::name() => {
                GetBrowserHistory::process(call, &context.safari_history).await
            }
            name if name == GetShellHistory::name() => {
                GetShellHistory::process(call, &context.shell_history).await
            }
            _ => unknown_tool(call),
        }
    }
}

/// Generate a commit message using the model, optionally calling back into file/patch tools.
#[tracing::instrument(
    name = "Generating the full summary of work done",
    level = "debug",
    skip(client, context)
)]
pub async fn generate_summary<C: Config>(
    client: &Client<C>,
    context: &Context,
    template_vars: &HashMap<&str, &str>,
) -> AppResult<WorkSummary> {
    // Kick off first turn with diff summary and commit prompt.
    let mut input_context = MinifiedContext::from(context);
    let queries: Vec<QueryType> = vec![
        QueryType::CommonGroups,
        QueryType::Highlights,
        QueryType::TimeBreakdown,
        QueryType::RepoSummary,
        QueryType::ShellOverview,
        QueryType::Summary,
    ];

    let mut work_summary = WorkSummary::default();
    let mut notes: Vec<String> = vec![];
    let agent = Agent::new(None);

    for query in queries {
        input_context.notes = notes.clone();
        let initial_user_message = serde_json::to_string_pretty(&input_context)?;

        let query_response = match query {
            QueryType::Summary => QueryResponse::Summary(
                agent
                    .run::<C, Context, SummaryToolRegistry, SummaryQuery>(
                        client,
                        context,
                        &initial_user_message,
                        template_vars,
                    )
                    .await?,
            ),
            QueryType::Highlights => QueryResponse::Highlights(
                agent
                    .run::<C, Context, SummaryToolRegistry, HighlightsQuery>(
                        client,
                        context,
                        &initial_user_message,
                        template_vars,
                    )
                    .await?,
            ),
            QueryType::RepoSummary => QueryResponse::RepoSummary(
                agent
                    .run::<C, Context, SummaryToolRegistry, RepoSummaryQuery>(
                        client,
                        context,
                        &initial_user_message,
                        template_vars,
                    )
                    .await?,
            ),
            QueryType::ShellOverview => QueryResponse::ShellOverview(
                agent
                    .run::<C, Context, SummaryToolRegistry, ShellOverviewQuery>(
                        client,
                        context,
                        &initial_user_message,
                        template_vars,
                    )
                    .await?,
            ),
            QueryType::TimeBreakdown => QueryResponse::TimeBreakdown(
                agent
                    .run::<C, Context, SummaryToolRegistry, TimeBreakdownQuery>(
                        client,
                        context,
                        &initial_user_message,
                        template_vars,
                    )
                    .await?,
            ),
            QueryType::CommonGroups => QueryResponse::CommonGroups(
                agent
                    .run::<C, Context, SummaryToolRegistry, CommonGroupsQuery>(
                        client,
                        context,
                        &initial_user_message,
                        template_vars,
                    )
                    .await?,
            ),
        };

        query_response.update_work_summary(&mut work_summary);
        notes.extend(query_response.extract_notes());
    }

    work_summary.notes = notes;
    Ok(work_summary)
}

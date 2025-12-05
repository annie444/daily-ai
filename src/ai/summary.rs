use std::path::PathBuf;

use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::evals::InputTextContent;
use async_openai::types::responses::{
    CreateResponse, FunctionToolCall, InputContent, InputItem, InputMessage, InputParam, InputRole,
    Item, MessageItem, OutputItem, OutputMessageContent, OutputStatus, Reasoning, ReasoningEffort,
    RefusalContent, ResponseFormatJsonSchema, ResponseTextParam, TextResponseFormatConfiguration,
    Tool, ToolChoiceOptions, ToolChoiceParam, Truncation,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, trace};

use super::ResponseCleaner;
use super::tools::{CustomTool, FetchUrl, unknown_tool};
use crate::AppResult;
use crate::classify::UrlCluster;
use crate::context::Context;
use crate::git::diff::DiffSummary;
use crate::git::{CommitMeta, GitRepoHistory};
use crate::shell::ShellHistoryEntry;
use crate::time_utils::system_time_to_offset_datetime;

static SUMMARY_PROMPT: &str = std::include_str!("full_summary/summary_prompt.txt");
static HIGHLIGHTS_PROMPT: &str = std::include_str!("full_summary/highlights_prompt.txt");
static TIME_BREAKDOWN_PROMPT: &str = std::include_str!("full_summary/time_breakdown_prompt.txt");
static COMMON_GROUPS_PROMPT: &str = std::include_str!("full_summary/common_groups_prompt.txt");
static REPO_SUMMARIES_PROMPT: &str = std::include_str!("full_summary/repo_summaries_prompt.txt");
static SHELL_OVERVIEW_PROMPT: &str = std::include_str!("full_summary/shell_overview_prompt.txt");

trait Query: JsonSchema + Serialize + for<'de> Deserialize<'de> {
    const PROMPT: &'static str;

    fn title() -> String {
        let schema = schema_for!(Self);
        schema.get("title").unwrap().as_str().unwrap().to_string()
    }

    fn response_format() -> ResponseFormatJsonSchema {
        let schema = schema_for!(Self);
        let title = schema.get("title").unwrap().as_str().unwrap();
        let description = schema.get("description").unwrap().as_str().unwrap();
        ResponseFormatJsonSchema {
            description: Some(description.to_string()),
            schema: Some(schema_for!(Self).as_value().to_owned()),
            name: title.to_string(),
            strict: None,
        }
    }

    fn prompt() -> &'static str {
        Self::PROMPT
    }

    fn from_str(s: &str) -> AppResult<Self> {
        let jd = &mut serde_json::Deserializer::from_str(s);
        match serde_path_to_error::deserialize(jd) {
            Ok(res) => Ok(res),
            Err(e) => {
                error!(
                    "Failed to deserialize query response into {}: {e}",
                    Self::title()
                );
                error!("Response content was: {s}");
                error!("Failed to parse JSON at path: {}", e.path());
                Err(e.into_inner().into())
            }
        }
    }
}

macro_rules! impl_json_schema {
    ($struct_name:ident, $prompt:ident) => {
        impl Query for $struct_name {
            const PROMPT: &'static str = $prompt;
        }
    };
}

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

impl_json_schema!(CommonGroupsQuery, COMMON_GROUPS_PROMPT);

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

impl_json_schema!(SummaryQuery, SUMMARY_PROMPT);

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

impl_json_schema!(HighlightsQuery, HIGHLIGHTS_PROMPT);

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

impl_json_schema!(RepoSummaryQuery, REPO_SUMMARIES_PROMPT);

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

impl_json_schema!(ShellOverviewQuery, SHELL_OVERVIEW_PROMPT);

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

impl_json_schema!(TimeBreakdownQuery, TIME_BREAKDOWN_PROMPT);

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

/// # get_diff
/// Retrieve the complete diff of changes in a repository.
/// This includes the commit history, branches, full diffs, and other metadata.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetDiff {
    /// Path to the repo
    pub repo: PathBuf,
    /// Optional path to the specific file to retrieve the diff for
    #[serde(default)]
    pub file_path: Option<PathBuf>,
}

impl CustomTool for GetDiff {
    type Context<'a> = Vec<GitRepoHistory>;
    const NAME: &'static str = "get_diff";
    const DESCRIPTION: &'static str = "Retrieve the complete diff of changes in a repository.";

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        let repo_hist = match context.iter().find(|r| r.diff.repo_path == self.repo) {
            Some(r) => r,
            None => {
                let error_msg = format!("Repository not found in history graph: {:?}", self.repo);
                error!(error_msg);
                return (OutputStatus::Incomplete, error_msg);
            }
        };
        match &self.file_path {
            Some(file_path) => {
                let diff_output = DiffSummary {
                    repo_path: repo_hist.diff.repo_path.clone(),
                    unmodified: repo_hist
                        .diff
                        .unmodified
                        .iter()
                        .filter(|d| *d == file_path)
                        .cloned()
                        .collect(),
                    added: repo_hist
                        .diff
                        .added
                        .iter()
                        .filter(|d| d.path == *file_path)
                        .cloned()
                        .collect(),
                    deleted: repo_hist
                        .diff
                        .deleted
                        .iter()
                        .filter(|d| *d == file_path)
                        .cloned()
                        .collect(),
                    modified: repo_hist
                        .diff
                        .modified
                        .iter()
                        .filter(|d| d.path == *file_path)
                        .cloned()
                        .collect(),
                    renamed: repo_hist
                        .diff
                        .renamed
                        .iter()
                        .filter(|d| d.from == *file_path || d.to == *file_path)
                        .cloned()
                        .collect(),
                    copied: repo_hist
                        .diff
                        .copied
                        .iter()
                        .filter(|d| d.from == *file_path || d.to == *file_path)
                        .cloned()
                        .collect(),
                    untracked: repo_hist
                        .diff
                        .untracked
                        .iter()
                        .filter(|d| d.path == *file_path)
                        .cloned()
                        .collect(),
                    typechange: repo_hist
                        .diff
                        .typechange
                        .iter()
                        .filter(|d| *d == file_path)
                        .cloned()
                        .collect(),
                    unreadable: repo_hist
                        .diff
                        .unreadable
                        .iter()
                        .filter(|d| *d == file_path)
                        .cloned()
                        .collect(),
                    conflicted: repo_hist
                        .diff
                        .conflicted
                        .iter()
                        .filter(|d| *d == file_path)
                        .cloned()
                        .collect(),
                };
                match serde_json::to_string_pretty(&diff_output) {
                    Ok(json) => (OutputStatus::Completed, json),
                    Err(e) => {
                        let error_msg = format!(
                            "Failed to serialize diff for file {} in repo {}: {e}",
                            file_path.display(),
                            self.repo.display()
                        );
                        error!(error_msg);
                        (OutputStatus::Incomplete, error_msg)
                    }
                }
            }
            None => match serde_json::to_string_pretty(&repo_hist.diff) {
                Ok(json) => (OutputStatus::Completed, json),
                Err(e) => {
                    let error_msg = format!(
                        "Failed to serialize diff for repo {}: {e}",
                        self.repo.display()
                    );
                    error!(error_msg);
                    (OutputStatus::Incomplete, error_msg)
                }
            },
        }
    }
}

/// # get_repo
/// Retrieve the complete history of a repository.
/// This includes the commit history, branches, full diffs, and other metadata.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetRepo {
    /// Path to the repo
    pub repo: PathBuf,
}

impl CustomTool for GetRepo {
    type Context<'a> = Vec<GitRepoHistory>;
    const NAME: &'static str = "get_repo";
    const DESCRIPTION: &'static str = "Retrieve the complete history of a repository.";

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        let repo_hist = match context.iter().find(|r| r.diff.repo_path == self.repo) {
            Some(r) => r,
            None => {
                let error_msg = format!(
                    "Repository not found in history graph: {}",
                    self.repo.display()
                );
                error!(error_msg);
                return (OutputStatus::Incomplete, error_msg);
            }
        };
        match serde_json::to_string_pretty(&repo_hist) {
            Ok(json) => (OutputStatus::Completed, json),
            Err(e) => {
                let error_msg = format!(
                    "Failed to serialize history for repo {}: {e}",
                    self.repo.display()
                );
                error!(error_msg);
                (OutputStatus::Incomplete, error_msg)
            }
        }
    }
}

/// # get_commit_messages
/// Get the list of commit messages collected.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetCommitMessages {
    /// Path to the repo
    pub repo: String,
    /// Maximum number of commit messages to retrieve
    #[serde(default)]
    pub max_messages: Option<usize>,
}

impl CustomTool for GetCommitMessages {
    type Context<'a> = Vec<GitRepoHistory>;
    const NAME: &'static str = "get_commit_messages";
    const DESCRIPTION: &'static str = "Get the list of commit messages collected.";

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        let repo_hist = match context
            .iter()
            .find(|r| r.diff.repo_path.to_string_lossy() == self.repo)
        {
            Some(r) => r,
            None => {
                let error_msg = format!("Repository not found in history graph: {}", self.repo);
                error!(error_msg);
                return (OutputStatus::Incomplete, error_msg);
            }
        };
        let messages: Vec<CommitMeta> = repo_hist
            .commits
            .iter()
            .take(self.max_messages.unwrap_or(repo_hist.commits.len()))
            .cloned()
            .collect();
        match serde_json::to_string_pretty(&messages) {
            Ok(json) => (OutputStatus::Completed, json),
            Err(e) => {
                let error_msg = format!(
                    "Failed to serialize commit messages for repo {}: {e}",
                    self.repo
                );
                error!(error_msg);
                (OutputStatus::Incomplete, error_msg)
            }
        }
    }
}

/// # get_browser_history
/// Get the browser history. For each entry there is a URL, title, visit count, and last visited timestamp.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetBrowserHistory {
    /// The group(s)/categor(y/ies) of URLs to retrieve
    pub groups: Option<Vec<String>>,
    /// Maximum number of URLs to retrieve
    #[serde(default)]
    pub max_urls: Option<usize>,
}

impl CustomTool for GetBrowserHistory {
    type Context<'a> = Vec<UrlCluster>;
    const NAME: &'static str = "get_browser_history";
    const DESCRIPTION: &'static str = "Get the browser history.";

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        let filtered_clusters: Vec<UrlCluster> = match &self.groups {
            Some(group_names) => {
                let group_names: Vec<String> =
                    group_names.iter().map(|s| s.to_lowercase()).collect();
                context
                    .iter()
                    .filter(|c| group_names.contains(&c.label.to_lowercase()))
                    .cloned()
                    .collect()
            }
            None => context.to_vec(),
        };
        let limited_urls: Vec<UrlCluster> = if let Some(max) = self.max_urls {
            filtered_clusters
                .into_iter()
                .map(|mut u| {
                    let url_len = u.urls.len();
                    u.urls = u.urls.into_iter().take(max.min(url_len)).collect();
                    u
                })
                .collect()
        } else {
            filtered_clusters
        };
        match serde_json::to_string_pretty(&limited_urls) {
            Ok(json) => (OutputStatus::Completed, json),
            Err(e) => {
                let error_msg = format!("Failed to serialize browser history: {e}");
                error!(error_msg);
                (OutputStatus::Incomplete, error_msg)
            }
        }
    }
}

/// # get_shell_history
/// Get the shell history. For each entry there is a command, timestamp, directory, exit code, and
/// other metadata.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct GetShellHistory {
    /// Optional starting timestamp to retrieve history from
    #[serde(default)]
    pub start_time: Option<String>,
    /// Optional ending timestamp to retrieve history to
    #[serde(default)]
    pub end_time: Option<String>,
    /// Maximum number of shell history entries to retrieve
    #[serde(default)]
    pub max_entries: Option<usize>,
    /// Optional filter for specific commands
    #[serde(default)]
    pub command: Option<String>,
    /// Optional filter for specific directories
    #[serde(default)]
    pub directory: Option<PathBuf>,
}

impl CustomTool for GetShellHistory {
    type Context<'a> = Vec<ShellHistoryEntry>;
    const NAME: &'static str = "get_shell_history";
    const DESCRIPTION: &'static str = "Get the shell history.";

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        let mut history: Vec<ShellHistoryEntry> = if let Some(start_time) = &self.start_time {
            let start = match humantime::parse_rfc3339_weak(start_time) {
                Ok(dt) => system_time_to_offset_datetime(dt),
                Err(e) => {
                    let error_msg = format!(
                        "Failed to parse start_time '{}' as RFC3339: {e}",
                        start_time
                    );
                    error!(error_msg);
                    return (OutputStatus::Incomplete, error_msg);
                }
            };
            context
                .iter()
                .filter(|entry| entry.date_time >= start)
                .cloned()
                .collect()
        } else {
            context.clone()
        };
        if let Some(end_time) = &self.end_time {
            let end = match humantime::parse_rfc3339_weak(end_time) {
                Ok(dt) => system_time_to_offset_datetime(dt),
                Err(e) => {
                    let error_msg =
                        format!("Failed to parse end_time '{}' as RFC3339: {e}", end_time);
                    error!(error_msg);
                    return (OutputStatus::Incomplete, error_msg);
                }
            };
            history.retain(|entry| entry.date_time <= end);
        }
        if let Some(command_filter) = &self.command {
            history.retain(|entry| entry.command.contains(command_filter));
        }
        if let Some(directory_filter) = &self.directory {
            history.retain(|entry| entry.directory == *directory_filter);
        }
        if let Some(max) = self.max_entries {
            history = history.into_iter().take(max).collect();
        }
        match serde_json::to_string_pretty(&history) {
            Ok(json) => (OutputStatus::Completed, json),
            Err(e) => {
                let error_msg = format!("Failed to serialize shell history: {e}");
                error!(error_msg);
                (OutputStatus::Incomplete, error_msg)
            }
        }
    }
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

impl QueryType {
    pub fn response_format(&self) -> ResponseFormatJsonSchema {
        match self {
            QueryType::Summary => SummaryQuery::response_format(),
            QueryType::Highlights => HighlightsQuery::response_format(),
            QueryType::RepoSummary => RepoSummaryQuery::response_format(),
            QueryType::ShellOverview => ShellOverviewQuery::response_format(),
            QueryType::TimeBreakdown => TimeBreakdownQuery::response_format(),
            QueryType::CommonGroups => CommonGroupsQuery::response_format(),
        }
    }

    pub fn prompt(&self) -> &'static str {
        match self {
            QueryType::Summary => SummaryQuery::prompt(),
            QueryType::Highlights => HighlightsQuery::prompt(),
            QueryType::RepoSummary => RepoSummaryQuery::prompt(),
            QueryType::ShellOverview => ShellOverviewQuery::prompt(),
            QueryType::TimeBreakdown => TimeBreakdownQuery::prompt(),
            QueryType::CommonGroups => CommonGroupsQuery::prompt(),
        }
    }

    pub fn get_response(&self, s: &str) -> AppResult<QueryResponse> {
        match self {
            QueryType::Summary => Ok(QueryResponse::Summary(SummaryQuery::from_str(s)?)),
            QueryType::Highlights => Ok(QueryResponse::Highlights(HighlightsQuery::from_str(s)?)),
            QueryType::RepoSummary => {
                Ok(QueryResponse::RepoSummary(RepoSummaryQuery::from_str(s)?))
            }
            QueryType::ShellOverview => Ok(QueryResponse::ShellOverview(
                ShellOverviewQuery::from_str(s)?,
            )),
            QueryType::TimeBreakdown => Ok(QueryResponse::TimeBreakdown(
                TimeBreakdownQuery::from_str(s)?,
            )),
            QueryType::CommonGroups => {
                Ok(QueryResponse::CommonGroups(CommonGroupsQuery::from_str(s)?))
            }
        }
    }
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

/// Generate a commit message using the model, optionally calling back into file/patch tools.
#[tracing::instrument(
    name = "Generating the full summary of work done",
    level = "debug",
    skip(client, context)
)]
pub async fn generate_summary<C: Config>(
    client: &Client<C>,
    context: &Context,
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
    let tools = vec![
        Tool::Function(FetchUrl::definition()),
        Tool::Function(GetDiff::definition()),
        Tool::Function(GetRepo::definition()),
        Tool::Function(GetCommitMessages::definition()),
        Tool::Function(GetBrowserHistory::definition()),
        Tool::Function(GetShellHistory::definition()),
    ];

    for query in queries {
        let mut previous_response_id: Option<String> = None;
        input_context.notes = notes.clone();

        let mut input_items: Vec<InputItem> = vec![
            InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                content: vec![InputContent::InputText(InputTextContent {
                    text: serde_json::to_string_pretty(&input_context)?,
                })],
                role: InputRole::User,
                status: None,
            }))),
            InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
                content: vec![InputContent::InputText(InputTextContent {
                    text: query.prompt().to_string(),
                })],
                role: InputRole::System,
                status: None,
            }))),
        ];

        loop {
            let request = CreateResponse {
                model: Some("openai/gpt-oss-20b".to_string()),
                input: InputParam::Items(input_items.clone()),
                background: Some(false),
                instructions: Some(query.prompt().to_string()),
                parallel_tool_calls: Some(false),
                reasoning: Some(Reasoning {
                    effort: Some(ReasoningEffort::High),
                    summary: None,
                }),
                store: Some(true),
                stream: Some(false),
                temperature: Some(0.05),
                text: Some(ResponseTextParam {
                    format: TextResponseFormatConfiguration::JsonSchema(query.response_format()),
                    verbosity: None,
                }),
                tool_choice: Some(ToolChoiceParam::Mode(ToolChoiceOptions::Auto)),
                tools: Some(tools.clone()),
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
                trace!("Raw response content: {}", response_content);
                let response_content = ResponseCleaner::new(&response_content).clean();
                trace!("Cleaned response content: {}", response_content);
                let query_response = query.get_response(&response_content)?;
                query_response.update_work_summary(&mut work_summary);
                notes.extend(query_response.extract_notes());
                break;
            }

            // Handle each tool call in order and feed results back into the conversation.
            for call in function_calls {
                match call.name.as_str() {
                    name if name == FetchUrl::NAME => {
                        input_items.extend(FetchUrl::process(call, &()).await);
                    }
                    name if name == GetDiff::NAME => {
                        input_items.extend(GetDiff::process(call, &context.commit_history).await);
                    }
                    name if name == GetRepo::NAME => {
                        input_items.extend(GetRepo::process(call, &context.commit_history).await);
                    }
                    name if name == GetCommitMessages::NAME => {
                        input_items.extend(
                            GetCommitMessages::process(call, &context.commit_history).await,
                        );
                    }
                    name if name == GetBrowserHistory::NAME => {
                        input_items.extend(
                            GetBrowserHistory::process(call, &context.safari_history).await,
                        );
                    }
                    name if name == GetShellHistory::NAME => {
                        input_items
                            .extend(GetShellHistory::process(call, &context.shell_history).await);
                    }
                    _ => input_items.extend(unknown_tool(call)),
                };
            }
        }
    }

    work_summary.notes = notes;
    Ok(work_summary)
}

use std::path::PathBuf;

use async_openai::types::responses::OutputStatus;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::error;

use super::CustomTool;
use crate::classify::UrlCluster;
use crate::git::diff::DiffSummary;
use crate::git::{CommitMeta, GitRepoHistory};
use crate::shell::ShellHistoryEntry;
use crate::time_utils::system_time_to_offset_datetime;

/// # get_diff
/// Retrieve the complete diff of changes in a repository.
/// This includes the commit history, branches, full diffs, and other metadata.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetDiff {
    /// Path to the repo
    pub repo: PathBuf,
    /// Optional path to the specific file to retrieve the diff for
    #[serde(default)]
    pub file_path: Option<PathBuf>,
}

/// # get_repo
/// Retrieve the complete history of a repository.
/// This includes the commit history, branches, full diffs, and other metadata.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetRepo {
    /// Path to the repo
    pub repo: PathBuf,
}

/// # get_commit_messages
/// Get the list of commit messages collected.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetCommitMessages {
    /// Path to the repo
    pub repo: String,
    /// Maximum number of commit messages to retrieve
    #[serde(default)]
    pub max_messages: Option<usize>,
}

/// # get_browser_history
/// Get the browser history. For each entry there is a URL, title, visit count, and last visited timestamp.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetBrowserHistory {
    /// The group(s)/categor(y/ies) of URLs to retrieve
    pub groups: Option<Vec<String>>,
    /// Maximum number of URLs to retrieve
    #[serde(default)]
    pub max_urls: Option<usize>,
}

/// # get_shell_history
/// Get the shell history. For each entry there is a command, timestamp, directory, exit code, and
/// other metadata.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetShellHistory {
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

impl CustomTool for GetDiff {
    type Context<'a> = Vec<GitRepoHistory>;

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

impl CustomTool for GetRepo {
    type Context<'a> = Vec<GitRepoHistory>;

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

impl CustomTool for GetCommitMessages {
    type Context<'a> = Vec<GitRepoHistory>;

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

impl CustomTool for GetBrowserHistory {
    type Context<'a> = Vec<UrlCluster>;

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

impl CustomTool for GetShellHistory {
    type Context<'a> = Vec<ShellHistoryEntry>;

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

use serde::{Deserialize, Serialize};

use crate::ai::summary::WorkSummary;
use crate::classify::UrlCluster;
use crate::git::hist::GitRepoHistory;
use crate::shell::ShellHistoryEntry;

/// Aggregate of all histories collected by the tool for a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub shell_history: Vec<ShellHistoryEntry>,
    pub safari_history: Vec<UrlCluster>,
    pub commit_history: Vec<GitRepoHistory>,
}

/// Aggregate of all histories collected by the tool for a run.
#[derive(Debug, Serialize, Deserialize)]
pub struct FullContext {
    pub shell_history: Vec<ShellHistoryEntry>,
    pub safari_history: Vec<UrlCluster>,
    pub commit_history: Vec<GitRepoHistory>,
    pub summary: Option<WorkSummary>,
}

impl From<(Context, WorkSummary)> for FullContext {
    fn from((context, summary): (Context, WorkSummary)) -> Self {
        FullContext {
            shell_history: context.shell_history,
            safari_history: context.safari_history,
            commit_history: context.commit_history,
            summary: Some(summary),
        }
    }
}

impl From<Context> for FullContext {
    fn from(context: Context) -> Self {
        FullContext {
            shell_history: context.shell_history,
            safari_history: context.safari_history,
            commit_history: context.commit_history,
            summary: None,
        }
    }
}

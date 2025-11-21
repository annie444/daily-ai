use crate::git::diff::DiffSummary;
use crate::safari::SafariHistoryItem;
use crate::shell::ShellHistoryEntry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Context {
    pub shell_history: Vec<ShellHistoryEntry>,
    pub safari_history: Vec<SafariHistoryItem>,
    pub commit_history: Vec<DiffSummary>,
}

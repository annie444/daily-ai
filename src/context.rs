use crate::classify::UrlCluster;
use crate::git::hist::GitRepoHistory;
use crate::shell::ShellHistoryEntry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Context {
    pub shell_history: Vec<ShellHistoryEntry>,
    pub safari_history: Vec<UrlCluster>,
    pub commit_history: Vec<GitRepoHistory>,
}

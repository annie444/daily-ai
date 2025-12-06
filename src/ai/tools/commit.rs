use std::path::PathBuf;

use async_openai::types::responses::OutputStatus;
use git2::{Diff, Repository};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::error;

use super::CustomTool;
use crate::git::diff::{get_file, get_patch};

pub struct CommitContext<'a> {
    pub repo: &'a Repository,
    pub diff: &'a Diff<'a>,
}

/// # get_file
/// Retrieve a file or a segment of a file from the repository.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetFile {
    /// Path to the file
    pub path: PathBuf,
    /// Optional starting line of the file (for partial retrieval)
    pub start_line: Option<usize>,
    /// Optional ending line of the file (for partial retrieval)
    pub end_line: Option<usize>,
}

/// # get_patch
/// Retrieve a patch or a segment of a patch from the repository.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct GetPatch {
    /// Path to the file the patch applies to
    pub path: PathBuf,
    /// Optional starting line of the patch (for partial retrieval)
    pub start_line: Option<usize>,
    /// Optional ending line of the patch (for partial retrieval)
    pub end_line: Option<usize>,
}

impl CustomTool for GetFile {
    type Context<'a> = CommitContext<'a>;

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        match get_file(
            context.repo,
            context.diff,
            &self.path,
            self.start_line,
            self.end_line,
        ) {
            Ok(content) => (OutputStatus::Completed, content),
            Err(e) => {
                let error_msg = format!("Error retrieving file {:?}: {}", self.path, e);
                error!("{}", error_msg);
                (OutputStatus::Incomplete, error_msg)
            }
        }
    }
}

impl CustomTool for GetPatch {
    type Context<'a> = CommitContext<'a>;

    async fn call(&self, context: &Self::Context<'_>) -> (OutputStatus, String) {
        match get_patch(
            context.diff,
            &self.path,
            self.start_line.map(|n| n as u32),
            self.end_line.map(|n| n as u32),
        ) {
            Ok(content) => (OutputStatus::Completed, content),
            Err(e) => {
                let error_msg = format!("Error retrieving patch for {:?}: {}", self.path, e);
                error!("{}", error_msg);
                (OutputStatus::Incomplete, error_msg)
            }
        }
    }
}

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, ser};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::debug;

use crate::AppResult;
use crate::cli::OutputFormat;
use crate::context::Context;
use crate::git::diff::{DiffFromTo, DiffSummary, DiffWithPatch};

/// Aggregated view of paths per repository used when writing summaries to disk.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct RepoPathsSummary {
    pub repo_path: PathBuf,
    pub unmodified: HashSet<PathBuf>,
    pub deleted: HashSet<PathBuf>,
    pub renamed: HashSet<DiffFromTo>,
    pub copied: HashSet<DiffFromTo>,
    pub typechange: HashSet<PathBuf>,
    pub unreadable: HashSet<PathBuf>,
    pub conflicted: HashSet<PathBuf>,
}

/// Write output in the requested format (json or directory layout).
#[tracing::instrument(name = "Saving output to disk", level = "debug", skip(context))]
pub async fn write_output<P: AsRef<Path> + std::fmt::Debug>(
    output: P,
    format: &OutputFormat,
    context: &Context,
) -> AppResult<()> {
    match format {
        OutputFormat::Json => write_json_output(output, context).await,
        OutputFormat::Dir => write_dir_output(output, context).await,
    }
}

/// Write output to a directory structure.
#[tracing::instrument(
    name = "Creating directories and writing output",
    level = "debug",
    skip(context)
)]
async fn write_dir_output<P: AsRef<Path> + std::fmt::Debug>(
    output: P,
    context: &Context,
) -> AppResult<()> {
    // Ensure base output directory exists.
    fs::create_dir_all(&output).await?;

    // Write shell history
    let shell_history_path = output.as_ref().join("shell_history.json");
    write_json_output(shell_history_path, &context.shell_history).await?;

    // Write safari history
    let safari_history_path = output.as_ref().join("safari_history.json");
    write_json_output(safari_history_path, &context.safari_history).await?;

    // Write git commit histories
    let mut unknown_repo_count = 1;
    for repo_history in &context.commit_history {
        let DiffSummary {
            repo_path,
            unmodified,
            added,
            deleted,
            modified,
            copied,
            renamed,
            untracked,
            typechange,
            unreadable,
            conflicted,
        } = repo_history.diff.clone();
        let repo_name = match repo_path.iter().next_back() {
            Some(name) => match name.to_str() {
                Some(name) => name.to_owned(),
                None => {
                    let repo_name = format!("unknown_repo_{}", unknown_repo_count);
                    unknown_repo_count += 1;
                    repo_name
                }
            },
            None => {
                let repo_name = format!("unknown_repo_{}", unknown_repo_count);
                unknown_repo_count += 1;
                repo_name
            }
        };
        let repo_summary_path = output.as_ref().join(repo_name);
        let git_history_path = repo_summary_path.join("git_history_paths.json");
        let commit_log_path = repo_summary_path.join("commit_log.json");
        fs::create_dir_all(&repo_summary_path).await?;
        let commit_summary = RepoPathsSummary {
            repo_path,
            unmodified,
            deleted,
            renamed,
            copied,
            typechange,
            unreadable,
            conflicted,
        };
        write_json_output(git_history_path, &commit_summary).await?;
        write_json_output(commit_log_path, &repo_history.commits).await?;
        for patches in [added, modified, untracked] {
            write_patches(&repo_summary_path, patches).await?;
        }
    }

    Ok(())
}

/// Write git patches to patch files.
#[tracing::instrument(name = "Writing patch files", level = "trace", skip(patches))]
async fn write_patches<P: AsRef<Path> + std::fmt::Debug>(
    dir: P,
    patches: Vec<DiffWithPatch>,
) -> AppResult<()> {
    for patch in patches {
        let patch_file = dir.as_ref().join(patch.path.with_extension("patch"));
        debug!("Writing patch to {:?}", patch_file);
        fs::create_dir_all(patch_file.parent().unwrap()).await?;
        write_file(&patch_file, patch.patch).await?;
    }
    Ok(())
}

/// Serialize an object to pretty JSON and write it to disk.
#[tracing::instrument(name = "Writing JSON file", level = "trace", skip(obj))]
async fn write_json_output<P: AsRef<Path> + std::fmt::Debug, S: ser::Serialize>(
    output: P,
    obj: &S,
) -> AppResult<()> {
    let data = serde_json::to_string_pretty(obj).unwrap();
    write_file(output, data).await
}

/// Write raw string data to a file, overwriting any existing content.
async fn write_file<P: AsRef<Path> + std::fmt::Debug>(output: P, data: String) -> AppResult<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output)
        .await?;
    file.write_all(data.as_bytes()).await?;
    file.flush().await?;
    Ok(())
}

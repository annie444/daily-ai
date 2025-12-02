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
#[tracing::instrument(name = "Saving output to disk", level = "info", skip(context))]
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
    level = "info",
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
#[tracing::instrument(name = "Writing patch files", level = "info", skip(patches))]
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
#[tracing::instrument(name = "Writing JSON file", level = "info", skip(obj))]
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

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::PathBuf};

    use time::{Duration, OffsetDateTime};
    use tokio::fs;

    use crate::{
        classify::UrlCluster,
        context::Context,
        git::diff::{DiffFromTo, DiffSummary, DiffWithPatch},
        git::hist::{CommitMeta, GitRepoHistory},
        safari::SafariHistoryItem,
        shell::ShellHistoryEntry,
    };

    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nonce = OffsetDateTime::now_utc().unix_timestamp_nanos();
        dir.push(format!("{name}_{nonce}"));
        dir
    }

    fn sample_context() -> Context {
        let shell_history = vec![ShellHistoryEntry {
            date_time: OffsetDateTime::UNIX_EPOCH,
            duration: Duration::seconds(1),
            host: "localhost".into(),
            directory: PathBuf::from("/tmp"),
            command: "echo test".into(),
            exit_code: 0,
            session_id: "abc".into(),
        }];
        let safari_history = vec![UrlCluster {
            label: "Example".into(),
            urls: vec![SafariHistoryItem {
                url: "https://example.com".into(),
                title: Some("Example".into()),
                visit_count: 1,
                last_visited: OffsetDateTime::UNIX_EPOCH,
            }],
        }];
        let diff = DiffSummary {
            repo_path: PathBuf::from("/repo"),
            unmodified: HashSet::new(),
            added: vec![DiffWithPatch {
                path: PathBuf::from("foo.txt"),
                patch: "+++".into(),
            }],
            deleted: HashSet::new(),
            modified: Vec::new(),
            renamed: HashSet::from([DiffFromTo {
                from: PathBuf::from("old"),
                to: PathBuf::from("new"),
            }]),
            copied: HashSet::new(),
            untracked: Vec::new(),
            typechange: HashSet::new(),
            unreadable: HashSet::new(),
            conflicted: HashSet::new(),
        };
        let commits = vec![CommitMeta {
            message: "init".into(),
            timestamp: OffsetDateTime::UNIX_EPOCH,
            branches: vec!["main".into()],
        }];
        let commit_history = vec![GitRepoHistory { diff, commits }];

        Context {
            shell_history,
            safari_history,
            commit_history,
        }
    }

    #[tokio::test]
    async fn write_json_output_creates_file() {
        let dir = temp_dir("json_output");
        fs::create_dir_all(&dir).await.unwrap();
        let file = dir.join("out.json");
        let data = vec![1.0_f64, 2.0_f64];

        write_json_output(&file, &data).await.unwrap();

        let contents = fs::read_to_string(&file).await.unwrap();
        assert!(contents.contains("[\n  1.0,\n  2.0\n]"));
        let _ = fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn write_patches_writes_patch_files() {
        let dir = temp_dir("patch_output");
        fs::create_dir_all(&dir).await.unwrap();
        let patches = vec![
            DiffWithPatch {
                path: PathBuf::from("nested/file.txt"),
                patch: "patch-content".into(),
            },
            DiffWithPatch {
                path: PathBuf::from("root.txt"),
                patch: "root".into(),
            },
        ];

        write_patches(&dir, patches).await.unwrap();

        let nested = dir.join("nested").join("file.patch");
        let root = dir.join("root.patch");
        assert!(nested.exists());
        assert!(root.exists());
        assert_eq!(fs::read_to_string(nested).await.unwrap(), "patch-content");
        assert_eq!(fs::read_to_string(root).await.unwrap(), "root");
        let _ = fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn write_dir_output_writes_expected_structure() {
        let dir = temp_dir("dir_output");
        let context = sample_context();

        write_dir_output(&dir, &context).await.unwrap();

        let shell_history = dir.join("shell_history.json");
        let safari_history = dir.join("safari_history.json");
        let repo_dir = dir.join("repo");
        let git_paths = repo_dir.join("git_history_paths.json");
        let commit_log = repo_dir.join("commit_log.json");
        let patch_file = repo_dir.join("foo.patch");

        assert!(shell_history.exists());
        assert!(safari_history.exists());
        assert!(git_paths.exists());
        assert!(commit_log.exists());
        assert!(patch_file.exists());

        // Verify git history paths contains repo_path
        let paths_contents = fs::read_to_string(&git_paths).await.unwrap();
        assert!(paths_contents.contains("\"/repo\""));

        let _ = fs::remove_dir_all(dir).await;
    }

    #[tokio::test]
    async fn write_output_json_writes_single_file() {
        let dir = temp_dir("write_output_json");
        fs::create_dir_all(&dir).await.unwrap();
        let file = dir.join("output.json");
        let context = sample_context();

        write_output(&file, &OutputFormat::Json, &context)
            .await
            .unwrap();

        let contents = fs::read_to_string(&file).await.unwrap();
        assert!(contents.contains("shell_history"));
        let _ = fs::remove_dir_all(dir).await;
    }
}

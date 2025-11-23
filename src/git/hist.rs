use std::collections::HashSet;

use crate::AppResult;
use crate::ai::commit_message::generate_commit_message;
use crate::git::diff::{DiffSummary, get_diff_summary};
use crate::shell::ShellHistoryEntry;
use crate::time_utils::{timestamp_secs_to_nsecs, unix_time_nsec_to_datetime, yesterday};
use async_openai::{Client, config::Config};
use git2::{Commit, DiffOptions, Oid, Repository, Status, StatusOptions, Tree};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use tracing::{debug, info, trace};

#[tracing::instrument(level = "trace")]
fn get_status_opts() -> StatusOptions {
    let mut opts = StatusOptions::new();
    opts.include_untracked(true)
        .include_ignored(false)
        .include_unmodified(true)
        .exclude_submodules(true)
        .recurse_untracked_dirs(true)
        .disable_pathspec_match(false)
        .recurse_ignored_dirs(false)
        .renames_head_to_index(true)
        .renames_index_to_workdir(true)
        .sort_case_sensitively(false)
        .sort_case_insensitively(true)
        .renames_from_rewrites(true)
        .no_refresh(false)
        .update_index(true)
        .include_unreadable(false);
    opts
}

#[tracing::instrument(level = "trace")]
fn get_diff_opts() -> DiffOptions {
    let mut opts = DiffOptions::new();
    opts.reverse(false)
        .include_ignored(false)
        .recurse_ignored_dirs(false)
        .include_untracked(true)
        .recurse_untracked_dirs(true)
        .include_unmodified(true)
        .include_typechange(true)
        .include_typechange_trees(true)
        .ignore_filemode(false)
        .ignore_submodules(false)
        .ignore_case(false)
        .skip_binary_check(false)
        .enable_fast_untracked_dirs(false)
        .update_index(true)
        .include_unreadable(true)
        .include_unreadable_as_untracked(false)
        .force_text(false)
        .force_binary(false)
        .ignore_whitespace(false)
        .ignore_whitespace_change(false)
        .ignore_whitespace_eol(false)
        .ignore_blank_lines(false)
        .show_untracked_content(true)
        .show_unmodified(true)
        .minimal(false)
        .patience(true)
        .show_binary(false)
        .indent_heuristic(true);
    opts
}

fn head_tree_and_parents<'b, 'a: 'b>(
    repo: &'a Repository,
) -> AppResult<(Tree<'b>, Vec<Commit<'b>>)> {
    if let Ok(head) = repo.head()
        && let Some(oid) = head.target()
    {
        let parent = repo.find_commit(oid)?;
        let tree = parent.tree()?;
        return Ok((tree, vec![parent]));
    }
    // Unborn HEAD: create an empty tree and no parents.
    let builder = repo.treebuilder(None)?;
    let empty_tree_id = builder.write()?;
    let empty_tree = repo.find_tree(empty_tree_id)?;
    Ok((empty_tree, Vec::new()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitMeta {
    pub message: String,
    #[serde(with = "crate::serde_helpers::offset_datetime")]
    pub timestamp: OffsetDateTime,
    pub branches: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRepoHistory {
    pub diff: DiffSummary,
    pub commits: Vec<CommitMeta>,
}

#[tracing::instrument(level = "trace", skip(client, repo))]
async fn check_repo_status<C: Config>(client: &Client<C>, repo: &Repository) -> AppResult<()> {
    let mut opts = get_status_opts();

    // You might set opts.include_untracked(true) etc if you care about untracked files
    let statuses = repo.statuses(Some(&mut opts))?;
    let mut staged_changes = false;
    let mut working_dir_changes = false;
    for entry in statuses.iter() {
        let s = entry.status();
        // look for flags that indicate working‐directory changes (vs just staged)
        if s.intersects(
            Status::WT_MODIFIED
                | Status::WT_DELETED
                | Status::WT_NEW
                | Status::WT_TYPECHANGE
                | Status::WT_RENAMED,
        ) {
            // There are working-directory changes
            trace!("Working directory has changes in: {:?}", entry.path());
            working_dir_changes = true;
        }
        if s.intersects(
            Status::INDEX_MODIFIED
                | Status::INDEX_DELETED
                | Status::INDEX_NEW
                | Status::INDEX_TYPECHANGE
                | Status::INDEX_RENAMED,
        ) {
            // There are staged changes
            trace!("Staged changes in: {:?}", entry.path());
            staged_changes = true;
        }
    }
    if !staged_changes && !working_dir_changes {
        debug!("No changes to commit.");
        return Ok(());
    }
    if staged_changes {
        info!(
            "Committing staged directory changes for {}...",
            repo.path().display()
        );
        let (head_tree, parents) = head_tree_and_parents(repo)?;
        let mut index = repo.index()?;
        let diff =
            repo.diff_tree_to_index(Some(&head_tree), Some(&index), Some(&mut get_diff_opts()))?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let commit_message = generate_commit_message(client, &diff, repo).await?;
        let sig = repo.signature()?;
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            commit_message.to_string().as_str(),
            &tree,
            &parents.iter().collect::<Vec<&Commit>>(),
        )?;
        info!("Staged changes committed.");
    }
    if working_dir_changes {
        info!(
            "Committing working directory changes for {}...",
            repo.path().display()
        );
        let (head_tree, parents) = head_tree_and_parents(repo)?;
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;
        let diff =
            repo.diff_tree_to_index(Some(&head_tree), Some(&index), Some(&mut get_diff_opts()))?;
        let commit_message = generate_commit_message(client, &diff, repo).await?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = repo.signature()?;
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            commit_message.to_string().as_str(),
            &tree,
            &parents.iter().collect::<Vec<&Commit>>(),
        )?;
        info!("Working directory changes committed.");
    }
    Ok(())
}

#[tracing::instrument(level = "debug", skip(client, shell_history))]
pub async fn get_git_history<C: Config>(
    client: &Client<C>,
    shell_history: &Vec<ShellHistoryEntry>,
) -> AppResult<Vec<GitRepoHistory>> {
    let mut visited = HashSet::new();
    let yesterday_dt = yesterday();
    let mut git_history = Vec::new();
    for entry in shell_history {
        if visited.contains(&entry.directory) {
            continue;
        }
        visited.insert(entry.directory.clone());
        if let Ok(repo) = Repository::open(&entry.directory) {
            check_repo_status(client, &repo).await?;
            // Refresh state in case check_repo_status created new commits
            if let Err(e) = repo.index().and_then(|mut idx| idx.read(true)) {
                debug!("Failed to refresh index for {:?}: {}", entry.directory, e);
            }
            let mut oldest_commit: (Option<Oid>, Option<Commit>) = (None, None);
            let mut daily_commits: Vec<CommitMeta> = Vec::new();
            debug!(
                "Checking git history for repository in {:?}",
                entry.directory
            );
            // Collect local branch tips and push all to revwalk to cover all branches.
            let mut branch_tips: Vec<(String, Oid)> = Vec::new();
            if let Ok(branches) = repo.branches(Some(git2::BranchType::Local)) {
                for branch in branches.flatten() {
                    if let Ok(name_opt) = branch.0.name()
                        && let (Some(name), Some(target)) = (name_opt, branch.0.get().target())
                    {
                        branch_tips.push((name.to_string(), target));
                    }
                }
            }
            let mut revwalk = match repo.revwalk() {
                Ok(rw) => rw,
                Err(e) => {
                    debug!("Revwalk unavailable for {:?}: {}", entry.directory, e);
                    continue;
                }
            };
            if branch_tips.is_empty() {
                if let Err(e) = revwalk.push_head() {
                    debug!(
                        "No HEAD to walk (likely unborn branch) for {:?}: {}",
                        entry.directory, e
                    );
                    continue;
                }
            } else {
                for (_, tip) in &branch_tips {
                    if let Err(e) = revwalk.push(*tip) {
                        debug!(
                            "Failed to push branch tip {:?} in {:?}: {}",
                            tip, entry.directory, e
                        );
                    }
                }
            }
            for oid in revwalk.flatten() {
                debug!("Found git commit {} in {:?}", oid, entry.directory);
                let commit = repo.find_commit(oid)?;
                trace!("Found commit object: {:?}", commit);
                let time = commit.time();
                let timestamp = unix_time_nsec_to_datetime(timestamp_secs_to_nsecs(time.seconds()));
                if timestamp >= yesterday_dt {
                    let message = commit.message().unwrap_or_default().to_string();
                    let mut branches = Vec::new();
                    for (name, tip) in &branch_tips {
                        if repo.graph_descendant_of(*tip, commit.id()).unwrap_or(false) {
                            branches.push(name.clone());
                        }
                    }
                    daily_commits.push(CommitMeta {
                        message,
                        timestamp,
                        branches,
                    });

                    if let Some(prev_commit) = &oldest_commit.1
                        && commit.time().seconds() < prev_commit.time().seconds()
                    {
                        oldest_commit = (Some(oid), Some(commit.clone()));
                    } else if oldest_commit.1.is_none() {
                        oldest_commit = (Some(oid), Some(commit.clone()));
                    }
                } else {
                    break;
                }
            }
            if let (Some(_), Some(commit)) = oldest_commit {
                let head = repo.head()?;
                let head_tree = head.peel_to_tree()?;
                let commit_tree = commit.tree()?;
                let diff = repo.diff_tree_to_tree(
                    Some(&commit_tree),
                    Some(&head_tree),
                    Some(&mut get_diff_opts()),
                )?;
                let repo_path = repo.path().parent().unwrap();
                if let Ok(diff_summary) = get_diff_summary(repo_path, &diff) {
                    git_history.push(GitRepoHistory {
                        diff: diff_summary,
                        commits: daily_commits.clone(),
                    });
                }
            }
        }
    }
    Ok(git_history)
}

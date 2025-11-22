use crate::AppResult;
use crate::ai::generate_commit_message;
use crate::git::diff::{DiffSummary, get_diff_summary};
use crate::shell::ShellHistoryEntry;
use crate::time_utils::{unix_time_nsec_to_datetime, yesterday};
use async_openai::{Client, config::Config};
use git2::{Commit, DiffOptions, Oid, Repository, Status, StatusOptions};
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
        let head = repo.head()?;
        let head_tree = head.peel_to_tree()?;
        let mut index = repo.index()?;
        let diff =
            repo.diff_tree_to_index(Some(&head_tree), Some(&index), Some(&mut get_diff_opts()))?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let commit_message = generate_commit_message(client, &diff, repo).await?;
        let parent_commit = head.peel_to_commit()?;
        let sig = repo.signature()?;
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            commit_message.to_string().as_str(),
            &tree,
            &[&parent_commit],
        )?;
        info!("Staged changes committed.");
    }
    if working_dir_changes {
        info!(
            "Committing working directory changes for {}...",
            repo.path().display()
        );
        let head = repo.head()?;
        let head_tree = head.peel_to_tree()?;
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), git2::IndexAddOption::DEFAULT, None)?;
        index.write()?;
        let diff =
            repo.diff_tree_to_index(Some(&head_tree), Some(&index), Some(&mut get_diff_opts()))?;
        let commit_message = generate_commit_message(client, &diff, repo).await?;
        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let parent_commit = head.peel_to_commit()?;
        let sig = repo.signature()?;
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            commit_message.to_string().as_str(),
            &tree,
            &[&parent_commit],
        )?;
        info!("Working directory changes committed.");
    }
    Ok(())
}

#[tracing::instrument(level = "debug", skip(client, shell_history))]
pub async fn get_git_history<C: Config>(
    client: &Client<C>,
    shell_history: &Vec<ShellHistoryEntry>,
) -> AppResult<Vec<DiffSummary>> {
    let yesterday_dt = yesterday();
    let mut git_history = Vec::new();
    for entry in shell_history {
        if let Ok(repo) = Repository::open(&entry.directory) {
            check_repo_status(client, &repo).await?;
            let mut oldest_commit: (Option<Oid>, Option<Commit>) = (None, None);
            debug!(
                "Checking git history for repository in {:?}",
                entry.directory
            );
            for oid in repo.revwalk()?.flatten() {
                debug!("Found git commit {} in {:?}", oid, entry.directory);
                let commit = repo.find_commit(oid)?;
                trace!("Found commit object: {:?}", commit);
                let time = commit.time();
                if unix_time_nsec_to_datetime(time.seconds(), 0) >= yesterday_dt {
                    if let Some(prev_commit) = &oldest_commit.1
                        && commit.time().seconds() < prev_commit.time().seconds()
                    {
                        oldest_commit = (Some(oid), Some(commit));
                    } else if oldest_commit.1.is_none() {
                        oldest_commit = (Some(oid), Some(commit));
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
                get_diff_summary(repo_path, &diff).map(|diff_summary| {
                    git_history.push(diff_summary);
                })?;
            }
        }
    }
    Ok(git_history)
}

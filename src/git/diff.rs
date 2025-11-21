use crate::AppResult;
use git2::{Delta, Diff, DiffDelta, DiffFormat, DiffHunk, DiffLine, Patch, Repository};
use log::error;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::{fmt::Write, path::PathBuf, str};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Hash)]
pub struct DiffFromTo {
    pub from: PathBuf,
    pub to: PathBuf,
}

pub fn get_file(
    repo: &Repository,
    diff: &Diff,
    path: &PathBuf,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> Option<String> {
    // To read file contents we need repository ownership.

    let mut chosen: Option<DiffDelta> = None;
    for delta in diff.deltas() {
        let matches_new = delta
            .new_file()
            .path()
            .map(|p| p == path.as_path())
            .unwrap_or(false);
        let matches_old = delta
            .old_file()
            .path()
            .map(|p| p == path.as_path())
            .unwrap_or(false);
        if matches_new || matches_old {
            chosen = Some(delta);
            break;
        }
    }

    let delta = chosen?;
    // Prefer the new file content; fall back to old for deletions.
    let blob_id = delta.new_file().id();

    let blob = repo.find_blob(blob_id).ok()?;
    let content = blob.content();
    let text = match std::str::from_utf8(content) {
        Ok(s) => s,
        Err(e) => {
            error!("Non-utf8 content for {:?}: {}", path, e);
            return None;
        }
    };

    if start_line.is_none() && end_line.is_none() {
        return Some(text.to_string());
    }

    let lines: Vec<&str> = text.split('\n').collect();
    let start = start_line.unwrap_or(1);
    let end = end_line.unwrap_or(lines.len());
    if start == 0 || start > end {
        return None;
    }
    let slice_start = start.saturating_sub(1);
    let slice_end = end.min(lines.len());
    let mut out = lines[slice_start..slice_end].join("\n");
    // Re-add trailing newline if the original had one and we sliced to the end.
    if text.ends_with('\n') && slice_end == lines.len() {
        out.push('\n');
    }
    Some(out)
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct DiffWithPatch {
    pub path: PathBuf,
    pub patch: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct DiffSummary {
    pub repo_path: PathBuf,
    pub unmodified: HashSet<PathBuf>,
    pub added: Vec<DiffWithPatch>,
    pub deleted: HashSet<PathBuf>,
    pub modified: Vec<DiffWithPatch>,
    pub renamed: HashSet<DiffFromTo>,
    pub copied: HashSet<DiffFromTo>,
    pub untracked: Vec<DiffWithPatch>,
    pub typechange: HashSet<PathBuf>,
    pub unreadable: HashSet<PathBuf>,
    pub conflicted: HashSet<PathBuf>,
}

impl DiffFromTo {
    pub fn from_delta(delta: &DiffDelta) -> Self {
        DiffFromTo {
            from: match delta.old_file().path() {
                Some(p) => p.to_path_buf(),
                None => PathBuf::from("unknown"),
            },
            to: match delta.new_file().path() {
                Some(p) => p.to_path_buf(),
                None => PathBuf::from("unknown"),
            },
        }
    }
}

impl DiffWithPatch {
    /// Append a diff line to the accumulated patch buffer for a file, adding headers as needed.
    /// Call this repeatedly for all lines of a delta to build a full patch string.
    pub fn append_line(
        delta: &DiffDelta,
        hunk: Option<DiffHunk>,
        line: &DiffLine,
        buf: &mut String,
        last_hunk: &mut Option<(u32, u32, u32, u32)>,
    ) {
        if buf.is_empty() {
            let old_path = delta
                .old_file()
                .path()
                .map(|p| format!("a/{}", p.display()))
                .unwrap_or_else(|| "a/unknown".to_string());
            let new_path = delta
                .new_file()
                .path()
                .map(|p| format!("b/{}", p.display()))
                .unwrap_or_else(|| "b/unknown".to_string());
            let _ = writeln!(buf, "--- {}", old_path);
            let _ = writeln!(buf, "+++ {}", new_path);
        }

        if let Some(h) = hunk {
            let range = (h.old_start(), h.old_lines(), h.new_start(), h.new_lines());
            if last_hunk.as_ref() != Some(&range) {
                *last_hunk = Some(range);
                let _ = writeln!(
                    buf,
                    "@@ -{},{} +{},{} @@",
                    h.old_start(),
                    h.old_lines(),
                    h.new_start(),
                    h.new_lines()
                );
            }
        }

        let origin = line.origin();
        let content = match str::from_utf8(line.content()) {
            Ok(content) => content,
            Err(e) => {
                error!("Error reading diff line content: {}", e);
                ""
            }
        };
        let _ = write!(buf, "{}{}", origin, content);
    }
}

fn get_filename(delta: &DiffDelta) -> PathBuf {
    match delta.new_file().path() {
        Some(p) => p.to_path_buf(),
        None => match delta.old_file().path() {
            Some(p) => p.to_path_buf(),
            None => PathBuf::from("unknown"),
        },
    }
}

type PatchCollector = HashMap<PathBuf, (String, Option<(u32, u32, u32, u32)>)>;

pub fn get_diff_summary<P: AsRef<Path>>(repo_path: P, diff: &Diff) -> AppResult<DiffSummary> {
    // Accumulate per-path patch strings and hunk state to avoid duplicating headers.
    let mut added_patches: PatchCollector = HashMap::new();
    let mut modified_patches: PatchCollector = HashMap::new();
    let mut untracked_patches: PatchCollector = HashMap::new();

    let mut summary = DiffSummary {
        repo_path: repo_path.as_ref().to_path_buf(),
        unmodified: HashSet::new(),
        added: Vec::new(),
        deleted: HashSet::new(),
        modified: Vec::new(),
        renamed: HashSet::new(),
        copied: HashSet::new(),
        untracked: Vec::new(),
        typechange: HashSet::new(),
        unreadable: HashSet::new(),
        conflicted: HashSet::new(),
    };
    diff.print(DiffFormat::Patch, |delta, hunk, line| {
        let path = get_filename(&delta);
        match delta.status() {
            Delta::Added => {
                let (buf, last_hunk) = added_patches
                    .entry(path.clone())
                    .or_insert_with(|| (String::new(), None));
                DiffWithPatch::append_line(&delta, hunk, &line, buf, last_hunk);
            }
            Delta::Deleted => {
                summary.deleted.insert(path);
            }
            Delta::Modified => {
                let (buf, last_hunk) = modified_patches
                    .entry(path.clone())
                    .or_insert_with(|| (String::new(), None));
                DiffWithPatch::append_line(&delta, hunk, &line, buf, last_hunk);
            }
            Delta::Renamed => {
                summary.renamed.insert(DiffFromTo::from_delta(&delta));
            }
            Delta::Copied => {
                summary.copied.insert(DiffFromTo::from_delta(&delta));
            }
            Delta::Untracked => {
                let (buf, last_hunk) = untracked_patches
                    .entry(path.clone())
                    .or_insert_with(|| (String::new(), None));
                DiffWithPatch::append_line(&delta, hunk, &line, buf, last_hunk);
            }
            Delta::Typechange => {
                summary.typechange.insert(path);
            }
            Delta::Unreadable => {
                summary.unreadable.insert(path);
            }
            Delta::Conflicted => {
                summary.conflicted.insert(path);
            }
            Delta::Unmodified => {
                summary.unmodified.insert(path);
            }
            Delta::Ignored => { /* Ignore ignored files */ }
        };
        true
    })?;

    summary.added = added_patches
        .into_iter()
        .map(|(path, (patch, _))| DiffWithPatch { path, patch })
        .collect();
    summary.modified = modified_patches
        .into_iter()
        .map(|(path, (patch, _))| DiffWithPatch { path, patch })
        .collect();
    summary.untracked = untracked_patches
        .into_iter()
        .map(|(path, (patch, _))| DiffWithPatch { path, patch })
        .collect();

    Ok(summary)
}

fn line_in_range(
    start: Option<u32>,
    end: Option<u32>,
    old_lineno: Option<u32>,
    new_lineno: Option<u32>,
) -> bool {
    if start.is_none() && end.is_none() {
        return true;
    }
    let within = |n: u32| -> bool {
        let lower_ok = start.map(|s| n >= s).unwrap_or(true);
        let upper_ok = end.map(|e| n <= e).unwrap_or(true);
        lower_ok && upper_ok
    };
    if let Some(n) = new_lineno {
        return within(n);
    }
    if let Some(n) = old_lineno {
        return within(n);
    }
    false
}

pub fn get_patch(
    diff: &Diff,
    path: &PathBuf,
    start_line: Option<u32>,
    end_line: Option<u32>,
) -> Option<String> {
    for (idx, delta) in diff.deltas().enumerate() {
        let matches_new = delta
            .new_file()
            .path()
            .map(|p| p == path.as_path())
            .unwrap_or(false);
        let matches_old = delta
            .old_file()
            .path()
            .map(|p| p == path.as_path())
            .unwrap_or(false);

        if !(matches_new || matches_old) {
            continue;
        }

        // Build a textual patch for just this delta.
        match Patch::from_diff(diff, idx) {
            Ok(Some(mut patch)) => {
                let mut rendered = String::new();
                // Add file headers for readability.
                let old_path = delta
                    .old_file()
                    .path()
                    .map(|p| format!("a/{}", p.to_string_lossy()))
                    .unwrap_or_else(|| "a/unknown".to_string());
                let new_path = delta
                    .new_file()
                    .path()
                    .map(|p| format!("b/{}", p.to_string_lossy()))
                    .unwrap_or_else(|| "b/unknown".to_string());
                let _ = writeln!(rendered, "--- {}", old_path);
                let _ = writeln!(rendered, "+++ {}", new_path);

                let mut last_hunk: Option<(u32, u32, u32, u32)> = None;
                let _ = patch.print(&mut |_, hunk, line| {
                    if !line_in_range(start_line, end_line, line.old_lineno(), line.new_lineno()) {
                        return true;
                    }

                    if let Some(h) = hunk {
                        let range = (h.old_start(), h.old_lines(), h.new_start(), h.new_lines());
                        if last_hunk != Some(range) {
                            last_hunk = Some(range);
                            let _ = writeln!(
                                rendered,
                                "@@ -{},{} +{},{} @@",
                                h.old_start(),
                                h.old_lines(),
                                h.new_start(),
                                h.new_lines()
                            );
                        }
                    }

                    let origin = line.origin();
                    let content = match str::from_utf8(line.content()) {
                        Ok(content) => content,
                        Err(e) => {
                            error!("Error reading diff line content: {}", e);
                            ""
                        }
                    };
                    let _ = write!(rendered, "{}{}", origin, content);
                    true
                });

                return Some(rendered);
            }
            Ok(None) => {
                // Binary or otherwise unavailable.
                return None;
            }
            Err(e) => {
                error!("Failed to build patch for {:?}: {}", path, e);
                return None;
            }
        }
    }

    None
}

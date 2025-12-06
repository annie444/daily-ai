use std::env;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sqlx::sqlite::SqlitePoolOptions;
use time::{Duration, OffsetDateTime};
use tracing::{debug, trace};

use crate::AppResult;
use crate::time_utils::{macos_past_ts, macos_to_datetime};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SafariHistoryItem {
    pub url: String,
    pub title: Option<String>,
    pub visit_count: i64,
    #[serde(with = "crate::serde_helpers::offset_datetime")]
    pub last_visited: OffsetDateTime,
}

/// Return true if a candidate path points to an existing file.
fn valid_db_path(path: &Path) -> bool {
    path.exists() && path.is_file()
}

/// Resolve the Safari History.db path from several common locations and overrides.
#[tracing::instrument(
    name = "Searching for the Safari history database file",
    level = "info"
)]
fn get_safari_history_db_path() -> PathBuf {
    let candidate = |p: PathBuf| if valid_db_path(&p) { Some(p) } else { None };

    candidate(
        env::current_dir()
            .map(|p| p.join("History.db"))
            .unwrap_or_default(),
    )
    .or_else(|| {
        env::var("SAFARI_HISTORY_DB_PATH")
            .ok()
            .and_then(|p| candidate(p.into()))
    })
    // Deprecated but kept for compatibility on older toolchains.
    .or_else(|| env::home_dir().and_then(|home| candidate(home.join("Library/Safari/History.db"))))
    .or_else(|| {
        env::var("HOME")
            .ok()
            .and_then(|home| candidate(PathBuf::from(home).join("Library/Safari/History.db")))
    })
    .or_else(|| {
        env::var("USERPROFILE")
            .ok()
            .and_then(|home| candidate(PathBuf::from(home).join("Library/Safari/History.db")))
    })
    .unwrap_or_else(|| PathBuf::from("/Users/username/Library/Safari/History.db"))
}

/// Fetch Safari history entries from the past 24 hours (UTC) ordered by most recent visit.
#[tracing::instrument(name = "Fetching the Safari history", level = "info")]
pub async fn get_safari_history(duration: &Duration) -> AppResult<Vec<SafariHistoryItem>> {
    let db_path = get_safari_history_db_path();
    let conn_str = format!("sqlite://{}?mode=ro", db_path.display()); // Read-only mode
    trace!("Connecting to Safari History database at {}", conn_str);

    let pool = SqlitePoolOptions::new()
        .connect(&conn_str)
        .await
        .map_err(|e| {
            crate::error::AppError::Other(format!("Failed to connect to Safari DB: {e}"))
        })?;

    trace!("Connected to Safari History database");

    let past_date = macos_past_ts(duration);

    // We group by item ID to get unique URLs, taking the max visit time (latest).
    // Note: 'visit_count' is in history_items.
    let rows = sqlx::query_as::<_, (String, Option<String>, i64, f64)>(
        r#"
        SELECT 
            i.url, 
            v.title, 
            i.visit_count, 
            MAX(v.visit_time) as visit_time
        FROM history_items i
        JOIN history_visits v ON i.id = v.history_item
        WHERE v.visit_time > ?
        GROUP BY i.id
        ORDER BY visit_time DESC
        "#,
    )
    .bind(past_date)
    .fetch_all(&pool)
    .await
    .map_err(|e| crate::error::AppError::Other(format!("Failed to query Safari history: {e}")))?;

    debug!("Fetched {} history items", rows.len());

    trace!("Processing Safari history items");

    let safari_history: Vec<SafariHistoryItem> = rows
        .into_iter()
        .filter(|(url, _, _, _)| {
            let mut url = url.to_lowercase();
            url = url.replace("https://", "");
            url = url.replace("http://", "");
            let domain = url.rsplit_once('/').map(|(base, _)| base).unwrap_or(&url);
            let (domain, path) = domain.split_once('/').unwrap_or((domain, ""));
            !domain.contains("oauth")
                && !domain.contains("login")
                && !path.contains("auth")
                && !path.contains("signin")
                && !domain.contains("sso")
                && !path.contains("callback")
                && !domain.contains("duosecurity")
        })
        .map(|(url, title, visit_count, visit_time)| {
            let last_visited = macos_to_datetime(visit_time);
            SafariHistoryItem {
                url,
                title,
                visit_count,
                last_visited,
            }
        })
        .collect();

    trace!("Completed processing Safari history items");

    Ok(safari_history)
}

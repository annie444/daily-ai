use std::env;
use std::path::{Path, PathBuf};

use sea_orm::{
    ColumnTrait, ConnectOptions, Database, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder,
};
use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};
use tracing::{debug, trace};

use crate::AppResult;
use crate::entity::{history_items, history_visits};
use crate::time_utils::{datetime_to_macos_time, macos_past_ts, macos_to_datetime, midnight_utc};

/// Minimal subset of Safari history we need for downstream processing.
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

/// Open the Safari history sqlite database at the provided path.
#[tracing::instrument(name = "Connecting to the Safari history database", level = "info")]
async fn connect_to_db<P: AsRef<Path> + std::fmt::Debug>(
    db_path: P,
) -> AppResult<DatabaseConnection> {
    let opt = ConnectOptions::new(format!("sqlite://{}", db_path.as_ref().display()));
    trace!("Connecting to Safari History database");
    let db = Database::connect(opt).await?;
    Ok(db)
}

/// Fetch Safari history entries from the past 24 hours (UTC) ordered by most recent visit.
#[tracing::instrument(name = "Fetching the Safari history", level = "info")]
pub async fn get_safari_history(duration: &Duration) -> AppResult<Vec<SafariHistoryItem>> {
    let db_path = get_safari_history_db_path();
    let db = connect_to_db(db_path).await?;

    trace!("Connected to Safari History database");

    let past_date = macos_past_ts(duration);
    let history_items = history_items::Entity::find()
        .find_with_related(history_visits::Entity)
        .filter(history_visits::Column::VisitTime.gt(past_date))
        .order_by_desc(history_visits::Column::VisitTime)
        .all(&db)
        .await?;

    debug!("Fetched {} history items", history_items.len());

    trace!("Processing Safari history items");

    let mid = midnight_utc();
    let mid_macos = datetime_to_macos_time(&mid);

    let safari_history = history_items
        .into_iter()
        .map(|(item, visits)| {
            // Use the first visit (most recent, due to order_by_desc) to drive title and timestamp.
            let last_visited = visits.first().map_or(mid, |visit| {
                macos_to_datetime(TryInto::<f64>::try_into(visit.visit_time).unwrap_or(mid_macos))
            });
            SafariHistoryItem {
                url: item.url,
                title: visits.first().and_then(|visit| visit.title.clone()),
                visit_count: item.visit_count,
                last_visited,
            }
        })
        .collect();

    trace!("Completed processing Safari history items");

    Ok(safari_history)
}

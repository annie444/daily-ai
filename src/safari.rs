use crate::AppResult;
use crate::entity::{history_items, history_visits};
use crate::time_utils::{datetime_to_macos_time, macos_to_datetime, macos_yesterday, midnight_utc};
use sea_orm::{
    ColumnTrait, ConnectOptions, Database, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};
use time::OffsetDateTime;
use tracing::{debug, trace};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SafariHistoryItem {
    pub url: String,
    pub title: Option<String>,
    pub visit_count: i64,
    #[serde(with = "crate::serde_helpers::offset_datetime")]
    pub last_visited: OffsetDateTime,
}

#[tracing::instrument(level = "trace")]
fn get_safari_history_db_path() -> PathBuf {
    if let Ok(curpath) = env::current_dir()
        && curpath.join("History.db").exists()
        && curpath.join("History.db").is_file()
    {
        curpath.join("History.db")
    } else if let Ok(env_path) = env::var("SAFARI_HISTORY_DB_PATH")
        && PathBuf::from(&env_path).exists()
        && PathBuf::from(&env_path).is_file()
    {
        PathBuf::from(env_path)
    } else if let Some(home) = env::home_dir()
        && home.join("Library/Safari/History.db").exists()
        && home.join("Library/Safari/History.db").is_file()
    {
        home.join("Library/Safari/History.db")
    } else if let Ok(home) = env::var("HOME")
        && PathBuf::from(&home)
            .join("Library/Safari/History.db")
            .exists()
        && PathBuf::from(&home)
            .join("Library/Safari/History.db")
            .is_file()
    {
        PathBuf::from(home).join("Library/Safari/History.db")
    } else if let Ok(user_profile) = env::var("USERPROFILE")
        && PathBuf::from(&user_profile)
            .join("Library/Safari/History.db")
            .exists()
        && PathBuf::from(&user_profile)
            .join("Library/Safari/History.db")
            .is_file()
    {
        PathBuf::from(user_profile).join("Library/Safari/History.db")
    } else {
        PathBuf::from("/Users/username/Library/Safari/History.db")
    }
}

#[tracing::instrument(level = "trace")]
async fn connect_to_db<P: AsRef<Path> + std::fmt::Debug>(
    db_path: P,
) -> AppResult<DatabaseConnection> {
    let opt = ConnectOptions::new(format!("sqlite://{}", db_path.as_ref().display()));
    trace!("Connecting to Safari History database");
    let db = Database::connect(opt).await?;
    Ok(db)
}

#[tracing::instrument(level = "debug")]
pub async fn get_safari_history() -> AppResult<Vec<SafariHistoryItem>> {
    let db_path = get_safari_history_db_path();
    let db = connect_to_db(db_path).await?;

    trace!("Connected to Safari History database");

    let yesterday = macos_yesterday();
    let history_items = history_items::Entity::find()
        .find_with_related(history_visits::Entity)
        .filter(history_visits::Column::VisitTime.gt(yesterday))
        .order_by_desc(history_visits::Column::VisitTime)
        .all(&db)
        .await?;

    debug!("Fetched {} history items", history_items.len());

    let mut safari_history = Vec::new();

    trace!("Processing Safari history items");

    let mid = midnight_utc();
    let mid_macos = datetime_to_macos_time(&mid);

    for (item, visits) in history_items {
        let safari_item = SafariHistoryItem {
            url: item.url,
            title: visits.first().and_then(|visit| visit.title.clone()),
            visit_count: item.visit_count,
            last_visited: visits.first().map_or(mid, |visit| {
                macos_to_datetime(TryInto::<f64>::try_into(visit.visit_time).unwrap_or(mid_macos))
            }),
        };
        safari_history.push(safari_item);
    }

    trace!("Completed processing Safari history items");

    Ok(safari_history)
}

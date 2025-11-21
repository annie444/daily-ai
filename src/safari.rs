use crate::AppResult;
use crate::entity::{history_items, history_visits};
use crate::time_utils::{datetime_to_macos_time, macos_to_datetime, macos_yesterday, midnight};
use chrono::NaiveDateTime;
use log::LevelFilter;
use log::{debug, trace};
use sea_orm::{
    ColumnTrait, ConnectOptions, Database, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct SafariHistoryItem {
    pub id: i64,
    pub url: String,
    pub title: Option<String>,
    pub visit_count: i64,
    pub visit_count_score: i64,
    #[serde(with = "crate::serde_helpers::naive_datetime")]
    pub last_visited: NaiveDateTime,
}

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

async fn connect_to_db<P: AsRef<Path>>(db_path: P) -> AppResult<DatabaseConnection> {
    let mut opt = ConnectOptions::new(format!("sqlite://{}", db_path.as_ref().display()));
    opt.sqlx_logging(true)
        .sqlx_logging_level(LevelFilter::Debug);
    trace!("Connecting to Safari History database");
    let db = Database::connect(opt).await?;
    Ok(db)
}

pub async fn get_safari_history() -> AppResult<Vec<SafariHistoryItem>> {
    let db_path = get_safari_history_db_path();
    let db = connect_to_db(db_path).await?;

    trace!("Connected to Safari History database");

    let history_items = history_items::Entity::find()
        .find_with_related(history_visits::Entity)
        .filter(history_visits::Column::VisitTime.gt(macos_yesterday()))
        .order_by_desc(history_visits::Column::VisitTime)
        .all(&db)
        .await?;

    debug!("Fetched {} history items", history_items.len());

    let mut safari_history = Vec::new();

    trace!("Processing Safari history items");

    for (item, visits) in history_items {
        let safari_item = SafariHistoryItem {
            id: item.id,
            url: item.url,
            title: visits.first().and_then(|visit| visit.title.clone()),
            visit_count: item.visit_count,
            visit_count_score: item.visit_count_score,
            last_visited: visits.first().map_or(midnight(), |visit| {
                macos_to_datetime(
                    visit
                        .visit_time
                        .to_string()
                        .parse::<f64>()
                        .unwrap_or(datetime_to_macos_time(&midnight())),
                )
            }),
        };
        safari_history.push(safari_item);
    }

    trace!("Completed processing Safari history items");

    Ok(safari_history)
}

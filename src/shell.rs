use std::path::PathBuf;

use atuin_client::{
    database::{Database, Sqlite},
    encryption,
    history::{History, store::HistoryStore},
    record::{sqlite_store::SqliteStore, store::Store, sync},
    settings::{FilterMode, Settings},
};
use atuin_common::record::RecordId;
use atuin_dotfiles::store::{AliasStore, var::VarStore};
use atuin_kv::store::KvStore;
use atuin_scripts::store::ScriptStore;
use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};
use tracing::{debug, info};

use crate::AppResult;
use crate::error::AppError;

/// Represents a single shell command execution retrieved from Atuin.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShellHistoryEntry {
    #[serde(with = "crate::serde_helpers::offset_datetime")]
    pub date_time: OffsetDateTime,
    #[serde(with = "crate::serde_helpers::duration")]
    pub duration: Duration,
    pub host: String,
    pub directory: PathBuf,
    pub command: String,
    pub exit_code: i64,
    pub session_id: String,
}

impl From<&History> for ShellHistoryEntry {
    /// Convert an Atuin history record into our internal serializable shape.
    fn from(history: &History) -> Self {
        ShellHistoryEntry {
            date_time: history.timestamp,
            duration: Duration::nanoseconds(std::cmp::max(history.duration, 0) as i64),
            host: history.hostname.clone(),
            directory: PathBuf::from(&history.cwd),
            command: history.command.clone(),
            exit_code: history.exit,
            session_id: history.session.clone(),
        }
    }
}

/// Rebuild all Atuin stores after sync to ensure indexes are consistent.
#[tracing::instrument(
    name = "Rebuilding Atuin databases after history sync",
    level = "info",
    skip_all
)]
async fn rebuild(
    encryption_key: [u8; 32],
    settings: &Settings,
    store: &SqliteStore,
    db: &dyn Database,
    downloaded: Option<&[RecordId]>,
) -> AppResult<()> {
    let host_id = Settings::host_id().expect("failed to get host_id");

    let downloaded = downloaded.unwrap_or(&[]);

    let kv_db = atuin_kv::database::Database::new(settings.kv.db_path.clone(), 1.0).await?;

    let history_store = HistoryStore::new(store.clone(), host_id, encryption_key);
    let alias_store = AliasStore::new(store.clone(), host_id, encryption_key);
    let var_store = VarStore::new(store.clone(), host_id, encryption_key);
    let kv_store = KvStore::new(store.clone(), kv_db, host_id, encryption_key);
    let script_store = ScriptStore::new(store.clone(), host_id, encryption_key);

    history_store
        .incremental_build(db, downloaded)
        .await
        .map_err(|e| {
            AppError::AtuinClient(format!(
                "Unable to rebuild the atuin history database: {}",
                e
            ))
        })?;

    alias_store.build().await.map_err(|e| {
        AppError::AtuinClient(format!("Unable to rebuild the atuin alias database: {}", e))
    })?;
    var_store.build().await.map_err(|e| {
        AppError::AtuinClient(format!(
            "Unable to rebuild the atuin variables database: {}",
            e
        ))
    })?;
    kv_store.build().await.map_err(|e| {
        AppError::AtuinClient(format!(
            "Unable to rebuild the atuin key-value database: {}",
            e
        ))
    })?;

    let script_db =
        atuin_scripts::database::Database::new(settings.scripts.db_path.clone(), 1.0).await?;
    script_store.build(script_db).await.map_err(|e| {
        AppError::AtuinClient(format!(
            "Unable to rebuild the atuin scripts database: {}",
            e
        ))
    })?;
    Ok(())
}

/// Sync the local history with the remote Atuin service, optionally rebuilding
/// local indexes when the record store is out of date.
#[tracing::instrument(
    name = "Syncing shell history with the Atuin server",
    level = "info",
    skip_all
)]
async fn sync_history<D: Database>(
    settings: &Settings,
    store: &SqliteStore,
    db: &D,
) -> AppResult<()> {
    if settings.sync.records {
        debug!("History recording is enabled; Syncing before fetching history");
        let encryption_key: [u8; 32] = encryption::load_key(settings)
            .map_err(|e| {
                AppError::AtuinClient(format!("Unable to fetch encryption key. Got error: {}", e))
            })?
            .into();
        let host_id = Settings::host_id().expect("failed to get host_id");
        let history_store = HistoryStore::new(store.clone(), host_id, encryption_key);

        let (uploaded, downloaded) = sync::sync(settings, store).await.map_err(|e| {
            AppError::AtuinClient(format!("Unable to sync shell history records: {}", e))
        })?;

        // Newly downloaded records might not be reflected in the local stores yet.
        rebuild(encryption_key, settings, store, db, Some(&downloaded)).await?;

        info!("{uploaded}/{} up/down to record store", downloaded.len());

        let history_length = db.history_count(true).await?;
        let store_history_length = store.len_tag("history").await.map_err(|e| {
            AppError::AtuinClient(format!(
                "Unable to get the length of the atuin history db: {}",
                e
            ))
        })?;
        #[allow(clippy::cast_sign_loss)]
        if history_length as u64 > store_history_length {
            info!("{history_length} in history index, but {store_history_length} in history store");
            info!("Running automatic history store init...");

            // Internally we use the global filter mode, so this context is ignored.
            // Don't recurse or loop here—init_store already pulls records into the store.
            history_store.init_store(db).await.map_err(|e| {
                AppError::AtuinClient(format!("Unable to initialize the history store: {}", e))
            })?;

            info!("Re-running sync due to new records locally");

            // we'll want to run sync once more, as there will now be stuff to upload
            let (uploaded, downloaded) = sync::sync(settings, store).await.map_err(|e| {
                AppError::AtuinClient(format!("Unable to sync atuin history database: {}", e))
            })?;

            rebuild(encryption_key, settings, store, db, Some(&downloaded)).await?;

            info!("{uploaded}/{} up/down to record store", downloaded.len());
        }
    } else {
        atuin_client::sync::sync(settings, false, db)
            .await
            .map_err(|e| AppError::AtuinClient(format!("Unable to sync atuin database: {}", e)))?;
    }
    Ok(())
}

/// Filter out deleted entries and those older than 24 hours.
#[tracing::instrument(name = "Filtering recent history", level = "info")]
fn filter_recent_history(records: &[History], duration: &Duration) -> Vec<ShellHistoryEntry> {
    let cutoff = OffsetDateTime::now_utc().saturating_sub(*duration);
    records
        .iter()
        .filter_map(|record| {
            if record.deleted_at.is_some() || record.timestamp < cutoff {
                None
            } else {
                Some(record.into())
            }
        })
        .collect()
}

/// Convert the Atuin sqlite + record store into a history iterator.
#[tracing::instrument(name = "Collecting shell history", level = "info")]
pub async fn get_history(sync: bool, duration: &Duration) -> AppResult<Vec<ShellHistoryEntry>> {
    let settings = Settings::new().map_err(|e| AppError::Other(e.to_string()))?;

    let db_path = PathBuf::from(settings.db_path.as_str());
    let record_store_path = PathBuf::from(settings.record_store_path.as_str());

    // The sqlite DB holds history rows; the record store holds encrypted blobs.
    let db = Sqlite::new(db_path, settings.local_timeout).await?;
    let store = SqliteStore::new(record_store_path, settings.local_timeout)
        .await
        .map_err(|e| AppError::AtuinClient(format!("Unable to open the sqlite store: {0}", e)))?;

    if sync {
        sync_history(&settings, &store, &db).await?;
    }

    // Use both default and global contexts to capture commands executed in any shell session.
    let history = db
        .list(
            &[settings.default_filter_mode(), FilterMode::Global],
            &atuin_client::database::current_context(),
            None,
            false,
            false,
        )
        .await?;

    Ok(filter_recent_history(&history, duration))
}

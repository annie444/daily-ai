pub(crate) mod ai;
mod context;
pub(crate) mod entity;
mod error;
pub(crate) mod git;
mod logging;
pub(crate) mod safari;
pub(crate) mod serde_helpers;
pub(crate) mod shell;
pub(crate) mod time_utils;

pub(crate) use error::{AppError, AppResult};

use std::process::exit;
use tracing::{error, info};
use tracing_unwrap::ResultExt;

#[tokio::main]
async fn main() {
    logging::setup_logger();

    let client = ai::get_lm_studio_client("localhost", 1234);

    let shell_history = match shell::get_history().await {
        Ok(history) => history,
        Err(e) => {
            error!("Error retrieving shell history: {}", e);
            exit(1);
        }
    };

    let safari_history = match safari::get_safari_history().await {
        Ok(history) => history,
        Err(e) => {
            error!("Error retrieving Safari history: {}", e);
            exit(1);
        }
    };

    let commit_history = match git::get_git_history(&client, &shell_history).await {
        Ok(history) => history,
        Err(e) => {
            error!("Error retrieving Git history: {}", e);
            exit(1);
        }
    };

    let combined_hist = context::Context {
        shell_history,
        safari_history,
        commit_history,
    };
    info!("{}", serde_json::to_string(&combined_hist).unwrap_or_log());
}

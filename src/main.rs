pub(crate) mod ai;
mod cli;
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

use clap::Parser;
use std::process::exit;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tracing::{error, info};
use tracing_unwrap::ResultExt;

#[tokio::main]
async fn main() {
    let args = cli::Cli::parse();

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

    let hist_str = serde_json::to_string_pretty(&combined_hist).unwrap_or_log();

    if let Some(output) = &args.output {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(output)
            .await
            .unwrap_or_log();
        file.write(hist_str.as_bytes()).await.unwrap_or_log();
        file.flush().await.unwrap_or_log();
        file.sync_all().await.unwrap_or_log();
        info!("Summary written to {}.", output.display());
    } else {
        info!("Combined History:");
        info!("{}", hist_str);
    }
}

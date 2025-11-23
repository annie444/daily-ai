pub(crate) mod ai;
pub(crate) mod classify;
pub(crate) mod cli;
mod context;
pub(crate) mod dirs;
pub(crate) mod entity;
mod error;
pub(crate) mod git;
mod io_utils;
mod logging;
pub(crate) mod safari;
pub(crate) mod serde_helpers;
pub(crate) mod shell;
pub(crate) mod time_utils;

pub(crate) use error::AppResult;

use clap::Parser;
use std::process::exit;
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

    let urls = match classify::embed_urls(&client, safari_history).await {
        Ok(urls) => urls,
        Err(e) => {
            error!("Error classifying Safari history: {}", e);
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
        safari_history: urls.clusters,
        commit_history,
    };

    let hist_str = serde_json::to_string_pretty(&combined_hist).unwrap_or_log();

    if let Some(output) = &args.output {
        io_utils::write_output(output, &args.format, &combined_hist)
            .await
            .unwrap_or_log();
    } else {
        info!("Combined History:");
        info!("{}", hist_str);
    }
}

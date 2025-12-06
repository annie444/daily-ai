pub(crate) mod ai;
pub(crate) mod classify;
pub(crate) mod cli;
mod context;
mod error;
pub(crate) mod git;
mod io_utils;
mod logging;
pub(crate) mod safari;
pub(crate) mod serde_helpers;
pub(crate) mod shell;
pub(crate) mod time_utils;

pub(crate) use error::AppResult;

use std::process::exit;

use clap::Parser;
use tracing::info;

use cli::{GetDefaultArgs, GetVerbosity};

/// Entrypoint: parse CLI args, set up logging, run command, and emit history output.
#[tokio::main]
async fn main() -> AppResult<()> {
    let args = cli::Cli::parse();

    logging::setup_logger(args.cmd.get_verbosity());

    let combined_hist = args.cmd.run().await?;

    let hist_str = serde_json::to_string_pretty(&combined_hist)?;

    let default_args = args.cmd.get_default_args();

    if let Some(output) = &default_args.output {
        io_utils::write_output(output, &default_args.format, &combined_hist).await?;
    } else {
        info!("Combined History:");
        info!("{}", hist_str);
    }
    exit(0);
}

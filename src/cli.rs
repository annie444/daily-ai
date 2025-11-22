use std::path::PathBuf;

use clap::{Parser, ValueEnum};

/// Generate a summary of your daily activities using AI!
///
/// This tool collects:
/// - your shell history (with [atuin](https://atuin.sh))
/// - Safari browsing history (from the sqlite database)
/// - Git commit history (from your local git repositories, based on your shell history)
///
/// Then, it sends this data to a language model server (like [LM Studio](https://lmstudio.ai/)) to generate a summary.
#[derive(Parser, Debug)]
#[command(author, version)]
pub struct Cli {
    /// Host for the language model server
    #[arg(long, default_value = "localhost")]
    pub host: String,

    /// Port for the language model server
    #[arg(long, default_value_t = 1234)]
    pub port: u16,

    /// API version for the language model server
    /// Defaults to "v1"
    #[arg(long, default_value = "v1")]
    pub api_version: String,

    /// Number of days of history to consider
    /// Defaults to 1 (i.e., today)
    #[arg(short, long, default_value_t = 1)]
    pub days: u32,

    /// Output format: "text" or "json"
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Text)]
    pub format: OutputFormat,

    /// Output file to write the summary to
    /// If not provided, prints to stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    Text,
    Json,
}

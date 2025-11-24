use std::{fmt::Display, io::Write, path::PathBuf};

use async_openai::{Client, config::Config};
use clap::{
    ArgAction, Args, ColorChoice, CommandFactory, Parser, Subcommand, ValueEnum,
    builder::styling::{AnsiColor, Color, Style, Styles},
};
use clap_complete::aot::{Generator, Shell, generate};
use clap_complete_nushell::Nushell;
use time::Duration;
use tracing::info;

use crate::{AppResult, classify, context::Context, git, safari, shell};

const STYLES: Styles = Styles::styled()
    .header(Style::new().bold())
    .usage(Style::new().bold())
    .error(Style::new().fg_color(Some(Color::Ansi(AnsiColor::Red))))
    .literal(
        Style::new()
            .bold()
            .fg_color(Some(Color::Ansi(AnsiColor::Green))),
    )
    .placeholder(Style::new().fg_color(Some(Color::Ansi(AnsiColor::Yellow))))
    .valid(Style::new().fg_color(Some(Color::Ansi(AnsiColor::Cyan))))
    .invalid(Style::new().fg_color(Some(Color::Ansi(AnsiColor::BrightRed))))
    .context(Style::new().fg_color(Some(Color::Ansi(AnsiColor::Magenta))))
    .context_value(
        Style::new()
            .bold()
            .fg_color(Some(Color::Ansi(AnsiColor::Cyan))),
    );

/// Long-form CLI description shown in `--help`.
const LONG_ABOUT: &str = "Daily AI - Summarize your daily activities using AI

This tool collects:
- your shell history (with \x1b]8;;https://atuin.sh\x1b\\\x1b[4;36matuin\x1b[24;39m\x1b]8;;\x1b\\)
- Safari browsing history (from the sqlite database)
- Git commit history (from your local git repositories, based on your shell history)

Then, it sends this data to a language model server (like \x1b]8;;https://lmstudio.ai/\x1b\\\x1b[4;36mLM Studio\x1b[24;39m\x1b]8;;\x1b\\) to generate a summary.";

/// Daily AI - Summarize your daily activities using AI.
#[derive(Parser, Debug, Clone)]
#[command(author, version, propagate_version = true, about, long_about = Some(LONG_ABOUT), styles = STYLES)]
pub struct Cli {
    /// Color choice for the output
    #[arg(long, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,

    /// Subcommand to run
    #[command(subcommand)]
    pub cmd: Cmd,
}

/// Output format for the collected history.
#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Output a JSON file containing all collected changes
    ///
    Json,

    /// Output the collected changes as a series of JSON files (one for shell history, and one for
    /// browsing history) and patch files for each git repository
    ///
    Dir,
}

/// Top-level commands supported by the CLI.
#[derive(Subcommand, Debug, Clone)]
pub enum Cmd {
    /// Generate a summary of your daily activities
    /// This is the default command
    Summarize {
        #[command(flatten)]
        shell: ShellCollectArgs,
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: clap_verbosity_flag::Verbosity,
    },

    /// Collect data without summarizing
    ///
    /// This is useful for debugging or if you want to inspect the collected data
    /// Each subcommand corresponds to a specific data source
    ///
    /// See the documentation for each subcommand for more information
    Collect {
        #[command(subcommand)]
        cmd: CollectCmd,
    },

    /// Generate shell completion for a given shell
    Completion {
        /// Output file to write the completion script to
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// The shell to generate the completion for
        #[arg(value_enum)]
        shell: CompletionShell,

        #[command(flatten)]
        verbosity: clap_verbosity_flag::Verbosity,
    },
}

/// Supported completion targets for shell auto-completion.
#[derive(ValueEnum, Clone, Debug)]
pub enum CompletionShell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
    Elvish,
    Nushell,
}

impl Display for CompletionShell {
    /// Render the canonical shell name string.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            CompletionShell::Bash => "bash",
            CompletionShell::Zsh => "zsh",
            CompletionShell::Fish => "fish",
            CompletionShell::PowerShell => "powershell",
            CompletionShell::Elvish => "elvish",
            CompletionShell::Nushell => "nushell",
        };
        write!(f, "{}", s)
    }
}

impl Generator for &CompletionShell {
    fn generate(&self, cmd: &clap::builder::Command, buf: &mut dyn Write) {
        match self {
            CompletionShell::Bash => Shell::Bash.generate(cmd, buf),
            CompletionShell::Zsh => Shell::Zsh.generate(cmd, buf),
            CompletionShell::Fish => Shell::Fish.generate(cmd, buf),
            CompletionShell::PowerShell => Shell::PowerShell.generate(cmd, buf),
            CompletionShell::Elvish => Shell::Elvish.generate(cmd, buf),
            CompletionShell::Nushell => Nushell.generate(cmd, buf),
        }
    }

    fn file_name(&self, name: &str) -> String {
        match self {
            CompletionShell::Bash => Shell::Bash.file_name(name),
            CompletionShell::Zsh => Shell::Zsh.file_name(name),
            CompletionShell::Fish => Shell::Fish.file_name(name),
            CompletionShell::PowerShell => Shell::PowerShell.file_name(name),
            CompletionShell::Elvish => Shell::Elvish.file_name(name),
            CompletionShell::Nushell => Nushell.file_name(name),
        }
    }
}

/// Subcommands for collecting data without summarizing
///
/// This is useful for debugging or if you want to inspect the collected data
/// Each subcommand corresponds to a specific data source
///
/// See the documentation for each subcommand for more information
/// Subcommands for collecting data without summarizing.
#[derive(Subcommand, Debug, Clone)]
pub enum CollectCmd {
    /// Collect shell history from atuin
    /// Requires atuin to be installed and configured
    /// See https://atuin.sh for more information
    Shell {
        #[command(flatten)]
        shell: ShellCollectArgs,
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: clap_verbosity_flag::Verbosity,
    },

    /// Collect Safari browsing history
    /// Only works on macOS
    /// Requires access to the Safari history database
    /// See https://developer.apple.com/documentation/safariservices/safari_history for more information
    /// Note: This command is a no-op on non-macOS systems
    Safari {
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: clap_verbosity_flag::Verbosity,
    },

    /// Collect git commit history from local repositories
    /// Based on the shell history collected from atuin
    /// Requires git to be installed and accessible in your PATH
    Git {
        #[command(flatten)]
        git: GitCollectArgs,
        #[command(flatten)]
        shell: ShellCollectArgs,
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: clap_verbosity_flag::Verbosity,
    },

    /// Collect all data sources (shell history, Safari history, git history)
    /// This is the default command
    All {
        #[command(flatten)]
        shell: ShellCollectArgs,
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: clap_verbosity_flag::Verbosity,
    },
}

/// Options controlling shell history collection.
#[derive(Args, Debug, Clone)]
pub struct ShellCollectArgs {
    /// Disable syncing atuin history before collecting
    #[arg(long = "no-sync", default_value_t = true, action = ArgAction::SetFalse)]
    pub sync: bool,
}

/// Options controlling git history collection.
#[derive(Args, Debug, Clone)]
pub struct GitCollectArgs {
    /// Include shell history in output when collecting git commits
    #[arg(long, default_value_t = false, action = ArgAction::SetTrue)]
    pub with_shell_history: bool,
}

/// Common options shared across commands.
#[derive(Args, Debug, Clone)]
pub struct DefaultArgs {
    /// Host for the language model server
    #[arg(long, default_value = "localhost")]
    pub host: String,

    /// Port for the language model server
    #[arg(long, default_value_t = 1234)]
    pub port: u16,

    /// OpenAI API version for the language model server
    ///
    /// Defaults to "v1" (the standard OpenAI API version)
    #[arg(long, default_value = "v1")]
    pub api_version: String,

    /// Duration (since now) of history to summarize
    ///
    /// Some valid suffixes are:
    /// - Months: `M`, `month`, or `months`
    /// - Weeks: `w`, `wk`, `wks`, `week`, or `weeks`
    /// - Days: `d`, `day`, or `days`
    /// - Hours: `h`, `hour`, or `hours`
    /// - Minutes: `m`, `min`, or `minutes`
    ///
    /// Defaults to 1d (i.e., yeserday)
    #[arg(short, long, default_value = "1d")]
    pub duration: Option<String>,

    /// Output format for the summary
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Json)]
    pub format: OutputFormat,

    /// Output file to write the summary to
    /// If not provided, prints to stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub trait GetDefaultArgs {
    fn get_default_args(&self) -> &DefaultArgs;
}

/// Helper trait for accessing verbosity flags on commands.
pub trait GetVerbosity {
    fn get_verbosity(&self) -> &clap_verbosity_flag::Verbosity;
}

impl GetDefaultArgs for Cmd {
    fn get_default_args(&self) -> &DefaultArgs {
        match self {
            Cmd::Summarize { default, .. } => default,
            Cmd::Collect { cmd } => cmd.get_default_args(),
            Cmd::Completion { .. } => {
                panic!("Completion command does not have default args")
            }
        }
    }
}

impl GetDefaultArgs for CollectCmd {
    fn get_default_args(&self) -> &DefaultArgs {
        match self {
            CollectCmd::Shell { default, .. } => default,
            CollectCmd::Safari { default, .. } => default,
            CollectCmd::Git { default, .. } => default,
            CollectCmd::All { default, .. } => default,
        }
    }
}

impl GetVerbosity for Cmd {
    fn get_verbosity(&self) -> &clap_verbosity_flag::Verbosity {
        match self {
            Cmd::Summarize { verbosity, .. } => verbosity,
            Cmd::Collect { cmd } => cmd.get_verbosity(),
            Cmd::Completion { verbosity, .. } => verbosity,
        }
    }
}

impl GetVerbosity for CollectCmd {
    fn get_verbosity(&self) -> &clap_verbosity_flag::Verbosity {
        match self {
            CollectCmd::Shell { verbosity, .. } => verbosity,
            CollectCmd::Safari { verbosity, .. } => verbosity,
            CollectCmd::Git { verbosity, .. } => verbosity,
            CollectCmd::All { verbosity, .. } => verbosity,
        }
    }
}

fn get_duration(duration_str: &Option<String>) -> Duration {
    duration_str
        .as_ref()
        .map(|dur_str| {
            Duration::try_from(
                humantime::parse_duration(dur_str)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(86400)),
            )
            .unwrap_or_else(|_| Duration::days(1))
        })
        .unwrap_or_else(|| Duration::days(1))
}

impl Cmd {
    /// Execute the chosen top-level command.
    #[tracing::instrument(name = "Running command", level = "debug", skip(self, client))]
    pub async fn run<C: Config>(&self, client: &Client<C>) -> AppResult<Context> {
        match self {
            Cmd::Summarize {
                shell: ShellCollectArgs { sync },
                default: DefaultArgs { duration, .. },
                ..
            } => {
                self.run_summarize(client, *sync, get_duration(duration))
                    .await
            }
            Cmd::Collect { cmd } => cmd.run(client).await,
            Cmd::Completion { shell, output, .. } => {
                let mut cmd = Cli::command();
                if let Some(output_path) = output {
                    let mut file = std::fs::OpenOptions::new()
                        .write(true)
                        .truncate(true)
                        .create(true)
                        .open(output_path)?;
                    // Write completion script to the requested file.
                    generate(shell, &mut cmd, "daily-ai", &mut file);
                    info!(
                        "Generated completion script for {} at {}",
                        shell,
                        output_path.display()
                    );
                } else {
                    // Fallback: print completion script to stdout.
                    generate(shell, &mut cmd, "daily-ai", &mut std::io::stdout());
                }
                std::process::exit(0);
            }
        }
    }

    #[tracing::instrument(
        name = "Collecting and summarizing history",
        level = "debug",
        skip(self, client)
    )]
    async fn run_summarize<C: Config>(
        &self,
        client: &Client<C>,
        sync: bool,
        duration: Duration,
    ) -> AppResult<Context> {
        // Collect shell, Safari, and git history, then return the aggregated context.
        let shell_history = shell::get_history(sync, &duration).await?;

        let safari_history =
            classify::embed_urls(client, safari::get_safari_history(&duration).await?).await?;

        let commit_history = git::get_git_history(client, &shell_history, &duration).await?;

        Ok(Context {
            shell_history,
            safari_history,
            commit_history,
        })
    }
}

impl CollectCmd {
    /// Execute the specific collect subcommand without summarization.
    #[tracing::instrument(name = "Collecting history", level = "debug", skip(self, client))]
    pub async fn run<C: Config>(&self, client: &Client<C>) -> AppResult<Context> {
        match self {
            CollectCmd::Shell {
                shell: ShellCollectArgs { sync },
                default: DefaultArgs { duration, .. },
                ..
            } => {
                let duration = get_duration(duration);
                let shell_history = shell::get_history(*sync, &duration).await?;
                Ok(Context {
                    shell_history,
                    safari_history: vec![],
                    commit_history: vec![],
                })
            }
            CollectCmd::Safari {
                default: DefaultArgs { duration, .. },
                ..
            } => {
                let duration = get_duration(duration);
                let safari_history =
                    classify::embed_urls(client, safari::get_safari_history(&duration).await?)
                        .await?;
                Ok(Context {
                    shell_history: vec![],
                    safari_history,
                    commit_history: vec![],
                })
            }
            CollectCmd::Git {
                shell: ShellCollectArgs { sync },
                git: GitCollectArgs { with_shell_history },
                default: DefaultArgs { duration, .. },
                ..
            } => {
                let duration = get_duration(duration);
                let shell_history = shell::get_history(*sync, &duration).await?;
                let commit_history =
                    git::get_git_history(client, &shell_history, &duration).await?;
                let shell_history = if *with_shell_history {
                    shell_history
                } else {
                    vec![]
                };
                Ok(Context {
                    shell_history,
                    safari_history: vec![],
                    commit_history,
                })
            }
            CollectCmd::All {
                shell: ShellCollectArgs { sync },
                default: DefaultArgs { duration, .. },
                ..
            } => {
                let duration = get_duration(duration);
                let shell_history = shell::get_history(*sync, &duration).await?;

                let safari_history =
                    classify::embed_urls(client, safari::get_safari_history(&duration).await?)
                        .await?;

                let commit_history =
                    git::get_git_history(client, &shell_history, &duration).await?;

                Ok(Context {
                    shell_history,
                    safari_history,
                    commit_history,
                })
            }
        }
    }
}

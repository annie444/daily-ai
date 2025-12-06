use std::fmt::Display;
use std::io::Write;
use std::path::PathBuf;

use async_openai::Client;
use async_openai::config::{Config, OpenAIConfig};
use clap::builder::styling::{AnsiColor, Color, Style, Styles};
use clap::{ArgAction, Args, ColorChoice, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::aot::{Generator, Shell, generate};
use clap_complete_nushell::Nushell;
use clap_verbosity_flag::{InfoLevel, Verbosity};
use time::Duration;
use tracing::{error, info};

use crate::ai::SchemaInfo;
use crate::context::{Context, FullContext};
use crate::{AppResult, ai, classify, git, safari, shell};

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
        verbosity: Verbosity<InfoLevel>,
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
        verbosity: Verbosity<InfoLevel>,
    },

    Show {
        #[command(subcommand)]
        query: Queries,
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

static SHELL_CMD_ABOUT: &str = "Collect shell history from atuin
Requires atuin to be installed and configured
See \x1b]8;;https://atuin.sh\x1b\\\x1b[4;36matuin.sh\x1b[24;39m\x1b]8;;\x1b\\ for more information";

static SAFARI_CMD_ABOUT: &str = "Collect Safari browsing history
Only works on macOS
Requires access to the Safari history database
See \x1b]8;;https://developer.apple.com/documentation/safariservices/safari_history\x1b\\\x1b[4;36mApple's developer documentation\x1b[24;39m\x1b]8;;\x1b\\ for more information
Note: This command is a no-op on non-macOS systems";

/// Subcommands for collecting data without summarizing
///
/// This is useful for debugging or if you want to inspect the collected data
/// Each subcommand corresponds to a specific data source
///
/// See the documentation for each subcommand for more information
/// Subcommands for collecting data without summarizing.
#[derive(Subcommand, Debug, Clone)]
pub enum CollectCmd {
    #[command(about = "Collect shell history from atuin", long_about = SHELL_CMD_ABOUT)]
    Shell {
        #[command(flatten)]
        shell: ShellCollectArgs,
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: Verbosity<InfoLevel>,
    },

    #[command(about = "Collect Safari browsing history", long_about = SAFARI_CMD_ABOUT)]
    Safari {
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: Verbosity<InfoLevel>,
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
        verbosity: Verbosity<InfoLevel>,
    },

    /// Collect all data sources (shell history, Safari history, git history)
    /// This is the default command
    All {
        #[command(flatten)]
        shell: ShellCollectArgs,
        #[command(flatten)]
        default: DefaultArgs,
        #[command(flatten)]
        verbosity: Verbosity<InfoLevel>,
    },
}

static SHOW_CMD_ABOUT: &str = "Show the AI queries that are available.
This is mainly for debugging purposes, but can also be used to see what queries are supported.
Useful for developers who want to extend the functionality of the tool.";

#[derive(Subcommand, Debug, Clone)]
#[command(about = "Show the AI queries that are available.", long_about = SHOW_CMD_ABOUT)]
pub enum Queries {
    /// Schemas for commit message generation
    CommitMessage(QueryArgs<CommitMessageTools, CommitMessageResponses>),

    /// Schemas for URL labeling
    LabelUrls(QueryArgs<LabelUrlsTools, LabelUrlsResponses>),

    /// Schemas for daily summary generation
    Summary(QueryArgs<SummaryTools, SummaryResponses>),
}

#[derive(Args, Debug, Clone)]
pub struct QueryArgs<T, R>
where
    T: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
    R: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
{
    #[command(flatten)]
    verbosity: Verbosity<InfoLevel>,
    #[command(subcommand)]
    opt: ToolsAndResponses<T, R>,
}

impl<T, R> GetVerbosity for QueryArgs<T, R>
where
    T: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
    R: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
{
    fn get_verbosity(&self) -> &Verbosity<InfoLevel> {
        &self.verbosity
    }
}

impl<T, R> QueryArgs<T, R>
where
    T: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
    R: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
{
    pub fn run(&self) {
        match &self.opt {
            ToolsAndResponses::Tools { tool, .. } => {
                let schema = tool.print_schema();
                tracing_indicatif::indicatif_println!("Schema for tool type {:?}:\n{schema}", tool);
            }
            ToolsAndResponses::Responses { response, .. } => {
                let schema = response.print_schema();
                tracing_indicatif::indicatif_println!(
                    "Schema for response type {:?}:\n{schema}",
                    response
                );
            }
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
#[command(about = "Show the AI queries that are available.", long_about = SHOW_CMD_ABOUT)]
pub enum ToolsAndResponses<T, R>
where
    T: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
    R: clap::ValueEnum + Clone + std::fmt::Debug + Send + Sync + 'static + PrintSchema,
{
    /// The tool to show information about
    Tools {
        #[command(flatten)]
        verbosity: Verbosity<InfoLevel>,
        /// The tool to show information about
        #[arg(value_enum)]
        tool: T,
    },

    /// The response type to show information about
    Responses {
        #[command(flatten)]
        verbosity: Verbosity<InfoLevel>,
        /// The response type to show information about
        #[arg(value_enum)]
        response: R,
    },
}

pub trait PrintSchema {
    fn print_schema(&self) -> String;
}

#[derive(ValueEnum, Debug, Clone)]
pub enum CommitMessageTools {
    GetFile,
    GetPatch,
}

impl PrintSchema for CommitMessageTools {
    fn print_schema(&self) -> String {
        let val = match self {
            Self::GetFile => ai::tools::commit::GetFile::schema_value(),
            Self::GetPatch => ai::tools::commit::GetPatch::schema_value(),
        };
        match serde_json::to_string_pretty(&val) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize schema for {:?}: {}", self, e);
                std::process::exit(1);
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum CommitMessageResponses {
    CommitMessage,
}

impl PrintSchema for CommitMessageResponses {
    fn print_schema(&self) -> String {
        match self {
            Self::CommitMessage => {
                match serde_json::to_string_pretty(
                    &ai::commit_message::CommitMessage::schema_value(),
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        error!("Failed to serialize schema for {:?}: {}", self, e);
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum LabelUrlsTools {
    FetchUrl,
}

impl PrintSchema for LabelUrlsTools {
    fn print_schema(&self) -> String {
        match self {
            Self::FetchUrl => {
                match serde_json::to_string_pretty(&ai::tools::fetch::FetchUrl::schema_value()) {
                    Ok(s) => s,
                    Err(e) => {
                        error!("Failed to serialize schema for {:?}: {}", self, e);
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum LabelUrlsResponses {
    UrlLabel,
}

impl PrintSchema for LabelUrlsResponses {
    fn print_schema(&self) -> String {
        match self {
            Self::UrlLabel => {
                match serde_json::to_string_pretty(&ai::label_urls::UrlLabel::schema_value()) {
                    Ok(s) => s,
                    Err(e) => {
                        error!("Failed to serialize schema for {:?}: {}", self, e);
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum SummaryTools {
    FetchUrl,
    GetDiff,
    GetRepo,
    GetCommitMessages,
    GetBrowserHistory,
    GetShellHistory,
}

impl PrintSchema for SummaryTools {
    fn print_schema(&self) -> String {
        let val = match self {
            Self::FetchUrl => ai::tools::fetch::FetchUrl::schema_value(),
            Self::GetDiff => ai::tools::summary::GetDiff::schema_value(),
            Self::GetRepo => ai::tools::summary::GetRepo::schema_value(),
            Self::GetCommitMessages => ai::tools::summary::GetCommitMessages::schema_value(),
            Self::GetBrowserHistory => ai::tools::summary::GetBrowserHistory::schema_value(),
            Self::GetShellHistory => ai::tools::summary::GetShellHistory::schema_value(),
        };
        match serde_json::to_string_pretty(&val) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize schema for {:?}: {}", self, e);
                std::process::exit(1);
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum SummaryResponses {
    FullSummary,
    Summary,
    Highlights,
    RepoSummary,
    ShellOverview,
    TimeBreakdown,
    CommonGroups,
}

impl PrintSchema for SummaryResponses {
    fn print_schema(&self) -> String {
        let val = match self {
            Self::FullSummary => ai::summary::WorkSummary::schema_value(),
            Self::Summary => ai::summary::SummaryQuery::schema_value(),
            Self::Highlights => ai::summary::HighlightsQuery::schema_value(),
            Self::RepoSummary => ai::summary::RepoSummaryQuery::schema_value(),
            Self::ShellOverview => ai::summary::ShellOverviewQuery::schema_value(),
            Self::TimeBreakdown => ai::summary::TimeBreakdownQuery::schema_value(),
            Self::CommonGroups => ai::summary::CommonGroupsQuery::schema_value(),
        };
        match serde_json::to_string_pretty(&val) {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to serialize schema for {:?}: {}", self, e);
                std::process::exit(1);
            }
        }
    }
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
    /// Whether to use secure connection (HTTPS) to the language model server
    /// Defaults to false for local servers (i.e. `localhost` and private subnets)
    /// Defaults to true for public IP addresses and hostnames
    /// Note: This is not a flag. You must provide a value (true or false) if you use this option.
    #[arg(long)]
    pub secure: Option<bool>,

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

impl DefaultArgs {
    pub fn get_client(&self) -> Client<Box<dyn Config>> {
        let schema = if let Some(secure) = self.secure {
            if secure { "https" } else { "http" }
        } else if self.host == "localhost"
            || self.host.ends_with(".local")
            || self.host.ends_with(".internal")
            || self.host.ends_with(".lan")
            || self.host.ends_with(".corp")
            || self.host.ends_with(".home.arpa")
            || self.host.ends_with(".private")
            || self.host.ends_with(".test")
            || self
                .host
                .parse::<std::net::Ipv4Addr>()
                .is_ok_and(|ip| ip.is_loopback() || ip.is_private() || ip.is_link_local())
            || self.host.parse::<std::net::Ipv6Addr>().is_ok_and(|ip| {
                ip.is_loopback() || ip.is_unique_local() || ip.is_unicast_link_local()
            })
        {
            "http"
        } else {
            "https"
        };
        let config = Box::new(OpenAIConfig::default().with_api_base(format!(
            "{schema}://{}:{}/{}",
            self.host, self.port, self.api_version
        ))) as Box<dyn Config>;

        Client::with_config(config)
    }
}

pub trait GetDefaultArgs {
    fn get_default_args(&self) -> &DefaultArgs;

    fn get_client(&self) -> Client<Box<dyn Config>> {
        self.get_default_args().get_client()
    }
}

/// Helper trait for accessing verbosity flags on commands.
pub trait GetVerbosity {
    fn get_verbosity(&self) -> &Verbosity<InfoLevel>;
}

impl GetDefaultArgs for Cmd {
    fn get_default_args(&self) -> &DefaultArgs {
        match self {
            Cmd::Summarize { default, .. } => default,
            Cmd::Collect { cmd } => cmd.get_default_args(),
            Cmd::Show { .. } => {
                panic!("Show command does not have default args")
            }
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
    fn get_verbosity(&self) -> &Verbosity<InfoLevel> {
        match self {
            Cmd::Summarize { verbosity, .. } => verbosity,
            Cmd::Collect { cmd } => cmd.get_verbosity(),
            Cmd::Completion { verbosity, .. } => verbosity,
            Cmd::Show { query } => query.get_verbosity(),
        }
    }
}

impl GetVerbosity for CollectCmd {
    fn get_verbosity(&self) -> &Verbosity<InfoLevel> {
        match self {
            CollectCmd::Shell { verbosity, .. } => verbosity,
            CollectCmd::Safari { verbosity, .. } => verbosity,
            CollectCmd::Git { verbosity, .. } => verbosity,
            CollectCmd::All { verbosity, .. } => verbosity,
        }
    }
}

impl GetVerbosity for Queries {
    fn get_verbosity(&self) -> &Verbosity<InfoLevel> {
        match self {
            Queries::CommitMessage(args) => args.get_verbosity(),
            Queries::LabelUrls(args) => args.get_verbosity(),
            Queries::Summary(args) => args.get_verbosity(),
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
    #[tracing::instrument(name = "Running command", level = "info", skip(self))]
    pub async fn run(&self) -> AppResult<FullContext> {
        match self {
            Cmd::Summarize {
                shell: ShellCollectArgs { sync },
                default: DefaultArgs { duration, .. },
                ..
            } => {
                let client = self.get_client();
                let duration_val = get_duration(duration);
                let duration_str = duration.as_deref().unwrap_or("1d");
                self.run_summarize(&client, *sync, duration_val, duration_str)
                    .await
            }
            Cmd::Collect { cmd } => Ok(cmd.run().await?.into()),
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
            Cmd::Show { query } => {
                query.run();
                std::process::exit(0);
            }
        }
    }

    #[tracing::instrument(
        name = "Collecting and summarizing history",
        level = "info",
        skip(self, client)
    )]
    async fn run_summarize<C: Config>(
        &self,
        client: &Client<C>,
        sync: bool,
        duration: Duration,
        duration_label: &str,
    ) -> AppResult<FullContext> {
        // Collect shell, Safari, and git history, then return the aggregated context.
        let shell_history = shell::get_history(sync, &duration).await?;

        let safari_history =
            classify::embed_urls(client, safari::get_safari_history(&duration).await?).await?;

        let commit_history = git::get_git_history(client, &shell_history, &duration).await?;

        let ctx = Context {
            shell_history,
            safari_history,
            commit_history,
        };

        let mut vars = std::collections::HashMap::new();
        vars.insert("duration", duration_label);
        let summary = ai::summary::generate_summary(client, &ctx, &vars).await?;

        Ok(FullContext::from((ctx, summary)))
    }
}

impl CollectCmd {
    /// Execute the specific collect subcommand without summarization.
    #[tracing::instrument(name = "Collecting history", level = "info", skip(self))]
    pub async fn run(&self) -> AppResult<Context> {
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
                let client = self.get_client();
                let duration = get_duration(duration);
                let safari_history =
                    classify::embed_urls(&client, safari::get_safari_history(&duration).await?)
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
                let client = self.get_client();
                let duration = get_duration(duration);
                let shell_history = shell::get_history(*sync, &duration).await?;
                let commit_history =
                    git::get_git_history(&client, &shell_history, &duration).await?;
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
                let client = self.get_client();
                let duration = get_duration(duration);
                let shell_history = shell::get_history(*sync, &duration).await?;

                let safari_history =
                    classify::embed_urls(&client, safari::get_safari_history(&duration).await?)
                        .await?;

                let commit_history =
                    git::get_git_history(&client, &shell_history, &duration).await?;

                Ok(Context {
                    shell_history,
                    safari_history,
                    commit_history,
                })
            }
        }
    }
}

impl Queries {
    pub fn run(&self) {
        match self {
            Queries::CommitMessage(args) => args.run(),
            Queries::LabelUrls(args) => args.run(),
            Queries::Summary(args) => args.run(),
        }
    }
}

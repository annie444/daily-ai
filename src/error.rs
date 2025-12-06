use thiserror::Error;

/// Unified application error type to simplify bubbling errors through async flows.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Errored while handling a file. {0}")]
    Command(#[from] std::io::Error),
    #[error("Error parsing a number. {0}")]
    Parse(#[from] std::num::ParseIntError),
    #[error("Error handling the database. {0}")]
    Database(#[from] sqlx::Error),
    #[error("Error from git. {0}")]
    Git(#[from] git2::Error),
    #[error("Error serializing json. {0}")]
    SerdeJsonSer(#[from] serde_json::Error),
    #[error("Error communicating with the AI. {0}")]
    AIClient(#[from] async_openai::error::OpenAIError),
    #[error("Error while writing information to a string. {0}")]
    BufferWrite(#[from] std::fmt::Error),
    #[error("Unable to parse string. {0}")]
    Utf8Parse(#[from] std::str::Utf8Error),
    #[error("{0}")]
    Other(String),
    #[error("Uh oh! The runtime had a problem. Here's what happened: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    #[error("Unable to convert HTTP header to a string. Here's what I found: {0}")]
    HeaderToStr(#[from] reqwest::header::ToStrError),
    #[error(
        "Something happened while processing shell history from Atuin. Atuin errored with: {0}"
    )]
    AtuinClient(String),
    #[error("Something happened while accessing the internet. Here's the error: {0}")]
    MCPClient(#[from] reqwest::Error),
    #[error("Unable to convert the duration string to a number. Got error: {0}")]
    DurationParse(#[from] humantime::DurationError),
    #[error("Duration seems too large... The value overflowed with the error: {0}")]
    DurationOverflow(#[from] time::error::ConversionRange),
    #[error("Something happened during linear algebra operations. Here's the error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),
    #[error("Something happened while grouping the URLs. This is the error: {0}")]
    Hdbscan(#[from] hdbscan::HdbscanError),
    #[error("{0}")]
    Dir(#[from] daily_ai_dirs::DirError),
}

/// Convenience alias for results that bubble `AppError`.
pub type AppResult<T> = Result<T, AppError>;

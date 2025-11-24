use thiserror::Error;

/// Unified application error type to simplify bubbling errors through async flows.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Errored while handling a file. {0}")]
    Command(#[from] std::io::Error),
    #[error("Error parsing a number. {0}")]
    Parse(#[from] std::num::ParseIntError),
    #[error("Error handling the database. {0}")]
    Database(#[from] sea_orm::DbErr),
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
    #[error("Error from SQLite driver. {0}")]
    Sqlx(#[from] sea_orm::sqlx::Error),
    #[error("{0}")]
    Other(String),
    #[error("Error while running a local ML model. {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Runtime error. {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    #[error("Error while tokenizing input. {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Error handling local ML model. {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("Directory not found error. {0}")]
    DirNotFound(String),
    #[error("Error converting header to a string. {0}")]
    HeaderToStr(#[from] reqwest::header::ToStrError),
    #[error("Error with the Atuin client. {0}")]
    AtuinClient(String),
    #[error("Error accessing the internet. {0}")]
    MCPClient(#[from] reqwest::Error),
    #[error("Error parsing the duration string. {0}")]
    DurationParse(#[from] humantime::DurationError),
    #[error("Duration value overflowed. {0}")]
    DurationOverflow(#[from] time::error::ConversionRange),
}

/// Convenience alias for results that bubble `AppError`.
pub type AppResult<T> = Result<T, AppError>;

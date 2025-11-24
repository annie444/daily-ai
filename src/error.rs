use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("IO Error: {0}")]
    Command(#[from] std::io::Error),
    #[error("Parse Error: {0}")]
    Parse(#[from] std::num::ParseIntError),
    #[error("Database Error: {0}")]
    Database(#[from] sea_orm::DbErr),
    #[error("Git Error: {0}")]
    Git(#[from] git2::Error),
    #[error("Serialization Error: {0}")]
    SerdeJsonSer(#[from] serde_json::Error),
    #[error("AI Client Error: {0}")]
    AIClient(#[from] async_openai::error::OpenAIError),
    #[error("Uneable to write to buffer: {0}")]
    BufferWrite(#[from] std::fmt::Error),
    #[error("Unable to parse string to UTF-8: {0}")]
    Utf8Parse(#[from] std::str::Utf8Error),
    #[error("SQLite Error: {0}")]
    Sqlx(#[from] sea_orm::sqlx::Error),
    #[error("Other error: {0}")]
    Other(String),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("Runtime error: {0}")]
    TokioJoin(#[from] tokio::task::JoinError),
    #[error("Tokenization error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("Directory not found: {0}")]
    DirNotFound(String),
    #[error("Error converting header to a string: {0}")]
    HeaderToStr(#[from] reqwest::header::ToStrError),
    #[error("Error with Atuin client: {0}")]
    AtuinClient(String),
    #[error("Error accessing the internet: {0}")]
    MCPClient(#[from] reqwest::Error),
}

pub type AppResult<T> = Result<T, AppError>;

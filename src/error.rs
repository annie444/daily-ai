use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("IO Error: {0}")]
    Command(#[from] std::io::Error),
    #[error("Parse Error: {0}")]
    Parse(#[from] std::num::ParseIntError),
    #[error("Time parse error: {0}")]
    ChronoParse(#[from] chrono::ParseError),
    #[error("Failed to capture standard output of the child process")]
    StdoutCapture,
    #[error("Failed to split host and directory from string: {0}")]
    HostDirSplit(String),
    #[error("Database Error: {0}")]
    Database(#[from] sea_orm::DbErr),
    #[error("Git Error: {0}")]
    Git(#[from] git2::Error),
    #[error("Serialization Error: {0}")]
    SerdeJsonSer(#[from] serde_json::Error),
    #[error("AI Client Error: {0}")]
    AIClient(#[from] async_openai::error::OpenAIError),
}

pub type AppResult<T> = Result<T, AppError>;

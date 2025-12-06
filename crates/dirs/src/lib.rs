use std::env;
use std::fmt::Display;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum DirError {
    #[error("Directory not found: {0}")]
    DirNotFound(String),
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type DirResult<T> = Result<T, DirError>;

/// Application name used to namespace directories.
pub static APP_NAME: &str = "dailyai";

#[derive(Debug)]
pub enum DirType {
    Data,
    Config,
    Cache,
}

impl Display for DirType {
    /// Pretty-print the default directory path hint for this dir type.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DirType::Data => write!(f, "~/.local/share/")?,
            DirType::Config => write!(f, "~/.config/")?,
            DirType::Cache => write!(f, "~/.cache/")?,
        };
        write!(f, "{}", APP_NAME)
    }
}

impl DirType {
    /// XDG environment variable key for this directory type.
    fn xdg_key(&self) -> &'static str {
        match self {
            DirType::Data => "XDG_DATA_HOME",
            DirType::Config => "XDG_CONFIG_HOME",
            DirType::Cache => "XDG_CACHE_HOME",
        }
    }

    /// Relative default path under HOME when XDG is not set.
    fn rel_path(&self) -> &'static str {
        match self {
            DirType::Data => ".local/share",
            DirType::Config => ".config",
            DirType::Cache => ".cache",
        }
    }

    /// Resolve the directory path from XDG or fallback environment hints.
    pub fn get_dir(&self) -> DirResult<PathBuf> {
        if let Some(dir) = env::var_os(self.xdg_key()) {
            Ok(PathBuf::from(dir).join(APP_NAME))
        } else if let Some(home_dir) = env::home_dir() {
            Ok(home_dir.join(self.rel_path()).join(APP_NAME))
        } else if let Ok(home) = env::var("HOME") {
            Ok(PathBuf::from(home).join(self.rel_path()).join(APP_NAME))
        } else if let Ok(userprofile) = env::var("USERPROFILE") {
            Ok(PathBuf::from(userprofile)
                .join(self.rel_path())
                .join(APP_NAME))
        } else {
            Err(DirError::DirNotFound(self.to_string()))
        }
    }

    /// Ensure the directory exists, creating it asynchronously if needed.
    pub async fn ensure_dir_async(&self) -> DirResult<PathBuf> {
        let dir = self.get_dir()?;
        tokio::fs::create_dir_all(&dir).await?;
        Ok(dir)
    }

    /// Ensure the directory exists, creating it if needed.
    pub fn ensure_dir(&self) -> DirResult<PathBuf> {
        let dir = self.get_dir()?;
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}

use crate::error::{AppError, AppResult};
use std::env;
use std::fmt::Display;
use std::path::PathBuf;

pub static APP_NAME: &str = "dailyai";

pub enum DirType {
    Data,
    Config,
    Cache,
}

impl Display for DirType {
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
    fn xdg_key(&self) -> &'static str {
        match self {
            DirType::Data => "XDG_DATA_HOME",
            DirType::Config => "XDG_CONFIG_HOME",
            DirType::Cache => "XDG_CACHE_HOME",
        }
    }

    fn rel_path(&self) -> &'static str {
        match self {
            DirType::Data => ".local/share",
            DirType::Config => ".config",
            DirType::Cache => ".cache",
        }
    }

    pub fn get_dir(&self) -> AppResult<PathBuf> {
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
            Err(AppError::DirNotFound(self.to_string()))
        }
    }

    pub fn ensure_dir(&self) -> AppResult<PathBuf> {
        let dir = self.get_dir()?;
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    pub async fn ensure_dir_async(&self) -> AppResult<PathBuf> {
        let dir = self.get_dir()?;
        tokio::fs::create_dir_all(&dir).await?;
        Ok(dir)
    }
}

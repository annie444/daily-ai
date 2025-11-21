use crate::{AppError, AppResult};
use chrono::{NaiveDateTime, TimeDelta};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tracing::{debug, trace};

#[derive(Debug, Serialize, Deserialize)]
pub struct ShellHistoryEntry {
    #[serde(with = "crate::serde_helpers::naive_datetime")]
    pub date_time: NaiveDateTime,
    #[serde(with = "crate::serde_helpers::duration")]
    pub duration: TimeDelta,
    pub host: String,
    pub directory: PathBuf,
    pub command: String,
}

#[tracing::instrument(level = "trace")]
fn parse_duration<S: AsRef<str> + std::fmt::Debug>(s: S) -> TimeDelta {
    crate::serde_helpers::duration::parse_duration(s).unwrap_or_else(|_| TimeDelta::seconds(0))
}

#[tracing::instrument(level = "trace")]
fn get_shell() -> String {
    match env::var("SHELL") {
        Ok(val) => val,
        Err(_) => env::var_os("SHELL")
            .and_then(|os_str| os_str.into_string().ok())
            .unwrap_or_else(|| "/bin/bash".to_string()),
    }
}

#[tracing::instrument(level = "trace")]
async fn atuin_sync<S: AsRef<str> + std::fmt::Debug>(shell: S) -> AppResult<()> {
    Command::new(shell.as_ref())
        .arg("-l")
        .arg("-c")
        .arg("atuin sync")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await?;
    debug!("Atuin sync completed");
    Ok(())
}

#[tracing::instrument(level = "trace")]
async fn atuin_history<S: AsRef<str> + std::fmt::Debug>(shell: S) -> AppResult<Child> {
    Command::new(shell.as_ref())
        .arg("-l")
        .arg("-c")
        .arg("atuin history list")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| e.into())
}

#[tracing::instrument(level = "trace", skip(line))]
fn parse_line<S: AsRef<str> + std::fmt::Debug>(line: S) -> AppResult<Option<ShellHistoryEntry>> {
    let sections = line.as_ref().split('\t').collect::<Vec<&str>>();
    if sections.len() < 3 {
        return Ok(None);
    }
    let (host, dir) = sections[2]
        .split_once(':')
        .ok_or(AppError::HostDirSplit(sections[2].to_string()))?;

    let datetime = NaiveDateTime::parse_from_str(sections[0], "%Y-%m-%d %H:%M:%S")?;
    if datetime
        < chrono::Local::now()
            .naive_local()
            .checked_sub_days(chrono::Days::new(1))
            .unwrap()
    {
        return Ok(None);
    }
    let cmd = sections[3..].join("\t");
    Ok(Some(ShellHistoryEntry {
        date_time: datetime,
        duration: parse_duration(sections[1]),
        host: host.to_string(),
        directory: dir.into(),
        command: cmd,
    }))
}

#[tracing::instrument(level = "debug")]
pub async fn get_history() -> AppResult<Vec<ShellHistoryEntry>> {
    let shell = get_shell();
    atuin_sync(&shell).await?;
    let mut child = atuin_history(&shell).await?;
    debug!("Capturing shell history from Atuin");

    let stdout = child.stdout.take().ok_or(AppError::StdoutCapture)?;
    let reader = BufReader::new(stdout);

    let mut history = Vec::new();
    let mut lines = reader.lines();
    trace!("Reading shell history lines");
    while let Some(line) = lines.next_line().await? {
        let entry = match parse_line(&line)? {
            Some(e) => e,
            None => continue,
        };
        history.push(entry);
    }
    trace!("Finished reading shell history lines");
    child.wait().await?;
    trace!("Atuin history command completed");
    Ok(history)
}

use chrono::Local;
use fern::colors::{Color, ColoredLevelConfig};
use fern::{Dispatch, FormatCallback};
use log::{LevelFilter, Record};
use std::env;
use std::fmt::Arguments;

static ESCAPE: &str = "\x1B[";
static END: &str = "m";
static RESET: &str = "0";
static COLORS: ColoredLevelConfig = ColoredLevelConfig {
    error: Color::Red,
    warn: Color::Yellow,
    info: Color::Green,
    debug: Color::Cyan,
    trace: Color::Magenta,
};

fn format(out: FormatCallback, message: &Arguments, record: &Record) {
    out.finish(format_args!(
        "[{}{}{}{} {} {}{}{}{}] {}",
        ESCAPE,
        COLORS.get_color(&record.level()).to_fg_str(),
        END,
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        record.level(),
        record.target(),
        ESCAPE,
        RESET,
        END,
        message
    ))
}

pub fn setup_logger() -> Result<(), fern::InitError> {
    let log_level = match env::var("DAILY_AI_LOG_LEVEL") {
        Ok(level) => match level
            .to_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .as_str()
        {
            "error" => LevelFilter::Error,
            "warn" => LevelFilter::Warn,
            "info" => LevelFilter::Info,
            "debug" => LevelFilter::Debug,
            "trace" => LevelFilter::Trace,
            _ => {
                if cfg!(debug_assertions) {
                    LevelFilter::max()
                } else {
                    LevelFilter::Info
                }
            }
        },
        Err(_) => {
            if cfg!(debug_assertions) {
                LevelFilter::max()
            } else {
                LevelFilter::Info
            }
        }
    };
    Dispatch::new()
        .format(format)
        .level(log_level)
        .chain(std::io::stdout())
        .apply()?;
    Ok(())
}

use time::{Duration, OffsetDateTime, Time};
use tracing::trace;

/// Seconds between Unix epoch (1970) and macOS epoch (2001).
const MACOS_EPOCH_OFFSET: f64 = 978_307_200.0;

/// Convert an `OffsetDateTime` to macOS timestamp (seconds since 2001-01-01) as f64.
#[tracing::instrument(
    name = "Converting standard date and time to a MacOS timestamp",
    level = "trace"
)]
pub fn datetime_to_macos_time(dt: &OffsetDateTime) -> f64 {
    let utc = dt.unix_timestamp_nanos();
    let ts = unix_time_sec_to_macos_time(utc);
    trace!("Converted datetime {} to macOS time {}", dt, ts);
    ts
}

/// Convert Unix time in nanoseconds to macOS timestamp in seconds.
#[tracing::instrument(
    name = "Converting a Unix timestamp to a MacOS timestamp",
    level = "trace"
)]
pub fn unix_time_sec_to_macos_time(nsecs: i128) -> f64 {
    let ts = nsecs as f64 / 1_000_000_000.0 - MACOS_EPOCH_OFFSET;
    trace!("Converted Unix time {}ns to macOS time {}", nsecs, ts);
    ts
}

/// macOS timestamp (seconds) for the given duration ago (default is 24 hours).
#[tracing::instrument(
    name = "Calculating the date and time in the past as a MacOS timestamp",
    level = "trace"
)]
pub fn macos_past_ts(duration: &Duration) -> f64 {
    let ts = OffsetDateTime::now_utc().saturating_sub(*duration);
    trace!("Calculated the past timestamp: {}", ts);
    unix_time_sec_to_macos_time(ts.unix_timestamp_nanos())
}

/// UTC datetime for the given duration ago (default is 24 hours).
#[tracing::instrument(name = "Calculate the date and time in the past", level = "trace")]
pub fn past_ts(duration: &Duration) -> OffsetDateTime {
    OffsetDateTime::now_utc().saturating_sub(*duration)
}

/// Convert seconds to nanoseconds.
#[tracing::instrument(name = "Converting seconds to nanoseconds", level = "trace")]
pub fn timestamp_secs_to_nsecs(secs: i64) -> i128 {
    (secs as i128) * 1_000_000_000
}

/// Convert Unix time (nanoseconds) to local `OffsetDateTime` for user-facing output.
#[tracing::instrument(name = "Converting a Unix timestamp to date and time", level = "trace")]
pub fn unix_time_nsec_to_datetime(secs: i128) -> OffsetDateTime {
    // Convert to local time for user-facing output.
    OffsetDateTime::from_unix_timestamp_nanos(secs)
        .unwrap()
        .to_offset(OffsetDateTime::now_local().unwrap().offset())
}

/// Convert macOS timestamp (seconds since 2001) to Unix time in nanoseconds.
#[tracing::instrument(
    name = "Converting a MacOS timestamp to a Unix standard timestamp",
    level = "trace"
)]
pub fn macos_to_unix_time(macos_time: f64) -> i128 {
    ((macos_time + MACOS_EPOCH_OFFSET) * 1_000_000_000.0) as i128
}

/// Convert macOS timestamp to UTC `OffsetDateTime`.
#[tracing::instrument(name = "Converting MacOS timestamp to date and time", level = "trace")]
pub fn macos_to_datetime(macos_time: f64) -> OffsetDateTime {
    let secs = macos_to_unix_time(macos_time);
    OffsetDateTime::from_unix_timestamp_nanos(secs).unwrap()
}

/// Midnight (00:00) today in UTC.
#[tracing::instrument(
    name = "Calculating the date and time of today at UTC midnight",
    level = "trace"
)]
pub fn midnight_utc() -> OffsetDateTime {
    let today = OffsetDateTime::now_utc();
    OffsetDateTime::new_utc(today.date(), Time::MIDNIGHT)
}

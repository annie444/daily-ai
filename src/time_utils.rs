use time::{Duration, OffsetDateTime, Time};
use tracing::trace;

const MACOS_EPOCH_OFFSET: f64 = 978307200.0;

#[tracing::instrument(level = "trace")]
pub fn datetime_to_macos_time(dt: &OffsetDateTime) -> f64 {
    let utc = dt.unix_timestamp_nanos();
    let ts = unix_time_sec_to_macos_time(utc);
    trace!("Converted datetime {} to macOS time {}", dt, ts);
    ts
}

#[tracing::instrument(level = "trace")]
pub fn unix_time_sec_to_macos_time(secs: i128) -> f64 {
    let ts = secs as f64 / 1_000_000_000.0 - MACOS_EPOCH_OFFSET;
    trace!("Converted Unix time {}s to macOS time {}", secs, ts);
    ts
}

#[tracing::instrument(level = "trace")]
pub fn macos_yesterday() -> f64 {
    let ts = OffsetDateTime::now_utc().saturating_sub(Duration::days(1));
    trace!("Calculated yesterday's date: {}", ts);
    unix_time_sec_to_macos_time(ts.unix_timestamp_nanos())
}

#[tracing::instrument(level = "trace")]
pub fn yesterday() -> OffsetDateTime {
    OffsetDateTime::now_utc().saturating_sub(Duration::days(1))
}

#[tracing::instrument(level = "trace")]
pub fn timestamp_secs_to_nsecs(secs: i64) -> i128 {
    (secs as i128) * 1_000_000_000
}

#[tracing::instrument(level = "trace")]
pub fn unix_time_nsec_to_datetime(secs: i128) -> OffsetDateTime {
    // Convert to local time for user-facing output.
    OffsetDateTime::from_unix_timestamp_nanos(secs)
        .unwrap()
        .to_offset(OffsetDateTime::now_local().unwrap().offset())
}

#[tracing::instrument(level = "trace")]
pub fn macos_to_unix_time(macos_time: f64) -> i128 {
    ((macos_time + MACOS_EPOCH_OFFSET) * 1_000_000_000.0) as i128
}

#[tracing::instrument(level = "trace")]
pub fn macos_to_datetime(macos_time: f64) -> OffsetDateTime {
    let secs = macos_to_unix_time(macos_time);
    OffsetDateTime::from_unix_timestamp_nanos(secs).unwrap()
}

#[tracing::instrument(level = "trace")]
pub fn midnight_utc() -> OffsetDateTime {
    let today = OffsetDateTime::now_utc();
    OffsetDateTime::new_utc(today.date(), Time::MIDNIGHT)
}

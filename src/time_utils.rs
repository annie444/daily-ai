use time::{Duration, OffsetDateTime, Time};
use tracing::trace;

/// Seconds between Unix epoch (1970) and macOS epoch (2001).
const MACOS_EPOCH_OFFSET: f64 = 978_307_200.0;

/// Convert an `OffsetDateTime` to macOS timestamp (seconds since 2001-01-01) as f64.
#[tracing::instrument(
    name = "Converting standard date and time to a MacOS timestamp",
    level = "debug"
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
    level = "debug"
)]
pub fn unix_time_sec_to_macos_time(nsecs: i128) -> f64 {
    let ts = nsecs as f64 / 1_000_000_000.0 - MACOS_EPOCH_OFFSET;
    trace!("Converted Unix time {}ns to macOS time {}", nsecs, ts);
    ts
}

/// macOS timestamp (seconds) for the given duration ago (default is 24 hours).
#[tracing::instrument(
    name = "Calculating the date and time in the past as a MacOS timestamp",
    level = "debug"
)]
pub fn macos_past_ts(duration: &Duration) -> f64 {
    let ts = OffsetDateTime::now_utc().saturating_sub(*duration);
    trace!("Calculated the past timestamp: {}", ts);
    unix_time_sec_to_macos_time(ts.unix_timestamp_nanos())
}

/// UTC datetime for the given duration ago (default is 24 hours).
#[tracing::instrument(name = "Calculate the date and time in the past", level = "debug")]
pub fn past_ts(duration: &Duration) -> OffsetDateTime {
    OffsetDateTime::now_utc().saturating_sub(*duration)
}

/// Convert seconds to nanoseconds.
#[tracing::instrument(name = "Converting seconds to nanoseconds", level = "debug")]
pub fn timestamp_secs_to_nsecs(secs: i64) -> i128 {
    (secs as i128) * 1_000_000_000
}

/// Convert Unix time (nanoseconds) to local `OffsetDateTime` for user-facing output.
#[tracing::instrument(name = "Converting a Unix timestamp to date and time", level = "debug")]
pub fn unix_time_nsec_to_datetime(secs: i128) -> OffsetDateTime {
    // Convert to local time for user-facing output.
    OffsetDateTime::from_unix_timestamp_nanos(secs)
        .unwrap()
        .to_offset(OffsetDateTime::now_local().unwrap().offset())
}

/// Convert macOS timestamp (seconds since 2001) to Unix time in nanoseconds.
#[tracing::instrument(
    name = "Converting a MacOS timestamp to a Unix standard timestamp",
    level = "debug"
)]
pub fn macos_to_unix_time(macos_time: f64) -> i128 {
    ((macos_time + MACOS_EPOCH_OFFSET) * 1_000_000_000.0) as i128
}

/// Convert macOS timestamp to UTC `OffsetDateTime`.
#[tracing::instrument(name = "Converting MacOS timestamp to date and time", level = "debug")]
pub fn macos_to_datetime(macos_time: f64) -> OffsetDateTime {
    let secs = macos_to_unix_time(macos_time);
    OffsetDateTime::from_unix_timestamp_nanos(secs).unwrap()
}

/// Midnight (00:00) today in UTC.
#[tracing::instrument(
    name = "Calculating the date and time of today at UTC midnight",
    level = "debug"
)]
pub fn midnight_utc() -> OffsetDateTime {
    let today = OffsetDateTime::now_utc();
    OffsetDateTime::new_utc(today.date(), Time::MIDNIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unix_and_macos_roundtrip() {
        let secs = 1_700_000_000_i64; // fixed epoch seconds
        let nsecs = timestamp_secs_to_nsecs(secs);
        let macos = unix_time_sec_to_macos_time(nsecs);
        let back = macos_to_unix_time(macos);
        assert_eq!(back, nsecs);
    }

    #[test]
    fn datetime_and_macos_roundtrip() {
        let dt = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let macos = datetime_to_macos_time(&dt);
        let back = macos_to_datetime(macos);
        // Allow 1 microsecond drift due to float conversion
        let delta = (back - dt).whole_microseconds().abs();
        assert!(delta <= 1, "drift too large: {}µs", delta);
    }

    #[test]
    fn midnight_is_midnight_utc() {
        let mid = midnight_utc();
        assert_eq!(mid.time(), Time::MIDNIGHT);
        assert_eq!(mid.offset(), OffsetDateTime::now_utc().offset());
    }

    #[test]
    fn macos_past_ts_matches_duration() {
        let dur = Duration::hours(2);
        let past_macos = macos_past_ts(&dur);
        let now = OffsetDateTime::now_utc().unix_timestamp();
        let past = macos_to_unix_time(past_macos) / 1_000_000_000;
        assert_eq!(now - past as i64, dur.whole_seconds());
    }

    #[test]
    fn unix_time_nsec_to_datetime_is_local_offset() {
        let ts = 1_700_000_000_i64;
        let nsecs = timestamp_secs_to_nsecs(ts);
        let dt = unix_time_nsec_to_datetime(nsecs);
        assert_eq!(dt.unix_timestamp(), ts);
    }
}

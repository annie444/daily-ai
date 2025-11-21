use chrono::{DateTime, Days, NaiveDateTime, NaiveTime, Utc};

const MACOS_EPOCH_OFFSET: f64 = 978307200.0;

#[tracing::instrument(level = "trace")]
pub fn datetime_to_macos_time(dt: &NaiveDateTime) -> f64 {
    let utc = dt.and_utc();
    unix_time_nsec_to_macos_time(utc.timestamp(), utc.timestamp_subsec_nanos())
}

#[tracing::instrument(level = "trace")]
pub fn unix_time_nsec_to_macos_time(secs: i64, nsecs: u32) -> f64 {
    secs as f64 + (nsecs as f64 / 1_000_000_000.0) - MACOS_EPOCH_OFFSET
}

#[tracing::instrument(level = "trace")]
pub fn macos_yesterday() -> f64 {
    let ts = Utc::now().checked_sub_days(Days::new(1)).unwrap();
    unix_time_nsec_to_macos_time(ts.timestamp(), ts.timestamp_subsec_nanos())
}

#[tracing::instrument(level = "trace")]
pub fn yesterday() -> NaiveDateTime {
    let ts = Utc::now().checked_sub_days(Days::new(1)).unwrap();
    ts.naive_utc()
}

#[tracing::instrument(level = "trace")]
pub fn unis_time_nsec_to_datetime(secs: i64, nsecs: u32) -> NaiveDateTime {
    DateTime::from_timestamp(secs, nsecs).unwrap().naive_utc()
}

#[tracing::instrument(level = "trace")]
pub fn macos_to_unix_time(macos_time: f64) -> (i64, u32) {
    let timestamp = macos_time + MACOS_EPOCH_OFFSET;
    let secs = timestamp.floor() as i64;
    let nsecs = ((timestamp - secs as f64) * 1_000_000_000.0) as u32;
    (secs, nsecs)
}

#[tracing::instrument(level = "trace")]
pub fn macos_to_datetime(macos_time: f64) -> chrono::NaiveDateTime {
    let (secs, nsecs) = macos_to_unix_time(macos_time);
    DateTime::from_timestamp(secs, nsecs).unwrap().naive_utc()
}

#[tracing::instrument(level = "trace")]
pub fn midnight() -> NaiveDateTime {
    NaiveDateTime::new(Utc::now().date_naive(), NaiveTime::MIN)
}

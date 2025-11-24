use serde::{Deserialize, Deserializer, Serializer};
use time::{
    Duration, OffsetDateTime, PrimitiveDateTime,
    format_description::{BorrowedFormatItem, well_known::Rfc3339},
    macros::format_description,
};

/// Serde helpers for `OffsetDateTime` values.
///
/// Input format: `YYYY-mm-dd HH:MM:SS`
/// Output format: RFC 3339 / ISO 8601 (e.g. `2025-01-02T03:04:05Z`).
pub mod offset_datetime {

    use super::*;

    pub(super) const INPUT_FORMAT: &[BorrowedFormatItem] = format_description!(
        "[year]-[month padding:zero]-[day padding:zero] [hour padding:zero]:[minute padding:zero]:[second padding:zero]"
    );
    pub(super) const OUTPUT_FORMAT: Rfc3339 = Rfc3339;

    /// Serialize an `OffsetDateTime` as RFC 3339.
    pub fn serialize<S>(dt: &OffsetDateTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(
            &dt.format(&OUTPUT_FORMAT)
                .map_err(serde::ser::Error::custom)?
                .to_string(),
        )
    }

    /// Deserialize either the friendly input format or RFC 3339 into an `OffsetDateTime` (UTC).
    pub fn deserialize<'de, D>(deserializer: D) -> Result<OffsetDateTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        PrimitiveDateTime::parse(&raw, INPUT_FORMAT)
            // Be tolerant and accept the output format as input too.
            .or_else(|_| PrimitiveDateTime::parse(&raw, &OUTPUT_FORMAT))
            .map_err(serde::de::Error::custom)
            .map(|pdt| pdt.assume_utc())
    }
}

/// Serde helpers for `std::time::Duration`.
///
/// The duration is represented as an integer followed by a unit suffix.
/// We choose the largest whole unit when serializing to avoid fractional values.
pub mod duration {
    use super::*;

    /// Serialize `Duration` to the largest integral unit (m, s, ms, us, ns).
    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = TryInto::<std::time::Duration>::try_into(*duration)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&format!("{}", humantime::Duration::from(duration)))
    }

    /// Deserialize a duration string like `3m`, `120ms`, `42ns`.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        parse_duration(&raw).map_err(serde::de::Error::custom)
    }

    /// Parse a duration string into `Duration`, preferring exact integer units.
    pub fn parse_duration<S: AsRef<str>>(s: S) -> Result<Duration, String> {
        // Match longest suffixes first to avoid the `m` vs `ms` ambiguity.
        Duration::try_from(
            humantime::parse_duration(s.as_ref())
                .map_err(|e| format!("Failed to parse duration '{}': {}", s.as_ref(), e))?,
        )
        .map_err(|e| format!("Duration value overflowed for '{}': {}", s.as_ref(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[test]
    fn roundtrips_naive_datetime() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct Wrapper {
            #[serde(with = "crate::serde_helpers::offset_datetime")]
            ts: OffsetDateTime,
        }

        let dt = PrimitiveDateTime::parse(
            "2024-12-31 23:59:59",
            crate::serde_helpers::offset_datetime::INPUT_FORMAT,
        )
        .unwrap()
        .assume_utc();
        let value = Wrapper { ts: dt };
        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(
            serialized, "{\"ts\":\"2024-12-31T23:59:59Z\"}",
            "Expected 2024-12-31T23:59:59Z got {}",
            serialized
        );

        let deserialized: Wrapper = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            deserialized, value,
            "Deserialized value did not match original. Got {:?}, expected {:?}",
            deserialized, value
        );
    }

    #[test]
    fn duration_selects_largest_integer_unit() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct Wrapper {
            #[serde(with = "crate::serde_helpers::duration")]
            duration: Duration,
        }

        let duration = Duration::microseconds(120);
        let value = Wrapper { duration };
        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(
            serialized, "{\"duration\":\"120us\"}",
            "Expected 120us got {}",
            serialized
        );

        // Should pick minutes when possible
        let minutes = Wrapper {
            duration: Duration::minutes(3),
        };
        let serialized_minutes = serde_json::to_string(&minutes).unwrap();
        assert_eq!(
            serialized_minutes, "{\"duration\":\"3m\"}",
            "Expected 3m got {}",
            serialized_minutes
        );

        let deserialized: Wrapper = serde_json::from_str(&serialized_minutes).unwrap();
        assert_eq!(
            deserialized, minutes,
            "Deserialized value did not match original. Got {:?}, expected {:?}",
            deserialized, minutes
        );
    }
}

use serde::{Deserialize, Deserializer, Serializer};
use std::time::Duration;
use time::{
    OffsetDateTime, PrimitiveDateTime,
    format_description::{BorrowedFormatItem, well_known::Rfc3339},
    macros::format_description,
};

/// Serde helpers for chrono::NaiveDateTime.
///
/// Input format: `YYYY-mm-dd HH:MM:SS`
/// Output format: ISO 8601 without timezone (e.g. `2025-01-02T03:04:05`).
pub mod offset_datetime {

    use super::*;

    pub(super) const INPUT_FORMAT: &[BorrowedFormatItem] = format_description!(
        "[year]-[month padding:zero]-[day padding:zero] [hour padding:zero]:[minute padding:zero]:[second padding:zero]"
    );
    pub(super) const OUTPUT_FORMAT: Rfc3339 = Rfc3339;

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

/// Serde helpers for chrono::TimeDelta.
///
/// The duration is represented as an integer followed by a unit suffix.
/// We choose the largest whole unit when serializing to avoid fractional values.
pub mod duration {
    use super::*;

    const MINUTE_NS: i128 = 60 * 1_000_000_000;
    const SECOND_NS: i128 = 1_000_000_000;
    const MILLI_NS: i128 = 1_000_000;
    const MICRO_NS: i128 = 1_000;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let secs = duration.as_secs_f64();
        let ns = (secs * 1_000_000_000.0).round() as i128;

        let (value, unit) = if ns % MINUTE_NS == 0 {
            (ns / MINUTE_NS, "m")
        } else if ns % SECOND_NS == 0 {
            (ns / SECOND_NS, "s")
        } else if ns % MILLI_NS == 0 {
            (ns / MILLI_NS, "ms")
        } else if ns % MICRO_NS == 0 {
            (ns / MICRO_NS, "us")
        } else {
            (ns, "ns")
        };

        serializer.serialize_str(&format!("{}{}", value, unit))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        parse_duration(&raw).map_err(serde::de::Error::custom)
    }

    pub fn parse_duration<S: AsRef<str>>(s: S) -> Result<Duration, String> {
        let raw = s.as_ref().trim();
        // Match longest suffixes first to avoid the `m` vs `ms` ambiguity.
        for suffix in ["ns", "us", "ms", "s", "m"] {
            if let Some(value_part) = raw.strip_suffix(suffix) {
                let magnitude = value_part
                    .trim()
                    .parse::<u64>()
                    .map_err(|e| format!("invalid duration '{}': {}", raw, e))?;

                return match suffix {
                    "ns" => Ok(Duration::from_nanos(magnitude)),
                    "us" => Ok(Duration::from_micros(magnitude)),
                    "ms" => Ok(Duration::from_millis(magnitude)),
                    "s" => Ok(Duration::from_secs(magnitude)),
                    "m" => Ok(Duration::from_mins(magnitude)),
                    _ => unreachable!(),
                };
            }
        }

        Err(format!("unsupported duration format: '{}'", raw))
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

        let duration = Duration::from_micros(120);
        let value = Wrapper { duration };
        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(
            serialized, "{\"duration\":\"120us\"}",
            "Expected 120us got {}",
            serialized
        );

        // Should pick minutes when possible
        let minutes = Wrapper {
            duration: Duration::from_mins(3),
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

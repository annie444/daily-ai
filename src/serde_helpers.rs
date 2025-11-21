use chrono::{NaiveDateTime, TimeDelta};
use serde::{Deserialize, Deserializer, Serializer};

/// Serde helpers for chrono::NaiveDateTime.
///
/// Input format: `YYYY-mm-dd HH:MM:SS`
/// Output format: ISO 8601 without timezone (e.g. `2025-01-02T03:04:05`).
pub mod naive_datetime {
    use super::*;

    const INPUT_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
    const OUTPUT_FORMAT: &str = "%Y-%m-%dT%H:%M:%S";

    pub fn serialize<S>(dt: &NaiveDateTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&dt.format(OUTPUT_FORMAT).to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<NaiveDateTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        NaiveDateTime::parse_from_str(&raw, INPUT_FORMAT)
            // Be tolerant and accept the output format as input too.
            .or_else(|_| NaiveDateTime::parse_from_str(&raw, OUTPUT_FORMAT))
            .map_err(serde::de::Error::custom)
    }
}

/// Serde helpers for chrono::TimeDelta.
///
/// The duration is represented as an integer followed by a unit suffix.
/// We choose the largest whole unit when serializing to avoid fractional values.
pub mod duration {
    use super::*;

    const MINUTE_NS: i64 = 60 * 1_000_000_000;
    const SECOND_NS: i64 = 1_000_000_000;
    const MILLI_NS: i64 = 1_000_000;
    const MICRO_NS: i64 = 1_000;

    pub fn serialize<S>(duration: &TimeDelta, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let ns = duration
            .num_nanoseconds()
            .ok_or_else(|| serde::ser::Error::custom("duration too large"))?;

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

    pub fn deserialize<'de, D>(deserializer: D) -> Result<TimeDelta, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        parse_duration(&raw).map_err(serde::de::Error::custom)
    }

    pub fn parse_duration<S: AsRef<str>>(s: S) -> Result<TimeDelta, String> {
        let raw = s.as_ref().trim();
        // Match longest suffixes first to avoid the `m` vs `ms` ambiguity.
        for suffix in ["ns", "us", "ms", "s", "m"] {
            if let Some(value_part) = raw.strip_suffix(suffix) {
                let magnitude = value_part
                    .trim()
                    .parse::<i64>()
                    .map_err(|e| format!("invalid duration '{}': {}", raw, e))?;

                return match suffix {
                    "ns" => Ok(TimeDelta::nanoseconds(magnitude)),
                    "us" => Ok(TimeDelta::microseconds(magnitude)),
                    "ms" => Ok(TimeDelta::milliseconds(magnitude)),
                    "s" => Ok(TimeDelta::seconds(magnitude)),
                    "m" => Ok(TimeDelta::minutes(magnitude)),
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
            #[serde(with = "crate::serde_helpers::naive_datetime")]
            ts: NaiveDateTime,
        }

        let dt = NaiveDateTime::parse_from_str("2024-12-31 23:59:59", "%Y-%m-%d %H:%M:%S").unwrap();
        let value = Wrapper { ts: dt };
        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(serialized, "{\"ts\":\"2024-12-31T23:59:59\"}");

        let deserialized: Wrapper = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, value);
    }

    #[test]
    fn duration_selects_largest_integer_unit() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct Wrapper {
            #[serde(with = "crate::serde_helpers::duration")]
            duration: TimeDelta,
        }

        let duration = TimeDelta::microseconds(120);
        let value = Wrapper { duration };
        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(serialized, "{\"duration\":\"120us\"}");

        // Should pick minutes when possible
        let minutes = Wrapper {
            duration: TimeDelta::minutes(3),
        };
        let serialized_minutes = serde_json::to_string(&minutes).unwrap();
        assert_eq!(serialized_minutes, "{\"duration\":\"3m\"}");

        let deserialized: Wrapper = serde_json::from_str(&serialized_minutes).unwrap();
        assert_eq!(deserialized, minutes);
    }
}

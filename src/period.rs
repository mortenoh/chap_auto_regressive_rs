//! Seasonal feature: the within-year position of a CHAP time period.
//!
//! Mirrors `transforms.year_position_from_period`: parse the period start date and
//! return its day-of-year divided by 365. A period is either monthly (`YYYY-MM`,
//! whose start is the first of the month) or a weekly range
//! (`YYYY-MM-DD/YYYY-MM-DD`, whose start is the first date).

use anyhow::{Context, Result, bail};

/// Return the within-year position of a period's start, in `[0, 1]`.
pub fn year_position(period: &str) -> Result<f64> {
    let (year, month, day) = parse_start(period)?;
    Ok(day_of_year(year, month, day) as f64 / 365.0)
}

/// Parse the `(year, month, day)` of a period's start date.
fn parse_start(period: &str) -> Result<(i32, u32, u32)> {
    let text = period.trim();
    if let Some((first, _)) = text.split_once('/') {
        // Weekly range: "YYYY-MM-DD/YYYY-MM-DD" -> the first ISO date.
        parse_ymd(first)
    } else {
        // Monthly: "YYYY-MM" -> the first of the month.
        let mut parts = text.split('-');
        let year: i32 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .with_context(|| format!("invalid time_period {text:?}"))?;
        let month: u32 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .with_context(|| format!("invalid time_period {text:?}"))?;
        if parts.next().is_some() {
            bail!("invalid monthly time_period {text:?} (expected YYYY-MM)");
        }
        Ok((year, month, 1))
    }
}

/// Parse an ISO `YYYY-MM-DD` date.
fn parse_ymd(text: &str) -> Result<(i32, u32, u32)> {
    let mut parts = text.split('-');
    let year = parts.next().and_then(|s| s.parse().ok());
    let month = parts.next().and_then(|s| s.parse().ok());
    let day = parts.next().and_then(|s| s.parse().ok());
    match (year, month, day) {
        (Some(y), Some(m), Some(d)) => Ok((y, m, d)),
        _ => bail!("invalid ISO date {text:?} (expected YYYY-MM-DD)"),
    }
}

/// 1-based day of the year, matching Python's `datetime.timetuple().tm_yday`.
fn day_of_year(year: i32, month: u32, day: u32) -> u32 {
    const CUMULATIVE: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = CUMULATIVE[(month - 1) as usize] + day;
    if month > 2 && is_leap(year) {
        doy += 1;
    }
    doy
}

/// Proleptic Gregorian leap-year test.
fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monthly_first_of_month() {
        assert_eq!(year_position("2020-01").unwrap(), 1.0 / 365.0);
        // March 1st in a leap year is day 31+29+1 = 61.
        assert_eq!(year_position("2020-03").unwrap(), 61.0 / 365.0);
        // March 1st in a non-leap year is day 31+28+1 = 60.
        assert_eq!(year_position("2021-03").unwrap(), 60.0 / 365.0);
    }

    #[test]
    fn weekly_uses_first_date() {
        assert_eq!(year_position("2020-01-06/2020-01-12").unwrap(), 6.0 / 365.0);
    }
}

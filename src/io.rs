//! CSV input/output matching the documented column contract.
//!
//! Input frames are tidy: one row per location and time period with columns
//! `location, time_period, rainfall, mean_temperature, population` and, for
//! history, `disease_cases` (which may be blank where surveillance is missing).

use std::path::Path;

use anyhow::{Context, Result, bail};

/// One input row.
#[derive(Debug, Clone)]
pub struct Row {
    /// Region identifier.
    pub location: String,
    /// CHAP time period (`YYYY-MM` or `YYYY-MM-DD/YYYY-MM-DD`).
    pub time_period: String,
    /// Rainfall covariate.
    pub rainfall: f64,
    /// Mean temperature covariate.
    pub mean_temperature: f64,
    /// Population covariate.
    pub population: f64,
    /// Observed cases; `None` when the column is absent, `NaN` when blank.
    pub disease_cases: Option<f64>,
}

/// Parse a numeric cell; a blank cell becomes `NaN` (a missing observation).
fn parse_cell(value: &str, column: &str, line: usize) -> Result<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(f64::NAN);
    }
    trimmed
        .parse::<f64>()
        .with_context(|| format!("row {line}: column {column:?} has non-numeric value {value:?}"))
}

/// Read a tidy input frame from a CSV file.
pub fn read_frame(path: &Path) -> Result<Vec<Row>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let index = |name: &str| headers.iter().position(|h| h == name);

    let loc_i = index("location").context("missing 'location' column")?;
    let period_i = index("time_period").context("missing 'time_period' column")?;
    let rain_i = index("rainfall").context("missing 'rainfall' column")?;
    let temp_i = index("mean_temperature").context("missing 'mean_temperature' column")?;
    let pop_i = index("population").context("missing 'population' column")?;
    let cases_i = index("disease_cases");

    let mut rows = Vec::new();
    for (offset, record) in reader.records().enumerate() {
        let line = offset + 2; // 1-based, plus the header line
        let record = record.with_context(|| format!("reading row {line}"))?;
        let get = |i: usize| record.get(i).unwrap_or("");
        let disease_cases = match cases_i {
            Some(i) => Some(parse_cell(get(i), "disease_cases", line)?),
            None => None,
        };
        rows.push(Row {
            location: get(loc_i).to_string(),
            time_period: get(period_i).to_string(),
            rainfall: parse_cell(get(rain_i), "rainfall", line)?,
            mean_temperature: parse_cell(get(temp_i), "mean_temperature", line)?,
            population: parse_cell(get(pop_i), "population", line)?,
            disease_cases,
        });
    }
    if rows.is_empty() {
        bail!("{} contains no data rows", path.display());
    }
    Ok(rows)
}

/// The forecast output: one row per location and future period.
#[derive(Debug, Clone)]
pub struct Forecast {
    /// The forecast period being labeled.
    pub time_period: String,
    /// The location being forecast.
    pub location: String,
    /// The sampled counts for this location and period (one per draw).
    pub samples: Vec<i64>,
}

/// Write forecasts as `time_period, location, sample_0 ... sample_{N-1}`.
pub fn write_forecast(path: &Path, rows: &[Forecast]) -> Result<()> {
    let n_samples = rows.first().map(|r| r.samples.len()).unwrap_or(0);
    let mut writer =
        csv::Writer::from_path(path).with_context(|| format!("creating {}", path.display()))?;

    let mut header = vec!["time_period".to_string(), "location".to_string()];
    header.extend((0..n_samples).map(|i| format!("sample_{i}")));
    writer.write_record(&header)?;

    for row in rows {
        let mut record = vec![row.time_period.clone(), row.location.clone()];
        record.extend(row.samples.iter().map(|s| s.to_string()));
        writer.write_record(&record)?;
    }
    writer.flush()?;
    Ok(())
}

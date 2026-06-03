//! Dense feature/target extraction, mirroring `transforms.get_series`.
//!
//! Locations are grouped in sorted (canonical) label order so each maps to the
//! same embedding index as during training; within a location, rows are sorted by
//! `time_period` (lexicographic order is chronological for both the monthly and the
//! weekly formats). Each period contributes the four features
//! `[rainfall, mean_temperature, population, year_position]`.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use ndarray::{Array2, Array3};

use crate::io::Row;
use crate::period::year_position;

/// The dense arrays for one tidy frame.
pub struct Series {
    /// Sorted location labels.
    pub locations: Vec<String>,
    /// Per-location sorted time-period labels.
    pub periods: Vec<Vec<String>>,
    /// Features, shape `(locations, periods, 4)`.
    pub x: Array3<f64>,
    /// Target, shape `(locations, periods)`; `None` when the frame has no cases.
    pub y: Option<Array2<f64>>,
}

/// Extract the dense `(features, target)` arrays from input rows.
pub fn get_series(rows: &[Row]) -> Result<Series> {
    let has_target = rows.iter().any(|r| r.disease_cases.is_some());

    // Group by location, then sort each group by time_period. BTreeMap keeps the
    // location keys in sorted order.
    let mut groups: BTreeMap<String, Vec<&Row>> = BTreeMap::new();
    for row in rows {
        groups.entry(row.location.clone()).or_default().push(row);
    }
    for group in groups.values_mut() {
        group.sort_by(|a, b| a.time_period.cmp(&b.time_period));
    }

    let locations: Vec<String> = groups.keys().cloned().collect();
    let counts: Vec<usize> = groups.values().map(|g| g.len()).collect();
    if counts
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1
    {
        bail!(
            "every location must have the same number of periods, but the period counts differ: {:?}",
            locations
                .iter()
                .cloned()
                .zip(counts.iter().copied())
                .collect::<Vec<_>>()
        );
    }

    let n_loc = locations.len();
    let n_periods = counts.first().copied().unwrap_or(0);
    let mut x = Array3::<f64>::zeros((n_loc, n_periods, 4));
    let mut y = if has_target {
        Some(Array2::<f64>::zeros((n_loc, n_periods)))
    } else {
        None
    };
    let mut periods = Vec::with_capacity(n_loc);

    for (li, location) in locations.iter().enumerate() {
        let group = &groups[location];
        let mut location_periods = Vec::with_capacity(n_periods);
        for (ti, row) in group.iter().enumerate() {
            x[[li, ti, 0]] = row.rainfall;
            x[[li, ti, 1]] = row.mean_temperature;
            x[[li, ti, 2]] = row.population;
            x[[li, ti, 3]] = year_position(&row.time_period)?;
            if let Some(y) = y.as_mut() {
                // A blank disease_cases cell parses to NaN, which the AR input
                // interpolation later fills; the raw NaN keeps it out of any label.
                y[[li, ti]] = row.disease_cases.unwrap_or(f64::NAN);
            }
            location_periods.push(row.time_period.clone());
        }
        periods.push(location_periods);
    }

    if x.iter().any(|v| v.is_nan()) {
        bail!("input features contain NaN values (rainfall, mean_temperature or population)");
    }

    Ok(Series {
        locations,
        periods,
        x,
        y,
    })
}

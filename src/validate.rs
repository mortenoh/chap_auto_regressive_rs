//! Predict-time input validation, mirroring `model._check_predict_inputs`.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::config::Meta;
use crate::io::Row;

/// Count rows per location, in sorted location order.
fn counts(rows: &[Row]) -> BTreeMap<String, usize> {
    let mut map = BTreeMap::new();
    for row in rows {
        *map.entry(row.location.clone()).or_insert(0) += 1;
    }
    map
}

/// Validate the history/future frames against the model before forecasting.
///
/// Checks (in order): the two frames cover the same location set; the locations
/// equal the model's training locations; each location's history has at least
/// `context_length` periods; each location's future has at most `prediction_length`.
pub fn check_predict_inputs(historic: &[Row], future: &[Row], meta: &Meta) -> Result<()> {
    let historic_counts = counts(historic);
    let future_counts = counts(future);

    let historic_locs: Vec<&String> = historic_counts.keys().collect();
    let future_locs: Vec<&String> = future_counts.keys().collect();
    if historic_locs != future_locs {
        bail!("historic and future must cover the same set of locations");
    }

    if let Some(training) = &meta.locations {
        let mut sorted_training = training.clone();
        sorted_training.sort();
        let actual: Vec<String> = historic_counts.keys().cloned().collect();
        if actual != sorted_training {
            bail!(
                "prediction locations must match the training locations {:?}, but got {:?}",
                sorted_training,
                actual
            );
        }
    }

    if historic_counts.values().any(|&c| c < meta.context_length) {
        let mut seen: Vec<usize> = historic_counts.values().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        bail!(
            "each location's history must have at least context_length={} periods, got period counts {:?}",
            meta.context_length,
            seen
        );
    }

    if future_counts.values().any(|&c| c > meta.prediction_length) {
        let mut seen: Vec<usize> = future_counts.values().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        bail!(
            "each location's future must have at most prediction_length={} periods, got period counts {:?}",
            meta.prediction_length,
            seen
        );
    }

    Ok(())
}

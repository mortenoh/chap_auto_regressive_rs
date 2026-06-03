//! Inference-only Rust port of the `chap_auto_regressive` forecasting model.
//!
//! Loads weights exported from the Python model (`weights.safetensors` + `meta.json`)
//! and reproduces the deterministic `predict` forward pass plus negative-binomial
//! sampling, reading and writing the same tidy CSV contract as the Python model.
//! Only the `base` architecture is supported.

pub mod config;
pub mod features;
pub mod interpolate;
pub mod io;
pub mod layers;
pub mod model;
pub mod period;
pub mod sample;
pub mod scaler;
pub mod validate;
pub mod weights;

use std::path::Path;

use anyhow::{Result, bail};
use ndarray::{Axis, s};

use crate::config::Meta;
use crate::io::{Forecast, Row};
use crate::weights::Params;

/// Default sampling seed (the Python model uses a fixed JAX key; exact draws are
/// not reproducible across the two, so we fix our own).
pub const DEFAULT_SEED: u64 = 1234;

/// A loaded model: its metadata and network parameters.
pub struct LoadedModel {
    pub meta: Meta,
    pub params: Params,
}

impl LoadedModel {
    /// Load `meta.json` and `weights.safetensors` from a weights directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let meta = Meta::load(dir)?;
        if meta.rnn_model_name != "base" {
            bail!(
                "unsupported architecture {:?}; only 'base' is implemented",
                meta.rnn_model_name
            );
        }
        let params = Params::load(dir)?;
        Ok(LoadedModel { meta, params })
    }

    /// Forecast the future periods as `num_samples` draws per location and period.
    pub fn predict(
        &self,
        historic: &[Row],
        future: &[Row],
        num_samples: usize,
        seed: u64,
    ) -> Result<Vec<Forecast>> {
        validate::check_predict_inputs(historic, future, &self.meta)?;

        let context = self.meta.context_length;
        let hist = features::get_series(historic)?;
        let fut = features::get_series(future)?;

        let hist_y = hist
            .y
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("historic frame is missing disease_cases"))?;

        // Take the last `context` periods of history and concatenate with the
        // future covariates along the time axis.
        let th = hist.x.shape()[1];
        let hist_x_ctx = hist.x.slice(s![.., th - context.., ..]);
        let full_x = ndarray::concatenate(Axis(1), &[hist_x_ctx, fut.x.view()])?;

        let scaled = scaler::apply(&self.meta.scaler, full_x.view());

        let hist_y_ctx = hist_y.slice(s![.., th - context..]).to_owned();
        let ar_y = interpolate::interpolate_nans(&hist_y_ctx).mapv(|v| v as f32);

        let eta = model::forward(&self.params, &scaled, &ar_y);
        let eta_slice = model::forecast_slice(&eta, context);

        let counts = sample::sample_counts(&eta_slice, num_samples, seed);

        // Assemble the output, labeling each location with its own future periods.
        let mut out = Vec::new();
        for (li, location) in fut.locations.iter().enumerate() {
            for (ti, period) in fut.periods[li].iter().enumerate() {
                out.push(Forecast {
                    time_period: period.clone(),
                    location: location.clone(),
                    samples: counts[li][ti].clone(),
                });
            }
        }
        Ok(out)
    }
}

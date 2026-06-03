//! The `meta.json` sidecar describing a trained model.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Fitted feature scaler (per-feature mean and standard deviation).
#[derive(Debug, Clone, Deserialize)]
pub struct Scaler {
    /// Per-feature means, in `feature_order`.
    pub mu: Vec<f64>,
    /// Per-feature standard deviations, in `feature_order` (zero-variance features
    /// are stored as `1.0` so standardization leaves them at `0`).
    pub std: Vec<f64>,
}

/// Model metadata exported alongside the weights.
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    /// Architecture name; only `"base"` is supported by this port.
    pub rnn_model_name: String,
    /// Number of past periods the model reads as context.
    pub context_length: usize,
    /// Number of periods the model was trained to forecast.
    pub prediction_length: usize,
    /// Number of training locations (rows in the embedding table).
    pub n_locations: usize,
    /// Size of each per-location embedding vector.
    pub embedding_dim: usize,
    /// Per-period feature names, in the order the network expects them.
    pub feature_order: Vec<String>,
    /// The fitted feature scaler.
    pub scaler: Scaler,
    /// Canonical (sorted) training locations; `None` for legacy exports.
    pub locations: Option<Vec<String>>,
}

impl Meta {
    /// Load and parse `meta.json` from a weights directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let meta: Meta =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(meta)
    }
}

//! The `meta.json` sidecar describing a trained model.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Fitted feature scaler (per-feature mean and standard deviation).
#[derive(Debug, Clone, Deserialize)]
pub struct Scaler {
    pub mu: Vec<f64>,
    pub std: Vec<f64>,
}

/// Model metadata exported alongside the weights.
#[derive(Debug, Clone, Deserialize)]
pub struct Meta {
    pub rnn_model_name: String,
    pub context_length: usize,
    pub prediction_length: usize,
    pub n_locations: usize,
    pub embedding_dim: usize,
    pub feature_order: Vec<String>,
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

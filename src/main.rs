//! CLI for the Rust inference port: a drop-in replacement for CHAP's `predict.py`.
//!
//! ```text
//! chap-ar-predict <weights_dir> <historic.csv> <future.csv> <out.csv> [--samples N] [--seed S]
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use chap_ar_predict::io::{read_frame, write_forecast};
use chap_ar_predict::{DEFAULT_SEED, LoadedModel};

/// Forecast disease cases from history and future covariates.
#[derive(Parser)]
#[command(name = "chap-ar-predict", about, version)]
struct Cli {
    /// Directory holding weights.safetensors and meta.json.
    weights_dir: PathBuf,
    /// History CSV (with observed disease_cases).
    historic: PathBuf,
    /// Future covariates CSV (no disease_cases).
    future: PathBuf,
    /// Output CSV path.
    out: PathBuf,
    /// Number of samples to draw per location and period.
    #[arg(long, default_value_t = 100)]
    samples: usize,
    /// RNG seed for sampling.
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model = LoadedModel::load(&cli.weights_dir)?;
    let historic = read_frame(&cli.historic)?;
    let future = read_frame(&cli.future)?;
    let forecast = model.predict(&historic, &future, cli.samples, cli.seed)?;
    write_forecast(&cli.out, &forecast)?;
    eprintln!(
        "wrote {} rows ({} samples each) to {}",
        forecast.len(),
        cli.samples,
        cli.out.display()
    );
    Ok(())
}

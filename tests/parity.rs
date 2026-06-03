//! Parity tests against the Python model, using the generated fixture.
//!
//! Regenerate the fixture from the Python repo with:
//!   PYTHONPATH=scripts uv run python scripts/make_fixture.py \
//!       ../chap_auto_regressive_rs/tests/fixtures

use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3, Axis};
use safetensors::SafeTensors;

use chap_ar_predict::io::read_frame;
use chap_ar_predict::layers::softplus;
use chap_ar_predict::{LoadedModel, features, interpolate, model, scaler};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Read a float32 tensor from a safetensors file as a flat `(shape, data)` pair.
fn read_tensor(bytes: &[u8], name: &str) -> (Vec<usize>, Vec<f32>) {
    let st = SafeTensors::deserialize(bytes).unwrap();
    let view = st.tensor(name).unwrap();
    let data = view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (view.shape().to_vec(), data)
}

fn tensor3(bytes: &[u8], name: &str) -> Array3<f32> {
    let (shape, data) = read_tensor(bytes, name);
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).unwrap()
}

fn tensor2(bytes: &[u8], name: &str) -> Array2<f32> {
    let (shape, data) = read_tensor(bytes, name);
    Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()
}

fn max_abs_diff(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// The deterministic forward pass must reproduce the Python `eta` to ~1e-5.
#[test]
fn forward_pass_matches_python_eta() {
    let dir = fixtures();
    let model = LoadedModel::load(&dir).unwrap();
    let parity = std::fs::read(dir.join("parity.safetensors")).unwrap();

    let scaled_x = tensor3(&parity, "scaled_x");
    let ar_y = tensor2(&parity, "ar_y");
    let expected = tensor3(&parity, "eta");

    let eta = model::forward(&model.params, &scaled_x, &ar_y);
    assert_eq!(eta.dim(), expected.dim());
    let diff = max_abs_diff(&eta, &expected);
    assert!(diff < 1e-5, "max abs eta diff {diff} exceeds 1e-5");
}

/// The feature extraction + windowing + scaling path must reproduce the exact
/// `scaled_x` array the Python model fed to the network.
#[test]
fn scaled_window_matches_python() {
    let dir = fixtures();
    let model = LoadedModel::load(&dir).unwrap();
    let parity = std::fs::read(dir.join("parity.safetensors")).unwrap();
    let expected_scaled = tensor3(&parity, "scaled_x");
    let expected_ar = tensor2(&parity, "ar_y");

    let historic = read_frame(&dir.join("historic.csv")).unwrap();
    let future = read_frame(&dir.join("future.csv")).unwrap();
    let context = model.meta.context_length;

    let hist = features::get_series(&historic).unwrap();
    let fut = features::get_series(&future).unwrap();
    let th = hist.x.shape()[1];
    let hist_x_ctx = hist.x.slice(ndarray::s![.., th - context.., ..]);
    let full_x = ndarray::concatenate(Axis(1), &[hist_x_ctx, fut.x.view()]).unwrap();
    let scaled = scaler::apply(&model.meta.scaler, full_x.view());

    let hist_y = hist.y.unwrap();
    let hist_y_ctx = hist_y.slice(ndarray::s![.., th - context..]).to_owned();
    let ar_y = interpolate::interpolate_nans(&hist_y_ctx).mapv(|v| v as f32);

    assert!(
        max_abs_diff(&scaled, &expected_scaled) < 1e-6,
        "scaled_x mismatch"
    );
    let ar_diff = ar_y
        .iter()
        .zip(expected_ar.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(ar_diff < 1e-6, "ar_y mismatch: {ar_diff}");
}

/// End-to-end predict: correct frame shape and per-location period labels.
#[test]
fn predict_produces_labeled_frame() {
    let dir = fixtures();
    let model = LoadedModel::load(&dir).unwrap();
    let historic = read_frame(&dir.join("historic.csv")).unwrap();
    let future = read_frame(&dir.join("future.csv")).unwrap();

    let out = model.predict(&historic, &future, 16, 7).unwrap();

    // 2 locations x 2 future periods.
    assert_eq!(out.len(), 4);
    let locations: std::collections::BTreeSet<_> = out.iter().map(|f| f.location.clone()).collect();
    assert_eq!(locations, ["A".to_string(), "B".to_string()].into());
    for f in &out {
        assert_eq!(f.samples.len(), 16);
        assert!(f.samples.iter().all(|&s| s >= 0));
        assert!(["2021-01", "2021-02"].contains(&f.time_period.as_str()));
    }
}

/// The predict-time validation guards must fire with the same conditions as the
/// Python model.
#[test]
fn predict_rejects_bad_inputs() {
    let dir = fixtures();
    let model = LoadedModel::load(&dir).unwrap();
    let historic = read_frame(&dir.join("historic.csv")).unwrap();
    let future = read_frame(&dir.join("future.csv")).unwrap();

    // History shorter than context_length (keep only 2 periods per location).
    let mut short = historic.clone();
    short.retain(|r| r.time_period.as_str() >= "2020-11");
    let err = model
        .predict(&short, &future, 4, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("context_length"), "got: {err}");

    // Future longer than prediction_length (3 periods).
    let mut long_future = future.clone();
    for loc in ["A", "B"] {
        let template = future.iter().find(|r| r.location == loc).unwrap().clone();
        long_future.push(chap_ar_predict::io::Row {
            time_period: "2021-03".to_string(),
            ..template
        });
    }
    let err = model
        .predict(&historic, &long_future, 4, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("prediction_length"), "got: {err}");

    // Locations the model was not trained on.
    let mut unseen_hist = historic.clone();
    let mut unseen_future = future.clone();
    for r in unseen_hist.iter_mut() {
        r.location = if r.location == "A" {
            "C".into()
        } else {
            "D".into()
        };
    }
    for r in unseen_future.iter_mut() {
        r.location = if r.location == "A" {
            "C".into()
        } else {
            "D".into()
        };
    }
    let err = model
        .predict(&unseen_hist, &unseen_future, 4, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("training locations"), "got: {err}");

    // NaN covariate.
    let mut nan_future = future.clone();
    nan_future[0].rainfall = f64::NAN;
    let err = model
        .predict(&historic, &nan_future, 4, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("NaN"), "got: {err}");
}

/// Parity against an external fixture dir (e.g. the real monthly/weekly CHAP
/// models). Runs only when `CHAP_FIXTURE_DIR` points at a dir holding
/// weights.safetensors, meta.json, historic.csv, future.csv and parity.safetensors.
#[test]
fn external_fixture_parity() {
    let Ok(dir) = std::env::var("CHAP_FIXTURE_DIR") else {
        eprintln!("skipping external_fixture_parity (set CHAP_FIXTURE_DIR to run)");
        return;
    };
    let dir = PathBuf::from(dir);
    let model = LoadedModel::load(&dir).unwrap();
    let parity = std::fs::read(dir.join("parity.safetensors")).unwrap();

    // 1. Deterministic forward pass reproduces Python's eta.
    let scaled_x = tensor3(&parity, "scaled_x");
    let ar_y = tensor2(&parity, "ar_y");
    let expected = tensor3(&parity, "eta");
    let eta = model::forward(&model.params, &scaled_x, &ar_y);
    assert_eq!(eta.dim(), expected.dim(), "eta shape mismatch");
    let eta_diff = max_abs_diff(&eta, &expected);
    eprintln!(
        "external eta max abs diff = {eta_diff:e}  (shape {:?})",
        eta.dim()
    );
    assert!(eta_diff < 1e-4, "external eta diff {eta_diff} exceeds 1e-4");

    // 2. The full feature/window/scaling path reproduces Python's scaled_x and ar_y
    //    straight from the CSVs (exercises real time-period formats and NaN cases).
    let historic = read_frame(&dir.join("historic.csv")).unwrap();
    let future = read_frame(&dir.join("future.csv")).unwrap();
    let context = model.meta.context_length;
    let hist = features::get_series(&historic).unwrap();
    let fut = features::get_series(&future).unwrap();
    let th = hist.x.shape()[1];
    let hist_x_ctx = hist.x.slice(ndarray::s![.., th - context.., ..]);
    let full_x = ndarray::concatenate(Axis(1), &[hist_x_ctx, fut.x.view()]).unwrap();
    let scaled = scaler::apply(&model.meta.scaler, full_x.view());
    let scaled_diff = max_abs_diff(&scaled, &scaled_x);
    eprintln!("external scaled_x max abs diff = {scaled_diff:e}");
    assert!(scaled_diff < 1e-4, "external scaled_x diff {scaled_diff}");

    let hist_y = hist.y.unwrap();
    let hist_y_ctx = hist_y.slice(ndarray::s![.., th - context..]).to_owned();
    let ar = interpolate::interpolate_nans(&hist_y_ctx).mapv(|v| v as f32);
    let ar_diff = ar
        .iter()
        .zip(ar_y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("external ar_y max abs diff = {ar_diff:e}");
    assert!(ar_diff < 1e-4, "external ar_y diff {ar_diff}");
}

/// Distributional parity: with many draws the per-period sample mean tracks the
/// analytic negative-binomial mean `softplus(eta0) * exp(eta1)`.
#[test]
fn sample_means_track_analytic_mean() {
    let dir = fixtures();
    let model = LoadedModel::load(&dir).unwrap();
    let parity = std::fs::read(dir.join("parity.safetensors")).unwrap();
    let eta = tensor3(&parity, "eta");
    let context = model.meta.context_length;
    let eta_slice = model::forecast_slice(&eta, context);

    let n = 60_000;
    let counts = chap_ar_predict::sample::sample_counts(&eta_slice, n, 99);
    let (n_loc, future, _) = eta_slice.dim();
    for l in 0..n_loc {
        for t in 0..future {
            let analytic =
                softplus(eta_slice[[l, t, 0]]) as f64 * (eta_slice[[l, t, 1]] as f64).exp();
            let sample_mean = counts[l][t].iter().sum::<i64>() as f64 / n as f64;
            let tol = 0.1 * analytic.max(1.0);
            assert!(
                (sample_mean - analytic).abs() < tol,
                "loc {l} period {t}: sample mean {sample_mean} vs analytic {analytic}"
            );
        }
    }
}

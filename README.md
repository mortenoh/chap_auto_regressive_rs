# chap-ar-predict

[![checks](https://github.com/mortenoh/chap_auto_regressive_rs/actions/workflows/checks.yml/badge.svg)](https://github.com/mortenoh/chap_auto_regressive_rs/actions/workflows/checks.yml)
[![license: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A pure-Rust, inference-only port of the
[`chap_auto_regressive`](../chap_auto_regressive) forecasting model.

Training stays in Python (JAX/Flax). This crate reproduces only the **predict**
path: it loads weights exported from a trained model and produces probabilistic
forecasts, reading and writing the same tidy CSV contract as the Python model. It
has no Python/JAX runtime dependency, which makes it a small, fast, self-contained
serving artifact and a drop-in replacement for CHAP's `predict.py`.

Only the `base` architecture (`rnn_model_name = "base"`) is supported.

## How it works

Train in Python as usual, then export the saved model to a portable directory the
Rust binary can load.

**From a CHAP wrapper** (e.g. [`auto_regressive_monthly`](../auto_regressive_monthly),
[`auto_regressive_weekly`](../auto_regressive_weekly)) the export script lives in
`scripts/` and reads the lengths from that wrapper's `build_model()`, so there is
nothing to pass by hand:

```bash
# in the wrapper repo, after train.py has written model.bin
uv run python scripts/export_weights.py model.bin weights_dir/
```

**From the library directly**, pass the lengths the model was trained with:

```bash
# in chap_auto_regressive/
uv run python scripts/export_weights.py model.pkl weights_dir/ \
    --context-length 24 --prediction-length 3 --rnn-model-name base
```

Either writes `weights_dir/weights.safetensors` (the network parameters) and
`weights_dir/meta.json` (lengths, feature order, the fitted scaler, and the sorted
training locations).

> No model handy? A ready-made example pickle ships at
> `tests/fixtures/model.pkl` (trained with `--context-length 4
> --prediction-length 2`); export that to try the flow end to end.

Then forecast with the Rust binary — a drop-in for the wrapper's `predict.py`:

```bash
chap-ar-predict weights_dir/ historic.csv future.csv out.csv --samples 100
```

- `historic.csv` — columns `location, time_period, rainfall, mean_temperature,
  population, disease_cases` (cases may be blank where surveillance is missing).
- `future.csv` — the same columns without `disease_cases`.
- `out.csv` — `time_period, location, sample_0 ... sample_{N-1}`.

## Parity with the Python model

The port targets **functional equivalence**, not bit-exact reproduction:

- The deterministic forward pass (`eta`) matches the Python model to `< 1e-5` given
  identical weights and inputs.
- Samples are drawn from the **same** negative-binomial distribution
  (`n = softplus(eta0)`, `p = sigmoid(-eta1)`) via a Gamma-Poisson mixture, but use
  an independent seeded RNG, so individual draws differ while the per-period
  distribution (and its mean `n * exp(eta1)`) matches.

These are checked in `tests/parity.rs` against a fixture generated from the Python
model. Regenerate the fixture with:

```bash
# from the Python repo
PYTHONPATH=scripts uv run python scripts/make_fixture.py \
    ../chap_auto_regressive_rs/tests/fixtures
```

### Validated against the real CHAP models

The port was checked end to end against both production wrappers, each trained on
its real input data and forecast with a backtest split:

| Model | context / pred | max `eta` diff | `scaled_x` / `ar_y` | output structure | per-period mean (20k samples) |
| --- | --- | --- | --- | --- | --- |
| `auto_regressive_monthly` | 12 / 3 | 8.9e-8 | exact (0) | identical to `predict.py` | within 1.9% |
| `auto_regressive_weekly` | 52 / 12 | 1.2e-7 | exact (0) | identical to `predict.py` | within 2.1% |

The weekly run also exercises the `YYYY-MM-DD/...` period format and real missing
`disease_cases` (the AR-input interpolation matches Python exactly). To reproduce,
build a fixture with `chap_auto_regressive/scripts/dump_parity.py` and run the
external check:

```bash
CHAP_FIXTURE_DIR=/path/to/fixture cargo test external_fixture_parity -- --nocapture
```

## Performance

CHAP invokes `predict.py` as a fresh process per call, so every prediction pays
Python's full cold-start. Measured end to end (median of 5 fresh runs, 100 samples):

| | Python `predict.py` (cold) | Rust `chap-ar-predict` |
| --- | --- | --- |
| monthly (context 12, pred 3) | 1.77 s | 0.003 s (~600x faster) |
| weekly (context 52, pred 12) | 1.83 s | 0.004 s (~500x faster) |
| Rust at 20,000 samples | — | 0.023 s |

Almost all of Python's time is fixed overhead, not compute:

```
import pandas              158 ms
import jax + flax          231 ms
import model/chap_ar        65 ms
load_predictor             1.3 ms
predict (JAX JIT compile) 1183 ms   <- compiles the forward pass on first call
--------------------------------
TOTAL cold                1638 ms
(same predict, warm:        84 ms)
```

None of that overhead shrinks with smaller inputs -- it is paid on every fresh
process. The Rust binary has no imports and no JIT, so it just runs the matmuls
(~3 ms) and sampling (the only part that scales: 3 ms at 100 -> 23 ms at 20k).

**Worth it when** predictions run as separate processes -- single forecasts, and
especially eval backtests that spawn one predict per fold (100 folds is ~3 min of
pure Python boot vs ~0.3 s here). **Less compelling** if you already batch many
predicts inside one long-lived warm Python process (~84 ms each there). Training
still requires Python/JAX; this port is inference-only.

## Develop

```bash
cargo test          # unit + parity/integration tests
cargo clippy --all-targets
cargo build --release
```

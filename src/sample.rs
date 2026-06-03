//! Negative-binomial sampling of the forecast, mirroring `nb_head` + `NB3.sample`.
//!
//! The head reads `eta` as `n = softplus(eta[..,0])`, `logits = eta[..,1]`, and the
//! distribution is `scipy.stats.nbinom(n, p = sigmoid(-logits))`. That negative
//! binomial is drawn here as a Gamma-Poisson mixture: with `scale = exp(logits)`,
//! `lambda ~ Gamma(shape = n, scale)` and `count ~ Poisson(lambda)` -- the mean is
//! `n * exp(logits)`, matching `NB3.mean`. Draws use an independent seeded RNG
//! (functional, not bit-exact, equivalence with JAX/scipy).

use ndarray::Array3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson};

use crate::layers::softplus;

/// Draw `num_samples` counts per location and forecast period.
///
/// `eta` is `(loc, future, 2)`. Returns `counts[loc][period]` = a vector of
/// `num_samples` draws. The success probability `p = sigmoid(-logits)` is implicit
/// in `scale = exp(logits) = (1 - p) / p`, the Gamma-Poisson scale.
pub fn sample_counts(eta: &Array3<f32>, num_samples: usize, seed: u64) -> Vec<Vec<Vec<i64>>> {
    let (n_loc, future, _) = eta.dim();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n_loc);

    for l in 0..n_loc {
        let mut location = Vec::with_capacity(future);
        for t in 0..future {
            let n = softplus(eta[[l, t, 0]]) as f64;
            let logits = eta[[l, t, 1]] as f64;
            let scale = logits.exp();
            let mut draws = Vec::with_capacity(num_samples);

            // Guard the degenerate parameters that would make Gamma/Poisson fail.
            let gamma = Gamma::new(n.max(f64::MIN_POSITIVE), scale.max(f64::MIN_POSITIVE)).ok();
            for _ in 0..num_samples {
                let lambda = gamma.as_ref().map(|g| g.sample(&mut rng)).unwrap_or(0.0);
                let count = if lambda.is_finite() && lambda > 0.0 {
                    Poisson::new(lambda)
                        .map(|p| p.sample(&mut rng).round() as i64)
                        .unwrap_or(0)
                } else {
                    0
                };
                draws.push(count);
            }
            location.push(draws);
        }
        out.push(location);
    }
    out
}

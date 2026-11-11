# Bayesian conformal risk control under temporal dependence

Snell and Griffiths' [Conformal Prediction as Bayesian Quadrature](https://arxiv.org/abs/2502.13228) (ICML 2025) assumes exchangeable data. This repository tests how far its HPD risk guarantee fails when the calibration data are autocorrelated, explains the failure with a simple approximation, and tests a correction. Everything is simulation based.

This is a fork of [jakesnell/conformal-as-bayes-quad](https://github.com/jakesnell/conformal-as-bayes-quad). The library in `src/bcp` is the original authors' work. The temporal dependence study is new: [`src/bcp/dependence.py`](src/bcp/dependence.py), [`scripts/dependence.py`](scripts/dependence.py), [`tests/test_dependence.py`](tests/test_dependence.py) and [`results/`](results/).

## Setup

Scores are a Gaussian copula transform of a latent AR(1) or seasonal AR process, so the marginal stays Uniform(0,1) and only the dependence strength φ changes. The loss is 1{S > λ} with target risk 0.1. The method should have true risk above target in at most 5% of trials (a "failure"). Each setting uses 5,000 trials, with exact binomial 95% intervals.

## Results

**1. Failures rise under dependence and stay high as n grows.**

| | φ = 0 | φ = 0.6 | φ = 0.9 |
|---|---|---|---|
| plain HPD, n = 100 | 3.8% | 11.6% | 35.9% |

At φ = 0.9 the rate is 36.8% at n = 100, 32.8% at n = 300 and 32.0% at n = 1,000 for AR(1), and 31.0%, 31.6% and 32.8% for the seasonal process. More data does not bring it back to 5% (figure: [`ncal_failure.png`](results/ncal_failure.png)).

**2. A variance inflation approximation explains the plateau.**

The posterior treats the n points as independent, but the average loss varies τ times more, where τ is the integrated autocorrelation time of the loss series. This factor does not depend on n, so the failure rate tends to Φ(−z/√τ) and levels off.

| process | τ | predicted | observed (95% CI) |
|---|---|---|---|
| AR(1), φ = 0.3 | 1.27 | 5.7% | 5.66% [5.04, 6.34] |
| AR(1), φ = 0.6 | 2.15 | 11.3% | 11.64% [10.76, 12.56] |
| seasonal-5, φ = 0.9 | 9.6 | 28.3% | 29.06% [27.80, 30.34] |
| AR(1), φ = 0.9, n = 100 | 10.0 | 28.7% | 35.90% [34.57, 37.25] |
| AR(1), φ = 0.9, n = 1,000 | 9.9 | 30.0% | 32.04% [30.75, 33.35] |

It matches under moderate dependence and understates failure when dependence is very strong and n is small (effective sample size around 10). This is a heuristic, not a proof (figure: [`plateau_check.png`](results/plateau_check.png)).

**3. Replacing n with an effective sample size reduces failures.**

The Dirichlet concentration is set to 1/τ̂ instead of 1, with τ̂ estimated from past scores by an autoregressive sieve (rank normal scores, Yule-Walker AR(p), order chosen by AIC). With φ = 0.9 and n = 100:

| process | plain HPD | lag-1 correction | sieve correction |
|---|---|---|---|
| AR(1) | 35.9% | 12.6% | 12.7% |
| seasonal-5 | 29.1% | 28.8% | 9.1% |

The seasonal process has almost no lag-1 autocorrelation, so a lag-1 correction does nothing there, while the sieve picks up the dependence (figure: [`sieve_failure.png`](results/sieve_failure.png)). As n grows, plain HPD to sieve failure rates at φ = 0.9 are:

| n | AR(1) | seasonal-5 |
|---|---|---|
| 100 | 36.8% → 12.9% | 31.0% → 8.7% |
| 300 | 32.8% → 4.3% | 31.6% → 6.1% |
| 1,000 | 32.0% → 2.5% | 32.8% → 5.1% |

## Limitations

- Simulations only, with one loss, one target risk and a 95% HPD level. No real temporal dataset is used.
- The sieve is fitted to a separate history of 1,000 past scores. A shorter history would give noisier estimates, which is not tested here.
- The sieve estimates τ for the latent score process, not the thresholded loss, so it over-corrects when n is large (2.5% for AR(1) at n = 1,000). At n = 100 it still leaves about 13% failures for AR(1).
- The correction has no general guarantee.
- Small differences between tables (for example 35.9% and 36.8%) come from separate runs with different random draws.

## Reproducing

Requires [uv](https://github.com/astral-sh/uv) and [just](https://github.com/casey/just).

```bash
just test   # unit tests
just all    # all experiments, slow at 5,000 trials
```

Individual commands are in the [`justfile`](justfile). Use `--num_trials 250` on `scripts/dependence.py` for a quick run. Re-running restarts the seed per setting, so numbers can shift by a point or two.

## Credit

Method and library by [Jake Snell](https://jakesnell.com) and [Tom Griffiths](https://cocosci.princeton.edu/tom/tom.php). Released under the MIT license.

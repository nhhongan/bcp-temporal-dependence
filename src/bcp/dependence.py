"""Tools for studying Bayesian-quadrature (HPD) risk control under temporal dependence.

Contents
--------
- ``gen_latent`` / ``dependent_uniform``: dependent score streams whose Uniform(0,1) marginal is the
  same for every dependence strength, so dependence is the only manipulated variable.
- ``hpd_threshold_conc``: HPD threshold with a tunable Dirichlet concentration ``conc``
  (``conc = n_eff / n`` deflates the posterior to account for dependence).
- ``c_lag1`` / ``c_sieve``: estimators of ``conc`` from a history of past scores.
- ``tau_loss`` / ``predicted_failure``: the asymptotic failure-rate approximation.
- ``failure_summary``: failure rate with an exact binomial confidence interval.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.scipy.stats import norm
from jaxopt import Bisection
from scipy import stats as sp_stats
from scipy.stats import multivariate_normal as mvn

from bcp.losses import scaled_count_loss
from bcp.utils import predict_identity


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def gen_latent(key, phi, n: int, lag: int = 1) -> Array:
    """Standardized Gaussian process ``z_t = phi * z_{t-lag} + sqrt(1 - phi^2) * e_t``.

    The marginal is N(0, 1) for every ``phi``. ``lag=1`` gives AR(1); ``lag=5`` gives a seasonal
    AR whose lag-1 autocorrelation is zero but whose integrated autocorrelation time matches AR(1).
    """
    eps = jr.normal(key, (n,))
    init = eps[:lag]

    def step(buf, e):
        new = phi * buf[0] + jnp.sqrt(1.0 - phi**2) * e
        return jnp.concatenate([buf[1:], new[None]]), new

    _, rest = jax.lax.scan(step, init, eps[lag:])
    return jnp.concatenate([init, rest])


def dependent_uniform(key, phi, n: int, lag: int = 1) -> Array:
    """Gaussian-copula transform of ``gen_latent``: dependent scores with Uniform(0,1) marginal."""
    return norm.cdf(gen_latent(key, phi, n, lag))


# ---------------------------------------------------------------------------
# HPD threshold with tunable concentration
# ---------------------------------------------------------------------------


def hpd_threshold_conc(
    cal,
    key,
    conc,
    target_risk: float,
    hpd_level: float = 0.95,
    num_dir: int = 300,
    max_risk: float = 1.0,
    max_threshold: float = 1.0,
    risk_bin: str = "last",
) -> Array:
    """HPD threshold using Dirichlet(conc, ..., conc) weights; ``conc=1`` is the original method.

    ``risk_bin`` selects which Dirichlet coordinate carries the unobserved worst-case loss
    ``max_risk``: ``"last"`` (the extra coordinate U_{n+1}, eq. (27) of the paper) or ``"first"``
    (matches ``bcp.thresholds.hpd_threshold``).
    """
    n = len(cal)
    w = jr.dirichlet(key, conc * jnp.ones(n + 1), shape=(num_dir,))
    extra = w[:, -1] if risk_bin == "last" else w[:, 0]

    def optimality_fun(thr):
        losses = scaled_count_loss(predict_identity(cal.scores, thr), cal.labels)
        l_plus = w[:, :-1] @ losses + extra * max_risk
        return jnp.quantile(l_plus, hpd_level) - target_risk

    return Bisection(
        optimality_fun=optimality_fun, lower=0.0, upper=max_threshold, check_bracket=False
    ).run().params


# ---------------------------------------------------------------------------
# Effective-sample-size estimators (concentration c = n_eff / n)
# ---------------------------------------------------------------------------


def c_lag1(u: Array, gamma: float = 1.0, floor: float = 0.02) -> Array:
    """Parametric correction: AR(1) formula ``((1-rho)/(1+rho))**gamma`` from the lag-1
    autocorrelation of the Gaussian latent. Blind to dependence at other lags."""
    z = norm.ppf(jnp.clip(u, 1e-6, 1 - 1e-6))
    z = z - z.mean()
    rho = jnp.clip((z[:-1] * z[1:]).sum() / (z * z).sum(), 0.0, 0.99)
    return jnp.clip(((1.0 - rho) / (1.0 + rho)) ** gamma, floor, 1.0)


def c_sieve(u: Array, pmax: int = 12, floor: float = 0.02) -> Array:
    """Model-free correction ``c = 1 / tau`` where ``tau`` is the integrated autocorrelation time
    estimated from an AR(p) sieve (Yule-Walker, ``p`` chosen by AIC) fitted to rank normal-scores.

    No copula or AR order is assumed, so it picks up seasonal dependence that ``c_lag1`` misses.
    """
    n = u.shape[0]
    x = norm.ppf((jnp.argsort(jnp.argsort(u)) + 0.5) / n)  # rank normal-scores
    x = x - x.mean()
    g0 = (x * x).sum() / n
    r = jnp.stack([(x[: n - k] * x[k:]).sum() / n / g0 for k in range(1, pmax + 1)])
    aics, taus = [jnp.float64(0.0)], [jnp.float64(1.0)]  # p = 0 (white noise)
    for p in range(1, pmax + 1):
        idx = jnp.abs(jnp.arange(p)[:, None] - jnp.arange(p)[None, :])
        R = jnp.where(idx == 0, 1.0, r[jnp.maximum(idx - 1, 0)])
        a = jnp.linalg.solve(R, r[:p])
        s2 = jnp.maximum(1.0 - a @ r[:p], 1e-6)
        aics.append(n * jnp.log(s2) + 2 * p)
        taus.append(s2 / jnp.maximum((1.0 - a.sum()) ** 2, 1e-6))  # long-run var / var
    tau = jnp.stack(taus)[jnp.argmin(jnp.stack(aics))]
    return jnp.clip(1.0 / jnp.maximum(tau, 1.0), floor, 1.0)


# ---------------------------------------------------------------------------
# Asymptotic failure-rate approximation
# ---------------------------------------------------------------------------


def tau_loss(phi: float, lam: float, kmax: int = 500) -> float:
    """Integrated autocorrelation time ``1 + 2 sum_k Corr(l_t, l_{t+k})`` of the loss series
    ``l_t = 1{S_t > lam}`` for a Gaussian AR(1) latent (exact via the bivariate normal CDF)."""
    q = sp_stats.norm.ppf(lam)
    p = 1 - lam
    tau = 1.0
    for k in range(1, kmax + 1):
        r = phi**k
        if r < 1e-7:
            break
        p11 = mvn(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-q, -q])  # P(S_t>q, S_{t+k}>q)
        tau += 2 * (p11 - p * p) / (p * (1 - p))
    return tau


def predicted_failure(tau: float, z: float) -> float:
    """Plateau ``Phi(-z / sqrt(tau))`` of the plain-HPD failure probability (independent of n)."""
    return float(sp_stats.norm.cdf(-z / np.sqrt(tau)))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def failure_summary(risks, target_risk: float) -> dict:
    """Data-conditional failure rate (% of trials with true risk > target) and exact 95% CI."""
    n = int(risks.shape[0])
    k = int((risks > target_risk).sum())
    ci = sp_stats.binomtest(k, n).proportion_ci(confidence_level=0.95)
    return dict(
        failure_pct=round(100 * k / n, 2),
        ci_low_pct=round(100 * ci.low, 2),
        ci_high_pct=round(100 * ci.high, 2),
    )

"""Experiments: Bayesian-quadrature (HPD) risk control under temporal dependence.

Sub-commands (each writes a CSV to --out_file; run `python scripts/dependence.py COMMAND --help`):

  baseline    Plain HPD / CRC on AR(1) calibration data. Shows the failure-rate inflation.
  ess-gamma   First attempt at a fix: Dirichlet concentration (1-rho)/(1+rho))^gamma from lag-1 rho.
  sieve       Model-free fix (AR(p) sieve ESS) vs. lag-1 fix vs. plain HPD on AR(1) and seasonal-5.
  sieve-ncal  Same, sweeping the calibration size to test whether the failure floor persists.
  plateau     Compare the asymptotic prediction Phi(-z/sqrt(tau)) to the `sieve`/`sieve-ncal` CSVs.
  plot        Figures from the CSVs.

Setup common to all simulations: scores S_t are a Gaussian-copula transform of a (seasonal) AR(1)
latent, so the marginal is Uniform(0,1) for every dependence strength phi. Loss is
1{S > lambda}; the true risk of a threshold is evaluated on N_VAL iid validation points. A
"failure" is a trial whose true risk exceeds the target. See `bcp.dependence` for the components.
"""

import functools

import click
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from bcp.data_set import DataSet
from bcp.dependence import (
    c_lag1,
    c_sieve,
    dependent_uniform,
    failure_summary,
    hpd_threshold_conc,
    predicted_failure,
    tau_loss,
)
from bcp.losses import scaled_count_loss
from bcp.thresholds import conformal_risk_control_threshold
from bcp.utils import predict_identity

jax.config.update("jax_enable_x64", True)

TARGET_RISK = 0.1
HPD_LEVEL, NUM_DIR = 0.95, 300  # 95% HPD level -> nominal 5% failure rate
N_VAL = 2000
DGPS = {"AR(1)": 1, "seasonal-5": 5}  # name -> lag of the latent recursion


def _true_risk(lam, key):
    uval = jr.uniform(key, (N_VAL,))  # iid validation, same Uniform(0,1) marginal
    return scaled_count_loss(predict_identity(uval[:, None], lam), uval[:, None]).mean()


def _simulate(run_batch, num_trials, batch_size, phi, **static):
    """Run `num_trials` Monte Carlo trials in batches; returns concatenated per-trial outputs."""
    outs, key, done = [], jr.key(0), 0
    while done < num_trials:
        key, sk = jr.split(key)
        outs.append(run_batch(jr.split(sk, batch_size), jnp.float64(phi), **static))
        done += batch_size
    return [jnp.concatenate(x) for x in zip(*outs)]


def _row(risks, lams, **extra):
    return dict(
        **extra, **failure_summary(risks, TARGET_RISK), mean_threshold=round(float(lams.mean()), 3)
    )


def _common(f):
    for opt in reversed(
        [
            click.option("--num_trials", default=5000, show_default=True),
            click.option("--batch_size", default=250, show_default=True),
            click.option("--out_file", required=True, help="CSV to write"),
        ]
    ):
        f = opt(f)
    return f


@click.group()
def cli():
    pass


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("method", "n_cal"))
def _baseline_batch(keys, phi, method, n_cal):
    def trial(key):
        kd, kv, kt = jr.split(key, 3)
        ucal = dependent_uniform(kd, phi, n_cal)
        cal = DataSet(jnp.zeros((n_cal, 0)), ucal[:, None])
        if method == "hpd":  # conc = 1: the original method (uses the library's bin convention)
            lam = hpd_threshold_conc(
                cal, kt, 1.0, TARGET_RISK, HPD_LEVEL, NUM_DIR, risk_bin="first"
            )
        else:
            lam = conformal_risk_control_threshold(
                cal, predict_identity, scaled_count_loss, TARGET_RISK, 1.0, 1.0
            )
        return lam, _true_risk(lam, kv)

    return jax.vmap(trial)(keys)


@cli.command()
@_common
@click.option("--n_cal", default=100, show_default=True)
@click.option("--phis", default="0,0.3,0.6,0.9,0.95", show_default=True)
def baseline(num_trials, batch_size, out_file, n_cal, phis):
    rows = []
    for method in ["hpd", "crc"]:
        for phi in map(float, phis.split(",")):
            lams, risks = _simulate(
                _baseline_batch, num_trials, batch_size, phi, method=method, n_cal=n_cal
            )
            rows.append(
                _row(risks, lams, method=method, phi=phi, n_cal=n_cal, n_trials=len(risks))
            )
            print(rows[-1])
    pd.DataFrame(rows).to_csv(out_file, index=False)


# ---------------------------------------------------------------------------
# ess-gamma  (exploratory, lag-1 estimate with exponent gamma)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("n_cal",))
def _ess_batch(keys, phi, gamma, n_cal):
    def trial(key):
        kd, kv, kt = jr.split(key, 3)
        ucal = dependent_uniform(kd, phi, n_cal)
        cal = DataSet(jnp.zeros((n_cal, 0)), ucal[:, None])
        c = c_lag1(ucal, gamma=gamma, floor=0.002)  # estimated from the calibration block itself
        lam = hpd_threshold_conc(cal, kt, c, TARGET_RISK, HPD_LEVEL, NUM_DIR, risk_bin="first")
        return lam, _true_risk(lam, kv), c

    return jax.vmap(trial)(keys)


@cli.command("ess-gamma")
@_common
@click.option("--gammas", default="1.0,1.5,2.0", show_default=True)
@click.option("--n_cal", default=100, show_default=True)
@click.option("--phis", default="0,0.3,0.6,0.9,0.95", show_default=True)
def ess_gamma(num_trials, batch_size, out_file, gammas, n_cal, phis):
    rows = []
    for gamma in map(float, gammas.split(",")):
        for phi in map(float, phis.split(",")):
            lams, risks, cs = _simulate(
                _ess_batch, num_trials, batch_size, phi, gamma=jnp.float64(gamma), n_cal=n_cal
            )
            rows.append(
                _row(risks, lams, method="hpd_ess", gamma=gamma, phi=phi,
                     mean_c=round(float(cs.mean()), 3))
            )
            print(rows[-1])
    pd.DataFrame(rows).to_csv(out_file, index=False)


# ---------------------------------------------------------------------------
# sieve / sieve-ncal
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("method", "lag", "n_cal", "n_hist"))
def _sieve_batch(keys, phi, method, lag, n_cal, n_hist):
    """Dependence is estimated on `n_hist` past scores; HPD is calibrated on the latest `n_cal`."""

    def trial(key):
        kd, kv, kt = jr.split(key, 3)
        u = dependent_uniform(kd, phi, n_hist + n_cal, lag)
        hist, ucal = u[:n_hist], u[n_hist:]
        cal = DataSet(jnp.zeros((n_cal, 0)), ucal[:, None])
        if method == "hpd":
            c = jnp.float64(1.0)
        elif method == "hpd_lag1":
            c = c_lag1(hist)
        else:
            c = c_sieve(hist)
        lam = hpd_threshold_conc(cal, kt, c, TARGET_RISK, HPD_LEVEL, NUM_DIR)
        return lam, _true_risk(lam, kv), c

    return jax.vmap(trial)(keys)


def _sieve_rows(methods, phis, n_cals, num_trials, batch_size, n_hist):
    rows = []
    for dgp, lag in DGPS.items():
        for method in methods:
            for n_cal in n_cals:
                # smaller batches for larger n_cal (memory)
                bs = min(batch_size, max(50, batch_size * 100 // n_cal))
                for phi in phis:
                    lams, risks, cs = _simulate(
                        _sieve_batch, num_trials, bs, phi,
                        method=method, lag=lag, n_cal=n_cal, n_hist=n_hist,
                    )
                    mc = float(cs.mean())
                    rows.append(
                        _row(risks, lams, dgp=dgp, method=method, phi=phi, n_cal=n_cal,
                             n_eff=round(n_cal * mc, 1), mean_c=round(mc, 3))
                    )
                    print(rows[-1])
    return rows


@cli.command()
@_common
@click.option("--n_cal", default=100, show_default=True)
@click.option("--n_hist", default=1000, show_default=True)
@click.option("--phis", default="0,0.3,0.6,0.9", show_default=True)
def sieve(num_trials, batch_size, out_file, n_cal, n_hist, phis):
    rows = _sieve_rows(
        ["hpd", "hpd_lag1", "hpd_sieve"], list(map(float, phis.split(","))),
        [n_cal], num_trials, batch_size, n_hist,
    )
    pd.DataFrame(rows).to_csv(out_file, index=False)


@cli.command("sieve-ncal")
@_common
@click.option("--n_cals", default="100,300,1000", show_default=True)
@click.option("--n_hist", default=1000, show_default=True)
@click.option("--phi", default=0.9, show_default=True)
def sieve_ncal(num_trials, batch_size, out_file, n_cals, n_hist, phi):
    rows = _sieve_rows(
        ["hpd", "hpd_sieve"], [phi], list(map(int, n_cals.split(","))),
        num_trials, batch_size, n_hist,
    )
    pd.DataFrame(rows).to_csv(out_file, index=False)


# ---------------------------------------------------------------------------
# plateau: asymptotic prediction vs. observation
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--sieve_csv", default="results/sieve.csv", show_default=True)
@click.option("--ncal_csv", default="results/sieve_ncal.csv", show_default=True)
@click.option("--out_dir", default="results", show_default=True)
def plateau(sieve_csv, ncal_csv, out_dir):
    """Failure -> Phi(-z / sqrt(tau_loss)), with z calibrated at phi = 0."""
    df = pd.read_csv(sieve_csv)
    df = df[df.method == "hpd"]
    rows = []
    for dgp, g in df.groupby("dgp", sort=False):
        z = norm.ppf(1 - g[g.phi == 0.0].failure_pct.iloc[0] / 100)  # absorbs finite-n slack
        for _, r in g.iterrows():
            t = tau_loss(r.phi, r.mean_threshold)
            rows.append(dict(dgp=dgp, phi=r.phi, tau_loss=round(t, 2), z=round(z, 3),
                             predicted_pct=round(100 * predicted_failure(t, z), 1),
                             observed_pct=r.failure_pct,
                             ci=f"[{r.ci_low_pct}, {r.ci_high_pct}]"))
    a = pd.DataFrame(rows)
    a.to_csv(f"{out_dir}/plateau_vs_phi.csv", index=False)
    print("Plain HPD, failure vs phi (n_cal = 100)\n", a.to_string(index=False))

    z = a[a.phi == 0.0].z.iloc[0]
    b = pd.read_csv(ncal_csv)
    b = b[b.method == "hpd"].copy()
    b["tau_loss"] = [round(tau_loss(p, lam), 2) for p, lam in zip(b.phi, b.mean_threshold)]
    b["pred_calib_z_pct"] = [round(100 * predicted_failure(t, z), 1) for t in b.tau_loss]
    b["pred_asym_z_pct"] = [round(100 * predicted_failure(t, norm.ppf(0.95)), 1) for t in b.tau_loss]
    cols = ["dgp", "n_cal", "tau_loss", "pred_calib_z_pct", "pred_asym_z_pct", "failure_pct"]
    b[cols].to_csv(f"{out_dir}/plateau_vs_ncal.csv", index=False)
    print("\nPlain HPD at phi = 0.9, failure vs calibration size\n", b[cols].to_string(index=False))

    grid = np.linspace(0, 0.95, 40)
    plt.figure(figsize=(5.5, 4))
    plt.plot(grid, [100 * predicted_failure(tau_loss(p, 0.92), z) for p in grid], "k--",
             label=r"asymptotic: $\Phi(-z/\sqrt{\tau})$")
    for dgp, g in a.groupby("dgp", sort=False):
        plt.plot(g.phi, g.observed_pct, "o", label=f"observed, {dgp}")
    plt.axhline(5, c="gray", lw=0.8, label="nominal 5%")
    plt.xlabel(r"dependence $\phi$"); plt.ylabel("plain HPD failure rate (%)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/plateau_check.png", dpi=150)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def _errorbar(ax, g, **kw):
    ax.errorbar(g.phi, g.failure_pct, yerr=[g.failure_pct - g.ci_low_pct,
                g.ci_high_pct - g.failure_pct], marker="o", capsize=3, **kw)


@cli.command()
@click.option("--results_dir", default="results", show_default=True)
def plot(results_dir):
    """Headline figures: failure inflation, ESS-fix comparison, and the n_cal sweep."""
    d = results_dir
    base = pd.read_csv(f"{d}/baseline.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m, g in base.groupby("method"):
        _errorbar(ax, g, label=m.upper())
    ax.axhline(5, ls="--", c="gray", label="nominal 5%")
    ax.set_xlabel(r"AR(1) dependence $\phi$"); ax.set_ylabel("failure rate (%)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{d}/baseline_failure.png", dpi=150)

    sv = pd.read_csv(f"{d}/sieve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (dgp, g0) in zip(axes, sv.groupby("dgp", sort=False)):
        for m, g in g0.groupby("method", sort=False):
            _errorbar(ax, g, label=m)
        ax.axhline(5, ls="--", c="gray")
        ax.set_title(dgp); ax.set_xlabel(r"dependence $\phi$")
    axes[0].set_ylabel("failure rate (%)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(f"{d}/sieve_failure.png", dpi=150)

    nc = pd.read_csv(f"{d}/sieve_ncal.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (dgp, m), g in nc.groupby(["dgp", "method"], sort=False):
        ax.errorbar(g.n_cal, g.failure_pct, yerr=[g.failure_pct - g.ci_low_pct,
                    g.ci_high_pct - g.failure_pct], marker="o", capsize=3, label=f"{m}, {dgp}",
                    ls="-" if m == "hpd_sieve" else ":")
    ax.axhline(5, ls="--", c="gray"); ax.set_xscale("log")
    ax.set_xlabel("calibration size $n$"); ax.set_ylabel(r"failure rate (%), $\phi=0.9$")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{d}/ncal_failure.png", dpi=150)

    gm = pd.read_csv(f"{d}/ess_gamma.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for gamma, g in gm.groupby("gamma"):
        _errorbar(a1, g, label=rf"HPD+ESS ($\gamma$={gamma})")
        a2.plot(g.phi, g.mean_threshold, marker="o", label=rf"HPD+ESS ($\gamma$={gamma})")
    b = base[base.method == "hpd"]
    _errorbar(a1, b, label="HPD", color="tab:orange")
    a2.plot(b.phi, b.mean_threshold, marker="s", color="tab:orange", label="HPD")
    a1.axhline(5, ls="--", c="gray")
    a1.set_title("Validity"); a2.set_title("Efficiency cost (higher = wider sets)")
    a1.set_ylabel("failure rate (%)"); a2.set_ylabel("mean threshold")
    for a in (a1, a2):
        a.set_xlabel(r"$\phi$"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{d}/ess_gamma.png", dpi=150)
    print(f"figures written to {d}/")


if __name__ == "__main__":
    cli()

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bcp.data_set import DataSet
from bcp.dependence import (
    c_lag1,
    c_sieve,
    dependent_uniform,
    failure_summary,
    gen_latent,
    hpd_threshold_conc,
    predicted_failure,
    tau_loss,
)
from bcp.losses import scaled_count_loss
from bcp.thresholds import hpd_threshold
from bcp.utils import predict_identity


def test_marginal_is_standard_normal_for_any_phi():
    z = gen_latent(jr.key(0), 0.9, 20000, lag=5)
    assert abs(float(z.mean())) < 0.3
    assert abs(float(z.std()) - 1.0) < 0.1


def test_lag_structure():
    z = np.asarray(gen_latent(jr.key(1), 0.9, 50000, lag=5))

    def ac(k):
        return np.corrcoef(z[:-k], z[k:])[0, 1]

    assert abs(ac(1)) < 0.05
    assert abs(ac(5) - 0.9) < 0.05


def test_hpd_conc_one_first_bin_matches_library():
    u = jr.uniform(jr.key(2), (50,))
    cal = DataSet(jnp.zeros((50, 0)), u[:, None])
    key = jr.key(3)
    ours = hpd_threshold_conc(cal, key, 1.0, 0.1, risk_bin="first")
    ref = hpd_threshold(cal, predict_identity, scaled_count_loss, 0.1, 1.0, 1.0, 0.95, 300, key)
    assert jnp.allclose(ours, ref)


def test_smaller_concentration_is_more_conservative():
    u = jr.uniform(jr.key(4), (100,))
    cal = DataSet(jnp.zeros((100, 0)), u[:, None])
    key = jr.key(5)
    assert hpd_threshold_conc(cal, key, 0.1, 0.1) > hpd_threshold_conc(cal, key, 1.0, 0.1)


def test_estimators_detect_dependence():
    iid = dependent_uniform(jr.key(6), 0.0, 1000)
    ar1 = dependent_uniform(jr.key(7), 0.9, 1000)
    seasonal = dependent_uniform(jr.key(8), 0.9, 1000, lag=5)
    assert c_sieve(iid) > 0.5
    assert c_sieve(ar1) < 0.2
    assert c_sieve(seasonal) < 0.2
    assert c_lag1(seasonal) > 0.5  # lag-1 estimator is blind to seasonal dependence


def test_tau_loss_and_prediction():
    assert abs(tau_loss(0.0, 0.9) - 1.0) < 1e-9
    assert tau_loss(0.9, 0.9) > tau_loss(0.6, 0.9) > 1.0
    assert predicted_failure(1.0, 1.645) < predicted_failure(10.0, 1.645)


def test_failure_summary():
    s = failure_summary(jnp.array([0.05, 0.2, 0.3, 0.01]), 0.1)
    assert s["failure_pct"] == 50.0
    assert s["ci_low_pct"] < 50.0 < s["ci_high_pct"]

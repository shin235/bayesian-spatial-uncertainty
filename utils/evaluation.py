from config import KL_WEIGHT
from .bayesian_model import build_bayesian_pm25
from .prediction import mc_predict

import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions


def compute_crps_mc(y_true, mc_samples):
    """
    Computes the Continuous Ranked Probability Score (CRPS) from Monte Carlo samples.
    """

    y_true = y_true.squeeze()
    N = y_true.shape[0]
    crps_vals = []
    for i in range(N):
        sample_i = mc_samples[:, i]
        yi = y_true[i]
        crps_i = np.mean(np.abs(sample_i - yi)) - 0.5 * \
            np.mean(np.abs(sample_i[:, None] - sample_i[None, :]))
        crps_vals.append(crps_i)
    return np.mean(crps_vals)


def evaluate_mc(saved_models, cov_test, phi_test, y_test, n_samples=200):
    """
    Evaluates the model performance using Monte Carlo sampling.

    Returns: mean predictions, list of means/stds per model, and evaluation metrics (RMSE, NLL, CRPS).
    """

    all_preds = []
    loc_list = []
    scale_list = []

    for path, _ in saved_models:
        model = build_bayesian_pm25(
            input_dim_phi=phi_test.shape[1],
            input_dim_cov=cov_test.shape[1],
            kl_weight=KL_WEIGHT
        )
        model.load_weights(path)

        samples, locs, scales = mc_predict(
            model, cov_test, phi_test, n_samples=n_samples)

        all_preds.append(samples)
        loc_list.extend(locs)
        scale_list.extend(scales)
        N = phi_test.shape[0]

    all_preds = np.array(all_preds).reshape(-1, N)
    mean_preds = np.mean(all_preds, axis=0)

    rmse = np.sqrt(np.mean((mean_preds - y_test.squeeze())**2))

    # NLL estimation for LogNormal
    nlls = []
    for loc, scale in zip(loc_list, scale_list):
        lognormal_dist = tfd.LogNormal(loc=loc, scale=scale)
        nll_i = -lognormal_dist.log_prob(y_test.squeeze())
        nlls.append(nll_i)
    nll = np.mean(nlls)

    crps = compute_crps_mc(y_test, all_preds)

    print("\nEvaluation Summary (Monte Carlo)")
    print(f"RMSE: {rmse:.3f}")
    print(f"NLL(LogNormal): {nll:.3f}")
    print(f"CRPS: {crps:.3f}")

    return mean_preds, loc_list, scale_list, rmse, nll, crps

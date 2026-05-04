"""
Ideal Observer model — sklearn-style estimator.
Assumes obs shape: n_options x n_trials (NaN = unobserved).
"""

import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# --- Core computations ---


def _transmat(vol, n_levels):
    T = np.eye(n_levels) * (1 - vol)
    T[T == 0] = vol / (n_levels - 1)
    return T


def _uniform_prior(n_levels, n_options=1):
    prior = np.ones(n_levels) / n_levels
    return np.tile(prior[:, None], (1, n_options)) if n_options > 1 else prior


def _likelihood(obs, latent_levels, sd):
    if np.isnan(obs):
        return np.ones(len(latent_levels))
    return sps.norm.pdf(obs, latent_levels, sd)


def _posterior(obs, sd, latent_levels, prior):
    lik = _likelihood(obs, latent_levels, sd)
    post = lik * prior
    return post / post.sum()


def _predict_prior(post, vol, n_levels):
    return _transmat(vol, n_levels) @ post


# --- Learning params parsing (mirrors delta rule style) ---


def _parse_learning_params(learning_params: dict) -> pd.DataFrame:
    rows = []
    for key, val in learning_params.items():
        parts = key.split("_", 1)
        row = {"base": parts[0], "param": val}
        if len(parts) > 1:
            for split in parts[1].split("_"):
                split_id, split_level = split.rsplit(".", 1)
                row[split_id] = split_level
        rows.append(row)
    return pd.DataFrame(rows).fillna(value=pd.NA)


def _get_splitids(spec: pd.DataFrame) -> list:
    return [c for c in spec.columns if c not in ("base", "param")]


def _get_single_vol(
    spec,
    splitids,
    obs,
    prior,
    latent_levels,
    trial_conditions=None,
    outcome_values=None,
    outcome_prior=None,
):

    if "cu" in splitids:
        mask &= spec["cu"] == ("c" if not np.isnan(obs) else "u")

    if "PE.sign" in splitids:
        expectation = latent_levels @ prior
        pe = obs - expectation
        sign = (
            "pos" if pe > 0 else "neg" if pe < 0 else np.random.choice(["pos", "neg"])
        )
        mask &= spec["PE.sign"] == sign

    if "PEO.sign" in splitids:
        if outcome_prior is None or outcome_values is None:
            raise ValueError(
                "PEO.sign split requires return_outcome_dist=True and outcome_range."
            )
        expectation = outcome_values @ outcome_prior
        pe = obs - expectation
        sign = (
            "pos" if pe > 0 else "neg" if pe < 0 else np.random.choice(["pos", "neg"])
        )
        mask &= spec["PEO.sign"] == sign

    if trial_conditions is not None:
        for col in splitids:
            if col not in ("cu", "PE.sign") and col in trial_conditions.index:
                mask &= spec[col] == str(trial_conditions[col])

    return float(spec.loc[mask, "param"].values[0])


def _latent2outcome(lat_dist, latent_levels, outcome_values, sd, normalize=True):
    """Mixture of Gaussians: sum_k P(level_k) * N(outcome | level_k, sd)"""
    out = np.stack(
        [sps.norm.pdf(outcome_values, loc=lv, scale=sd) for lv in latent_levels]
    )  # (n_levels, n_outcomes)
    result = lat_dist @ out  # (n_outcomes,)
    return result / result.sum() if normalize else result


# --- Main sequence function ---


def ideal_observer(
    obs,
    latent_levels,
    sd,
    learning_params,
    conditions=None,
    priors_init=None,
    return_outcome_dist=False,
    outcome_range=None,
    normalize_outcome_dist=True,
):

    n_options, n_trials = obs.shape
    latent_levels = np.asarray(latent_levels)
    n_levels = len(latent_levels)
    sd = np.ones_like(obs) * sd if np.isscalar(sd) else np.asarray(sd)
    priors_init = (
        _uniform_prior(n_levels, n_options)
        if priors_init is None
        else np.asarray(priors_init)
    )
    conditions = pd.DataFrame(conditions) if conditions is not None else None

    spec = _parse_learning_params(learning_params)
    splitids = _get_splitids(spec)

    pred_post = np.full((n_levels, n_options, n_trials + 1), np.nan)
    prev_post = pred_post.copy()
    pred_post[:, :, 0] = priors_init

    if return_outcome_dist:
        outcome_values = np.arange(outcome_range[0], outcome_range[1] + 1)
        n_out = len(outcome_values)
        pred_out = np.full((n_out, n_options, n_trials + 1), np.nan)
        prev_out = pred_out.copy()
        # init outcome prior from latent prior
        for iopt in range(n_options):
            pred_out[:, iopt, 0] = _latent2outcome(
                priors_init[:, iopt] if n_options > 1 else priors_init,
                latent_levels,
                outcome_values,
                sd[iopt, 0],
                normalize_outcome_dist,
            )
    else:
        outcome_values = None

    for iopt in range(n_options):
        for itrial in range(n_trials):
            prior = pred_post[:, iopt, itrial]
            o = obs[iopt, itrial]
            s = sd[iopt, itrial]
            trial_cond = conditions.loc[itrial] if conditions is not None else None
            outcome_prior = pred_out[:, iopt, itrial] if return_outcome_dist else None

            vol = _get_single_vol(
                spec,
                splitids,
                o,
                prior,
                latent_levels,
                trial_cond,
                outcome_values,
                outcome_prior,
            )
            post = _posterior(o, s, latent_levels, prior)
            next_prior = _predict_prior(post, vol, n_levels)

            prev_post[:, iopt, itrial + 1] = post
            pred_post[:, iopt, itrial + 1] = next_prior

            if return_outcome_dist:
                prev_out[:, iopt, itrial + 1] = _latent2outcome(
                    post, latent_levels, outcome_values, s, normalize_outcome_dist
                )
                pred_out[:, iopt, itrial + 1] = _latent2outcome(
                    next_prior, latent_levels, outcome_values, s, normalize_outcome_dist
                )

    posteriors = {"predictive_posterior": pred_post, "prevol_posterior": prev_post}
    if return_outcome_dist:
        posteriors["predictive_outcome_distribution"] = pred_out
        posteriors["prevol_outcome_distribution"] = prev_out
    return posteriors


# --- Sklearn estimator ---


class IdealObserverEstimator(BaseEstimator):
    """
    Parameters
    ----------
    latent_levels : array-like
        Possible reward levels (e.g. [30, 50, 70]).
    sd : float or array (n_options x n_trials)
        Observation noise.
    learning_params : dict
        Volatility, optionally split (e.g. {'vol': 0.1} or {'vol_cu.c': 0.2, 'vol_cu.u': 0.05}).
    option_cols : list of str
        Columns in X with per-option observations.
    condition_cols : list of str, optional
        Columns in X used as trial conditions for split volatilities.
    """

    def __init__(
        self,
        latent_levels,
        sd,
        learning_params,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
    ):
        self.latent_levels = latent_levels
        self.sd = sd
        self.learning_params = learning_params
        self.option_cols = option_cols
        self.condition_cols = condition_cols

    def _parse_X(self, X):
        obs = X[list(self.option_cols)].to_numpy().T  # (n_options, n_trials)
        conditions = (
            X[list(self.condition_cols)].reset_index(drop=True)
            if self.condition_cols
            else None
        )
        return obs, conditions

    def fit(self, X: pd.DataFrame, y=None):
        obs, conditions = self._parse_X(X)
        self.posteriors_ = ideal_observer(
            obs=obs,
            latent_levels=self.latent_levels,
            sd=self.sd,
            learning_params=self.learning_params,
            conditions=conditions,
        )
        self.option_cols_ = list(self.option_cols)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Returns the predictive posterior expectation per option per trial.
        Shape: (n_trials, n_options).
        """
        check_is_fitted(self, "posteriors_")
        levels = np.asarray(self.latent_levels)
        pred = self.posteriors_["predictive_posterior"][
            :, :, :-1
        ]  # (n_levels, n_options, n_trials)
        expectations = np.einsum("l,lot->ot", levels, pred).T  # (n_trials, n_options)
        cols = ["IO__" + c for c in self.option_cols_]
        return pd.DataFrame(expectations, columns=cols, index=X.index)

    def fit_predict(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).predict(X)

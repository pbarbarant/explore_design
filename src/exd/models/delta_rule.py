import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _parse_learning_params(learning_params: dict) -> pd.DataFrame:
    """Parse {'lr-cu.c_sd.low': 0.3, ...} into a structured spec DataFrame."""
    rows = []
    for key, val in learning_params.items():
        parts = key.split("_", 1)  # base vs splits
        row = {"base": parts[0], "param": val}
        if len(parts) > 1:
            for split in parts[1].split("_"):
                split_id, split_level = split.rsplit(".", 1)
                row[split_id] = split_level
        rows.append(row)
    return pd.DataFrame(rows).fillna(value=pd.NA)


def _get_splitids(spec: pd.DataFrame) -> list:
    """Return split dimension column names (everything except base and param)."""
    return [c for c in spec.columns if c not in ("base", "param")]


def get_single_lr(
    spec, splitids, is_chosen, prediction_error, trial_conditions=None
):
    mask = pd.Series([True] * len(spec), index=spec.index)

    if "cu" in splitids:
        mask &= spec["cu"] == ("c" if is_chosen else "u")

    if "PE.sign" in splitids:
        sign = (
            "pos"
            if prediction_error > 0
            else "neg"
            if prediction_error < 0
            else np.random.choice(["pos", "neg"])
        )
        mask &= spec["PE.sign"] == sign

    if trial_conditions is not None:
        for col in splitids:
            if col in trial_conditions.index and col not in ("cu", "PE.sign"):
                mask &= spec[col] == str(trial_conditions[col])

    return float(spec.loc[mask, "param"].values[0])


def delta_rule(
    obs, learning_params: dict, conditions=None, initial_prediction=50
):
    n_options, n_trials = obs.shape
    spec = _parse_learning_params(learning_params)
    splitids = _get_splitids(spec)

    reward_predictions = np.full((n_options, n_trials + 1), np.nan)
    reward_predictions[:, 0] = initial_prediction

    for iopt, opt_trials in enumerate(obs):
        for itrial, obs_trial in enumerate(opt_trials):
            trial_conditions = (
                conditions.loc[itrial] if conditions is not None else None
            )
            is_chosen = not np.isnan(obs_trial)
            prediction_error = (
                obs_trial - reward_predictions[iopt, itrial]
                if is_chosen
                else initial_prediction - reward_predictions[iopt, itrial]
            )
            lr = get_single_lr(
                spec, splitids, is_chosen, prediction_error, trial_conditions
            )
            reward_predictions[iopt, itrial + 1] = (
                reward_predictions[iopt, itrial] + lr * prediction_error
            )

    return reward_predictions


class DeltaRuleEstimator(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible estimator wrapping the delta rule learning model.

    Parameters
    ----------
    learning_params : dict or pd.DataFrame
        Learning rates, optionally split by cu, ff, pe, sd dimensions.
    initial_prediction : float
        Initial reward prediction for all options (default=50).
    option_cols : list of str
        Columns in X containing observed outcomes per option (e.g. ['obsA', 'obsB']).
    condition_cols : list of str, optional
        Columns in X to use as trial conditions (e.g. ['forced', 'SD']).
    """

    def __init__(
        self,
        learning_params,
        initial_prediction=50,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
    ):
        self.learning_params = learning_params
        self.initial_prediction = initial_prediction
        self.option_cols = option_cols
        self.condition_cols = condition_cols

    def _parse_X(self, X: pd.DataFrame):
        """Extract obs array and conditions from events dataframe."""
        obs = X[list(self.option_cols)].to_numpy().T  # (n_options, n_trials)

        conditions = None
        if self.condition_cols is not None:
            conditions = X[list(self.condition_cols)].reset_index(drop=True)

        return obs, conditions

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the model by running the delta rule over trials.

        Parameters
        ----------
        X : pd.DataFrame
            Events dataframe with onset, duration, and trial columns.
        y : ignored
        """
        obs, conditions = self._parse_X(X)

        self.reward_predictions_ = delta_rule(
            obs=obs,
            learning_params=self.learning_params,
            conditions=conditions,
            initial_prediction=self.initial_prediction,
        )
        self.n_options_ = obs.shape[0]
        self.n_trials_ = obs.shape[1]
        return self

    def predict(self, X: pd.DataFrame):
        """
        Return reward predictions for each option and trial.

        Returns
        -------
        pd.DataFrame, shape (n_trials, n_options)
            Predicted values aligned to trials in X.
        """
        check_is_fitted(self, "reward_predictions_")
        # predictions[:, :-1] are pre-trial predictions (what the model expected)
        preds = self.reward_predictions_[:, :-1].T  # (n_trials, n_options)
        output_cols = ["DR__" + name for name in self.option_cols]
        return pd.DataFrame(preds, columns=list(output_cols), index=X.index)

    def fit_predict(self, X: pd.DataFrame, y=None):
        """Fit model and return predictions in one step."""
        return self.fit(X, y).predict(X)

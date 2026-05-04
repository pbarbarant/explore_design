import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class MotorEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, option_cols=()):
        self.option_cols = option_cols

    def fit(self, X: pd.DataFrame, y=None):
        def _func(x):
            if x == "A":
                return "R"
            elif x == "B":
                return "L"
            else:
                return pd.NA

        motor_response = X["arm_choice"].apply(_func)
        self.output_col = pd.DataFrame({"trial_type": motor_response})
        return self

    def predict(self, X: pd.DataFrame):
        return self.output_col

    def fit_predict(self, X: pd.DataFrame, y=None):
        # check_is_fitted(self, "reward_predictions_")
        return self.fit(X, y).predict(X)

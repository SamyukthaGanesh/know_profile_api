import numpy as np
import pandas as pd
from typing import Dict, Any

class MiniLogReg:
    def __init__(self):
        self.w = None
        self.b = 0.0
        self.mean_ = None
        self.std_ = None
        self.num_cols = None

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.num_cols].values - self.mean_) / self.std_
        return Z

    def fit(self, X: pd.DataFrame, y: np.ndarray, lr: float = 0.05, steps: int = 800):
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        arr = X[self.num_cols].values.astype(float)
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0) + 1e-9
        Z = (arr - self.mean_) / self.std_

        n, d = Z.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(steps):
            z = Z.dot(self.w) + self.b
            p = 1/(1+np.exp(-z))
            grad_w = Z.T.dot(p - y) / n
            grad_b = (p - y).mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._prep(X)
        z = Z.dot(self.w) + self.b
        return 1/(1+np.exp(-z))

    def feature_importance(self) -> Dict[str, float]:
        return {c: float(abs(w)) for c, w in zip(self.num_cols, self.w)}

def label_eligibility(features: pd.DataFrame) -> pd.Series:
    elig = ((features["credit_score"] >= 700) &
            (features["dti_ratio"] < 0.45) &
            ((features.get("recurring_Rent",0) >= 4) | (features.get("recurring_Utilities",0) >= 4)))
    return elig.astype(int)

MODEL = MiniLogReg()
ARTIFACTS: Dict[str, Any] = {}

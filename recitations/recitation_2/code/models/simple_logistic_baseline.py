import os
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from scipy.special import logsumexp

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class SimpleLogisticBaseline:
    """
    Simple binary logistic baseline for 2-alternative choice sets.
    """

    def __init__(self, random_state=42, max_iter=1000):
        # Pipeline so features are scaled automatically
        self.clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(random_state=random_state, max_iter=max_iter))
        ])
        self.feature_names_ = None
        self.fitted_ = False

    @staticmethod
    def _build_pairs(X, y, obs_ids, alternatives):
        """
        Convert per-row (alt-level) data into per-choice-set pairs.
        Returns:
            X_pair: (N_sets, n_features) with x_alt2 - x_alt1
            y_pair: (N_sets,) 1 if alt2 chosen, else 0
            pair_index: list of tuples (i1, i2) mapping back to original row indices
        """
        # Put everything in a DataFrame
        df = pd.DataFrame({
            "obs": obs_ids,
            "alt": alternatives,
            "y": y
        })
        # Attach features as columns with fixed names
        n_features = X.shape[1]
        for j in range(n_features):
            df[f"x{j}"] = X[:, j]

        X_pairs = []
        y_pairs = []
        pair_index = []

        # Group by choice set
        for obs, g in df.groupby("obs", sort=True):
            # Expect exactly two rows, alts 1 and 2
            if len(g) != 2:
                continue
            g = g.sort_values("alt")
            if not np.array_equal(g["alt"].values, [1, 2]):
                continue

            # original indices in the provided arrays
            i1, i2 = g.index.values.tolist()

            # extract features
            x1 = np.array([g.iloc[0][f"x{j}"] for j in range(n_features)], dtype=float)
            x2 = np.array([g.iloc[1][f"x{j}"] for j in range(n_features)], dtype=float)

            # build difference (alt2 - alt1)
            x_diff = x2 - x1

            # build label: 1 if alt2 chosen, else 0
            y_rows = g["y"].values
            if y_rows.sum() != 1:
                continue
            y_pair = int(y_rows[1] == 1)

            X_pairs.append(x_diff)
            y_pairs.append(y_pair)
            pair_index.append((i1, i2))

        if len(X_pairs) == 0:
            # No valid pairs -> return empty arrays
            return np.empty((0, n_features), dtype=float), np.empty((0,), dtype=int), []

        return np.vstack(X_pairs), np.array(y_pairs, dtype=int), pair_index

    def fit(self, X, y, obs_ids, alternatives, feature_names=None):
        """
        Train on pair-differences (alt2 - alt1). One row per choice set.
        """
        X_pair, y_pair, _ = self._build_pairs(X, y, obs_ids, alternatives)
        if X_pair.shape[0] == 0:
            raise ValueError("No valid 2-alternative choice sets found for training.")

        self.feature_names_ = feature_names if feature_names is not None else ["price", "time", "comfort", "change"][:X.shape[1]]
        self.clf.fit(X_pair, y_pair)
        self.fitted_ = True
        return self

    def predict_proba(self, X, obs_ids, alternatives):
        """
        Return per-row probabilities P(chosen=1 for that row), consistent with MNL output:
          For each choice set:
            p = P(alt2 chosen | set) from the binary model
            assign:
              prob(row alt1) = 1 - p
              prob(row alt2) = p
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before calling predict_proba.")

        # Build pairs to get mapping back to rows
        X_dummy_y = np.zeros(len(X), dtype=int)  # y not needed, just placeholder
        X_pair, _, pair_index = self._build_pairs(X, X_dummy_y, obs_ids, alternatives)

        # Default 0.5 if set is malformed / not predicted
        probs = np.full(len(X), 0.5, dtype=float)

        if X_pair.shape[0] == 0:
            return probs

        # p_alt2 for each set
        p_alt2 = self.clf.predict_proba(X_pair)[:, 1]

        # Map back to rows
        for k, (i1, i2) in enumerate(pair_index):
            p = p_alt2[k]
            probs[i1] = 1.0 - p   # alt1 row
            probs[i2] = p         # alt2 row

        return probs

    def coef_summary(self):
        """
        Return a small summary for interpretability: coefficients on (alt2 - alt1).
        Positive coef => higher value increases probability of choosing alternative 2.
        """
        if not self.fitted_:
            raise ValueError("Fit the model first.")
        coefs = self.clf.named_steps["logit"].coef_.ravel()
        return pd.DataFrame({
            "feature_on_(alt2-alt1)": self.feature_names_,
            "coef": coefs
        })
    
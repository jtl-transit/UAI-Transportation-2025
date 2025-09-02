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


class MultinomialLogit:
    """
    Multinomial Logit model for discrete choice analysis.

    The utility function for alternative j is:
    U_j = β₀_j + β₁*price_j + β₂*time_j + β₃*comfort_j + β₄*change_j + ε_j

    Where ε_j follows a Type I Extreme Value distribution.
    """

    def __init__(self, include_constants=True, scale=False, random_state=42, maxiter=1000):
        self.include_constants = include_constants
        self.scale = scale
        self.random_state = random_state
        self.maxiter = maxiter

        self.params = None
        self.param_names = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.convergence_info = None

        self.scaler_ = None
        self.n_alts_ = None
        self.n_features_ = None
        self.groups_ = None       # list of np arrays (row indices per choice set)
        self.alt_by_row_ = None   # alternatives (as given) aligned with X rows
        self.y_ = None            # y aligned with X rows
        self.feature_names_ = None

    @staticmethod
    def _build_groups(obs_ids):
        """Return list of index arrays, one per choice set (obs_id)."""
        obs_ids = np.asarray(obs_ids)
        groups = []
        for obs in np.unique(obs_ids):
            idx = np.flatnonzero(obs_ids == obs)
            if idx.size >= 2:     # keep only valid sets
                groups.append(idx)
        return groups

    def _split_params(self, params):
        """Split theta into (ASC(s), beta)."""
        if self.include_constants:
            asc = params[:self.n_alts_ - 1]     # reference alt = 1
            beta = params[self.n_alts_ - 1:]
        else:
            asc = np.zeros(self.n_alts_ - 1)
            beta = params
        return asc, beta

    def _utilities_for_group(self, asc, beta, Xg, altg):
        """
        Compute utilities for a group:
          U_j = ASC_j + x_j^T beta, with ASC_1 fixed to 0
        alt labels are assumed 1..J (if not, we remap at fit-time).
        """
        # attribute part
        u = Xg @ beta
        # add ASCs (alt 2..J have indices 0..J-2 in asc)
        add = np.zeros(len(altg), dtype=float)
        mask = (altg > 1)
        add[mask] = asc[altg[mask] - 2]
        return u + add

    def _neg_loglik(self, params, X, y, obs_ids, alternatives):
        asc, beta = self._split_params(params)

        total = 0.0
        for idx in self.groups_:
            Xg = X[idx]
            yg = y[idx]
            altg = alternatives[idx]

            # basic sanity: exactly one chosen in the set
            if yg.sum() != 1:
                # skip malformed sets
                continue

            ug = self._utilities_for_group(asc, beta, Xg, altg)
            total += (ug[yg == 1][0] - logsumexp(ug))
        return -total

    def fit(self, X, y, obs_ids, alternatives, feature_names=None):
        """
        X: (N, p) alt-level features
        y: (N,) 1 if chosen row, else 0
        obs_ids: (N,) choice-set id (grouping key)
        alternatives: (N,) alternative label per row (prefer 1..J)
        """
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        obs_ids = np.asarray(obs_ids)
        alternatives = np.asarray(alternatives, int)

        # Optional scaling (recommended if features have very different scales)
        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            Xs = self.scaler_.transform(X)
        else:
            Xs = X

        # Remap alternatives to 1..J (in case labels aren’t normalized)
        unique_alts = np.sort(np.unique(alternatives))
        alt_map = {a: i + 1 for i, a in enumerate(unique_alts)}
        alt_norm = np.vectorize(alt_map.get)(alternatives)
        self.n_alts_ = len(unique_alts)

        # Build groups (by obs_id)
        groups = self._build_groups(obs_ids)

        # Keep only well-formed groups (>= 2 rows and exactly one chosen)
        good_groups = []
        for idx in groups:
            if y[idx].sum() == 1 and len(idx) >= 2:
                good_groups.append(idx)
        if len(good_groups) == 0:
            raise ValueError("No valid choice sets (with exactly one chosen) found.")

        self.groups_ = good_groups
        self.n_features_ = Xs.shape[1]
        self.feature_names_ = (
            feature_names if feature_names is not None
            else [f"x{i+1}" for i in range(self.n_features_)]
        )

        # Parameter names
        if self.include_constants:
            self.param_names = [f"ASC_alt_{j}" for j in range(2, self.n_alts_ + 1)] + self.feature_names_
            n_params = (self.n_alts_ - 1) + self.n_features_
        else:
            self.param_names = self.feature_names_
            n_params = self.n_features_

        # Store aligned copies for objective
        self.alt_by_row_ = alt_norm
        self.y_ = y

        # Initialization: small negatives for "cost-like" features often help.
        rng = np.random.default_rng(self.random_state)
        theta0 = rng.normal(0.0, 0.1, n_params)

        # Optimize
        result = minimize(
            fun=self._neg_loglik,
            x0=theta0,
            args=(Xs, y, obs_ids, alt_norm),
            method="BFGS",
            options={"maxiter": self.maxiter, "disp": False}
        )

        self.params = result.x
        self.convergence_info = result
        self.log_likelihood = -result.fun

        # AIC/BIC per number of *choice sets* (not rows)
        n_sets = len(self.groups_)
        k = n_params
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n_sets) - 2 * self.log_likelihood

        print("MNL fitted",
              f"| LL = {self.log_likelihood:.3f}",
              f"| AIC = {self.aic:.2f}",
              f"| BIC = {self.bic:.2f}",
              f"| Converged = {result.success}")

        return self

    def predict_proba(self, X, obs_ids, alternatives):
        """
        Return per-row probabilities (same length/order as X).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")

        X = np.asarray(X, float)
        obs_ids = np.asarray(obs_ids)
        alternatives = np.asarray(alternatives, int)

        if self.scale and self.scaler_ is not None:
            Xs = self.scaler_.transform(X)
        else:
            Xs = X

        # Remap alts to 1..J using the mapping seen at fit-time
        # (If new alts appear, this will raise.)
        unique_fit = np.arange(1, self.n_alts_ + 1)
        # Build a mapping from raw alt labels to 1..J using ranks of unique labels in this call:
        unique_now = np.sort(np.unique(alternatives))
        if len(unique_now) != self.n_alts_:
            raise ValueError("Predict encountered a different number of alternatives than during fit.")
        alt_map_now = {a: i + 1 for i, a in enumerate(unique_now)}
        alt_norm = np.vectorize(alt_map_now.get)(alternatives)

        asc, beta = self._split_params(self.params)

        probs = np.zeros(len(X), dtype=float)
        # group by obs_id and softmax within each set
        for idx in self._build_groups(obs_ids):
            Xg = Xs[idx]
            altg = alt_norm[idx]
            ug = self._utilities_for_group(asc, beta, Xg, altg)
            pg = np.exp(ug - logsumexp(ug))
            probs[idx] = pg
        return probs

    def get_parameter_summary(self):
        if self.params is None:
            raise ValueError("Fit the model first.")
        return pd.DataFrame({
            "Parameter": self.param_names,
            "Estimate": self.params
        })
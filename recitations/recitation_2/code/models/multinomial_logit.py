import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import logsumexp

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
        self.max_iter = maxiter  # Changed from maxiter to max_iter

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

    def _neg_loglik_and_grad(self, params, X, y, obs_ids, alternatives):
        """
        Compute negative log-likelihood and its gradient simultaneously for efficiency.
        
        Returns:
        --------
        tuple : (negative log-likelihood, gradient)
        """
        asc, beta = self._split_params(params)
        
        total_ll = 0.0
        grad_asc = np.zeros(self.n_alts_ - 1)  # gradient w.r.t. ASCs
        grad_beta = np.zeros(self.n_features_)  # gradient w.r.t. betas
        
        for idx in self.groups_:
            Xg = X[idx]
            yg = y[idx]
            altg = alternatives[idx]
            
            # Skip malformed sets
            if yg.sum() != 1:
                continue
                
            # Compute utilities
            ug = self._utilities_for_group(asc, beta, Xg, altg)
            
            # Numerical stability: subtract max before exponential
            ug_max = np.max(ug)
            ug_stable = ug - ug_max
            exp_u = np.exp(ug_stable)
            sum_exp_u = np.sum(exp_u)
            
            # Log-likelihood contribution
            chosen_idx = np.where(yg == 1)[0][0]
            total_ll += ug_stable[chosen_idx] - np.log(sum_exp_u)
            
            # Gradient computation
            probs = exp_u / sum_exp_u  # choice probabilities
            
            # Gradient w.r.t. beta (feature parameters)
            grad_beta += Xg[chosen_idx] - np.sum(probs[:, np.newaxis] * Xg, axis=0)
            
            # Gradient w.r.t. ASCs (alternative specific constants)
            if self.include_constants:
                for j, alt in enumerate(altg):
                    if alt > 1:  # alt 1 is reference
                        asc_idx = alt - 2
                        if j == chosen_idx:
                            grad_asc[asc_idx] += 1.0
                        grad_asc[asc_idx] -= probs[j]
        
        # Combine gradients
        if self.include_constants:
            grad = np.concatenate([grad_asc, grad_beta])
        else:
            grad = grad_beta
            
        return -total_ll, -grad

    def _neg_loglik(self, params, X, y, obs_ids, alternatives):
        """Original negative log-likelihood function (kept for compatibility)"""
        ll, _ = self._neg_loglik_and_grad(params, X, y, obs_ids, alternatives)
        return ll

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

        # Try multiple optimization methods
        methods = ['L-BFGS-B', 'BFGS', 'Newton-CG']
        best_result = None
        best_success = False
        
        for method in methods:
            for init_strategy in ['zeros', 'random', 'smart']:
                try:
                    # Initialize parameters
                    if init_strategy == 'zeros':
                        init_params = np.zeros(n_params)
                    elif init_strategy == 'random':
                        np.random.seed(self.random_state)
                        init_params = np.random.normal(0, 0.1, n_params)
                    else:  # smart
                        init_params = self._smart_initialization(Xs, y)
                    
                    # Define objective with current data
                    def objective(params):
                        return self._neg_loglik_and_grad(params, Xs, y, obs_ids, alt_norm)
                    
                    # Set up bounds if using L-BFGS-B
                    bounds = None
                    if method == 'L-BFGS-B':
                        # Loose bounds to prevent extreme values
                        bounds = [(-10, 10) for _ in range(n_params)]
                    
                    # Run optimization
                    if method == 'Newton-CG':
                        result = minimize(
                            fun=lambda x: objective(x)[0],
                            x0=init_params,
                            method=method,
                            jac=lambda x: objective(x)[1],
                            options={'maxiter': self.max_iter, 'disp': False}
                        )
                    else:
                        result = minimize(
                            fun=objective,
                            x0=init_params,
                            method=method,
                            jac=True,
                            bounds=bounds,
                            options={'maxiter': self.max_iter, 'disp': False}
                        )
                    
                    # Check if this is the best result so far
                    if result.success and (best_result is None or result.fun < best_result.fun):
                        best_result = result
                        best_success = True
                    
                    # If we got a successful result, we can break early
                    if result.success:
                        break
                        
                except Exception as e:
                    continue
            
            # If we found a successful result, we can break
            if best_success:
                break
        
        if best_success and best_result is not None:
            self.params = best_result.x
            self.opt_result_ = best_result
            self.fitted_ = True
            return self
        else:
            self.fitted_ = False
            self.params = None
            return False

    def _smart_initialization(self, X, y):
        """
        Smart initialization using simple logistic regression as a starting point.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Use a simple logistic regression to get reasonable initial values
            lr = LogisticRegression(fit_intercept=self.include_constants, random_state=self.random_state, max_iter=1000)
            lr.fit(X, y)
            
            n_params = (self.n_alts_ - 1) + self.n_features_ if self.include_constants else self.n_features_
            theta0 = np.zeros(n_params)
            
            if self.include_constants:
                # Initialize ASCs with small random values
                theta0[:self.n_alts_-1] = np.random.default_rng(self.random_state).normal(0.0, 0.1, self.n_alts_-1)
                # Initialize feature parameters with logistic regression coefficients
                theta0[self.n_alts_-1:] = lr.coef_[0] * 0.1  # Scale down for stability
            else:
                theta0 = lr.coef_[0] * 0.1
                
            return theta0
        except:
            # Fallback to random initialization
            n_params = (self.n_alts_ - 1) + self.n_features_ if self.include_constants else self.n_features_
            return np.random.default_rng(self.random_state).normal(0.0, 0.01, n_params)

        # Smart initialization strategy
        n_params = (self.n_alts_ - 1) + self.n_features_ if self.include_constants else self.n_features_
        
        print(f"DEBUG: n_params = {n_params}, n_alts = {self.n_alts_}, n_features = {self.n_features_}")
        
        # Multiple initialization strategies for robustness
        initialization_strategies = [
            # Strategy 1: Small random values around zero
            lambda: np.random.default_rng(self.random_state).normal(0.0, 0.01, n_params),
            # Strategy 2: Initialize with simple logistic regression coefficients
            lambda: self._smart_initialization(Xs, y),
            # Strategy 3: Larger random initialization 
            lambda: np.random.default_rng(self.random_state).normal(0.0, 0.5, n_params),
        ]
        
        # Try multiple optimization methods for robustness
        optimization_methods = [
            {'method': 'BFGS', 'jac': True},
            {'method': 'L-BFGS-B', 'jac': True},
            {'method': 'Newton-CG', 'jac': True},
            {'method': 'BFGS', 'jac': False},  # Fallback without gradients
        ]
        
        best_result = None
        best_ll = -np.inf
        
        print(f"DEBUG: Starting optimization with {len(self.groups_)} choice sets")
        
        for i, init_strategy in enumerate(initialization_strategies):
            theta0 = init_strategy()
            print(f"DEBUG: Trying initialization strategy {i+1}, theta0 shape: {theta0.shape}")
            
            for j, opt_config in enumerate(optimization_methods):
                try:
                    print(f"DEBUG: Trying optimization method {j+1}: {opt_config['method']}")
                    # Set up optimization options
                    options = {
                        'maxiter': self.maxiter,
                        'disp': False,
                        'gtol': 1e-6,
                        'ftol': 1e-9
                    }
                    
                    if opt_config['jac']:
                        result = minimize(
                            fun=self._neg_loglik_and_grad,
                            x0=theta0,
                            args=(Xs, y, obs_ids, alt_norm),
                            method=opt_config['method'],
                            jac=True,
                            options=options
                        )
                    else:
                        result = minimize(
                            fun=self._neg_loglik,
                            x0=theta0,
                            args=(Xs, y, obs_ids, alt_norm),
                            method=opt_config['method'],
                            options=options
                        )
                    
                    print(f"DEBUG: Optimization completed, success: {result.success}, fun: {result.fun}")
                    
                    # Check if this is the best result so far
                    current_ll = -result.fun
                    if result.success and current_ll > best_ll:
                        best_result = result
                        best_ll = current_ll
                        print(f"DEBUG: New best result found, LL: {best_ll}")
                        
                    # If we got a good convergence, we can stop early
                    if result.success and result.fun < 1e-6:
                        print("DEBUG: Good convergence achieved, stopping early")
                        break
                        
                except Exception as e:
                    print(f"DEBUG: Exception in optimization: {e}")
                    # If this method fails, try the next one
                    continue
            
            # If we found a good solution, no need to try other initializations
            if best_result is not None and best_result.success:
                print("DEBUG: Found good solution, stopping initialization search")
                break
        
        # Use the best result found
        if best_result is None:
            print("DEBUG: No good result found, using fallback")
            # Last resort: simple BFGS with basic initialization
            theta0 = np.random.default_rng(self.random_state).normal(0.0, 0.01, n_params)
            best_result = minimize(
                fun=self._neg_loglik,
                x0=theta0,
                args=(Xs, y, obs_ids, alt_norm),
                method="BFGS",
                options={"maxiter": self.maxiter, "disp": False}
            )
            print(f"DEBUG: Fallback result - success: {best_result.success}, fun: {best_result.fun}")

        # Store results from the best optimization
        print(f"DEBUG: Storing results from best optimization")
        self.params = best_result.x
        self.convergence_info = best_result
        self.log_likelihood = -best_result.fun

        # Enhanced convergence diagnostics
        convergence_status = self._assess_convergence(best_result)
        print(f"DEBUG: Convergence status: {convergence_status}")

        # AIC/BIC per number of *choice sets* (not rows)
        n_sets = len(self.groups_)
        k = n_params
        self.aic = 2 * k - 2 * self.log_likelihood
        self.bic = k * np.log(n_sets) - 2 * self.log_likelihood

        print("MNL fitted",
              f"| LL = {self.log_likelihood:.3f}",
              f"| AIC = {self.aic:.2f}",
              f"| BIC = {self.bic:.2f}",
              f"| Converged = {convergence_status}")

        print(f"DEBUG: About to return self")
        return self
    
    def _assess_convergence(self, result):
        """
        Enhanced convergence assessment beyond just result.success
        """
        if not result.success:
            return False
            
        # Additional convergence checks
        checks = []
        
        # Check 1: Gradient norm should be small
        if hasattr(result, 'jac') and result.jac is not None:
            grad_norm = np.linalg.norm(result.jac)
            checks.append(grad_norm < 1e-4)
        else:
            checks.append(True)  # Can't check without gradient
            
        # Check 2: Function tolerance
        if hasattr(result, 'fun'):
            checks.append(not np.isnan(result.fun) and not np.isinf(result.fun))
        
        # Check 3: Parameter values should be reasonable (not too extreme)
        param_check = np.all(np.abs(self.params) < 50)  # Reasonable parameter bounds
        checks.append(param_check)
        
        # Check 4: Log-likelihood should be finite and negative
        ll_check = np.isfinite(self.log_likelihood) and self.log_likelihood < 0
        checks.append(ll_check)
        
        return all(checks)

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
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


# --- UTILS ---
def make_aligned_from_ids(df_long, person_ids):
    """
    Returns arrays aligned 1:1 for the specified set of people.
    """
    feature_cols = ['price', 'time', 'comfort', 'change']
    mask = df_long['id'].isin(person_ids)

    X   = df_long.loc[mask, feature_cols].to_numpy()
    y   = df_long.loc[mask, 'chosen'].to_numpy(dtype=int)
    obs = df_long.loc[mask, 'obs_id'].to_numpy()
    alts= df_long.loc[mask, 'alternative'].to_numpy()

    # sanity checks
    n = len(X)
    assert n == len(y) == len(obs) == len(alts), \
        f"Aligned arrays mismatch: X={n}, y={len(y)}, obs={len(obs)}, alts={len(alts)}"
    return X, y, obs, alts, mask


class ASUDNNGeneral:
    """
    Alternative-Specific Utility DNN for general J alternatives per choice set.
    - Trains on (n_sets, J, p_feat) and applies softmax over J utilities.
    """

    def __init__(self,
                 hidden_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.30,
                 learning_rate: float = 8e-4,
                 shared_layers: bool = True,
                 use_advanced_features: bool = True,
                 use_relative_features: bool = True,
                 enforce_constant_J: bool = True):
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.shared_layers = shared_layers
        self.use_advanced_features = use_advanced_features
        self.use_relative_features = use_relative_features
        self.enforce_constant_J = enforce_constant_J

        self.model: Optional[Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_dim_: Optional[int] = None
        self.J_: Optional[int] = None
        self.history = None

    # ------------- grouping ----------------

    @staticmethod
    def _group_sets(X: np.ndarray, y: np.ndarray, obs_ids: np.ndarray, alts: np.ndarray,
                    require_one_chosen: bool = True
                   ) -> Tuple[List[np.ndarray], List[int], List[List[int]]]:
        """
        Group rows into sets by obs_id. Return:
          sets_X   : list of (m_i, p) arrays (rows in set order sorted by 'alt')
          sets_y   : list of chosen alternative indices (0..m_i-1); if require_one_chosen=False,
                     this returns dummy zeros (unused downstream in predict).
          sets_idx : list of lists with row indices into the provided arrays, aligned with sets_X order

        If require_one_chosen=True: keep only sets with len>=2 and exactly one chosen row.
        If require_one_chosen=False: keep sets with len>=2 regardless of y.
        """
        X = np.asarray(X); y = np.asarray(y, int)
        obs_ids = np.asarray(obs_ids); alts = np.asarray(alts, int)

        df = pd.DataFrame({'obs': obs_ids, 'alt': alts, 'y': y})
        feat_cols = [f'x{j}' for j in range(X.shape[1])]
        for j, c in enumerate(feat_cols): df[c] = X[:, j]

        sets_X, sets_y, sets_idx = [], [], []
        for _, g in df.groupby('obs', sort=False):
            if len(g) < 2:
                continue
            if require_one_chosen:
                if g['y'].sum() != 1:
                    continue
            g = g.sort_values('alt')
            Xi = g[feat_cols].to_numpy(float)
            idxi = g.index.to_list()
            if require_one_chosen:
                chosen_pos = int(np.flatnonzero(g['y'].to_numpy() == 1)[0])
            else:
                chosen_pos = 0  # dummy; unused for prediction
            sets_X.append(Xi); sets_y.append(chosen_pos); sets_idx.append(idxi)
        return sets_X, sets_y, sets_idx

    def _stack_constant_J(self, sets_X, sets_y, sets_idx):
        J_sizes = np.array([Xi.shape[0] for Xi in sets_X])
        unique, counts = np.unique(J_sizes, return_counts=True)
        J_mode = int(unique[np.argmax(counts)])
        if self.enforce_constant_J and len(unique) > 1:
            kept = [i for i, Ji in enumerate(J_sizes) if Ji == J_mode]
            dropped = len(J_sizes) - len(kept)
            if dropped > 0:
                print(f"[ASUDNNGeneralJ] Mixed J {list(unique)}; keeping J={J_mode} sets ({len(kept)}), dropping {dropped}.")
            sets_X = [sets_X[i] for i in kept]
            sets_y = [sets_y[i] for i in kept]
            sets_idx = [sets_idx[i] for i in kept]
        elif not self.enforce_constant_J and len(unique) > 1:
            raise ValueError("Variable J not supported without padding; set enforce_constant_J=True.")
        X_sets = np.stack(sets_X, axis=0)   # (n, J, p)
        y_sets = np.asarray(sets_y, int)
        return X_sets, y_sets, sets_idx

    # ------------- featurization ----------------

    def _advanced_per_alt(self, X: np.ndarray) -> np.ndarray:
        # X: (n, J, 4) with [price,time,comfort,change]
        price, time = X[...,0], X[...,1]
        comfort, change = X[...,2], X[...,3]
        logp = np.log(price + 1e-8)
        logt = np.log(time  + 1e-8)
        pxt  = price * time
        cxc  = comfort * change
        return np.concatenate([X,
                               logp[...,None], logt[...,None],
                               pxt[...,None],  cxc[...,None]], axis=-1)  # 8 per alt

    def _relative_to_reference(self, Z: np.ndarray, X_orig: np.ndarray) -> np.ndarray:
        # Z: (n, J, pZ), X_orig: (n, J, 4) for picking ref by min price
        ref_idx = np.argmin(X_orig[:, :, 0], axis=1)  # (n,)
        Z_ref = Z[np.arange(Z.shape[0]), ref_idx]     # (n, pZ)
        Z_ref = np.repeat(Z_ref[:, None, :], Z.shape[1], axis=1)
        return np.concatenate([Z, Z - Z_ref], axis=-1)

    def _featurize(self, X_sets: np.ndarray) -> np.ndarray:
        Z = X_sets
        if self.use_advanced_features:
            Z = self._advanced_per_alt(Z)       # (n, J, p0=8)
        if self.use_relative_features:
            Z = self._relative_to_reference(Z, X_sets)  # (n, J, 2*p0)
        return Z

    # ------------- model building ----------------

    def _build_tower(self, input_dim: int, prefix: str) -> Model:
        inp = layers.Input(shape=(input_dim,), name=f'{prefix}_in')
        x = layers.BatchNormalization(name=f'{prefix}_bn0')(inp)
        for i, units in enumerate(self.hidden_units, 1):
            x = layers.Dense(units, activation='relu', name=f'{prefix}_dense{i}')(x)
            x = layers.BatchNormalization(name=f'{prefix}_bn{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'{prefix}_drop{i}')(x)
        out = layers.Dense(1, activation='linear', name=f'{prefix}_utility')(x)
        return Model(inp, out, name=f'{prefix}_tower')

    def _build_model(self, J: int, input_dim: int) -> Model:
        inp = layers.Input(shape=(J, input_dim), name='alts_features')  # (None, J, p)

        if self.shared_layers:
            tower = self._build_tower(input_dim, prefix='shared')
            utilities = layers.TimeDistributed(tower, name='td_shared')(inp)  # (None, J, 1)
            utilities = layers.Reshape((J,), name='utilities')(utilities)      # -> (None, J)
        else:
            utils_list = []
            for j in range(J):
                slice_j = layers.Lambda(lambda t, jj=j: t[:, jj, :],
                                        output_shape=(input_dim,),
                                        name=f'slice_alt{j+1}')(inp)          # (None, p)
                tower_j = self._build_tower(input_dim, prefix=f'alt{j+1}')
                uj = tower_j(slice_j)                                          # (None, 1)
                utils_list.append(uj)
            utilities = layers.Concatenate(axis=1, name='utilities')(utils_list)  # (None, J)

        probs = layers.Activation('softmax', name='choice_probabilities')(utilities)
        model = Model(inp, probs, name='asu_dnn_J')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ------------- training & inference ----------------

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            obs_ids: np.ndarray,
            alternatives: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
            epochs: int = 120,
            batch_size: int = 64,
            verbose: int = 0):
        # group sets requiring exactly one chosen alt (train-time)
        sets_X, sets_y, _ = self._group_sets(X, y, obs_ids, alternatives, require_one_chosen=True)
        if not sets_X:
            raise ValueError("No valid choice sets (>=2 alts and exactly one chosen).")
        X_sets, y_sets, _ = self._stack_constant_J(sets_X, sets_y, _)
        self.J_ = int(X_sets.shape[1])

        Z = self._featurize(X_sets)                 # (n, J, p_feat)
        self.input_dim_ = int(Z.shape[2])

        self.scaler = StandardScaler().fit(Z.reshape(-1, self.input_dim_))
        Zs = self.scaler.transform(Z.reshape(-1, self.input_dim_)).reshape(Z.shape)

        self.model = self._build_model(self.J_, self.input_dim_)
        if verbose:
            print(f"[ASUDNNGeneralJ] J={self.J_} | feats/alt={self.input_dim_} | sets={len(y_sets)} | params={self.model.count_params():,}")

        val_tuple = None
        if validation_data is not None:
            Xv, yv, obsv, altv = validation_data
            v_sets_X, v_sets_y, _ = self._group_sets(Xv, yv, obsv, altv, require_one_chosen=True)
            if not v_sets_X:
                raise ValueError("Validation data has no valid choice sets.")
            v_X_sets, v_y_sets, _ = self._stack_constant_J(v_sets_X, v_sets_y, _)
            if v_X_sets.shape[1] != self.J_:
                raise ValueError(f"Validation J={v_X_sets.shape[1]} differs from train J={self.J_}.")
            V = self._featurize(v_X_sets)
            Vs = self.scaler.transform(V.reshape(-1, self.input_dim_)).reshape(V.shape)
            val_tuple = (Vs, v_y_sets)

        es = callbacks.EarlyStopping(monitor='val_loss' if val_tuple else 'loss',
                                     patience=25, restore_best_weights=True, verbose=0)
        rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss' if val_tuple else 'loss',
                                            factor=0.8, patience=12, min_lr=1e-7, verbose=0)

        self.history = self.model.fit(Zs, y_sets,
                                      validation_data=val_tuple,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      callbacks=[es, rlrop],
                                      verbose=verbose)
        return self

    def predict_proba(self,
                      X: np.ndarray,
                      obs_ids: np.ndarray,
                      alternatives: np.ndarray) -> np.ndarray:
        """
        Return row-aligned probabilities with the same length/order as X.
        """
        if self.model is None or self.J_ is None or self.input_dim_ is None:
            raise ValueError("Model must be fitted before prediction.")

        # group sets WITHOUT requiring a chosen row (predict-time)
        sets_X, _, sets_idx = self._group_sets(X, np.zeros(len(X), int), obs_ids, alternatives,
                                               require_one_chosen=False)
        if not sets_X:
            return np.full(len(X), 0.5, float)

        X_sets, _, row_map = self._stack_constant_J(sets_X, [0]*len(sets_X), sets_idx)
        if X_sets.shape[1] != self.J_:
            raise ValueError(f"Predict encountered J={X_sets.shape[1]} but trained with J={self.J_}.")

        Z = self._featurize(X_sets)
        Zs = self.scaler.transform(Z.reshape(-1, self.input_dim_)).reshape(Z.shape)
        P = self.model.predict(Zs, verbose=0)   # (n, J)

        probs = np.full(len(X), 0.5, float)
        for s, row_ids in enumerate(row_map):
            ps = P[s]
            for j, row_id in enumerate(row_ids):
                probs[row_id] = float(ps[j])
        return probs

    # ------------- visualization ----------------

    def plot_training_history(self):
        """Plot training history (loss & accuracy)."""
        if self.history is None:
            raise ValueError("Model must be trained first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(self.history.history.get('loss', []), label='Training Loss', color='#FF6B6B')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='#4ECDC4')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(self.history.history.get('accuracy', []), label='Training Accuracy', color='#FF6B6B')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='#4ECDC4')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

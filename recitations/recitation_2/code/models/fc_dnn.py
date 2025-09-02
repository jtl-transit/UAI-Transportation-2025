import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

from sklearn.preprocessing import StandardScaler


# ---------- 4) FC-DNN utilities ----------

def build_flattened_sets(X, y, obs_ids, alts):
    """
    Build per-choice-set inputs for binary (2-alt) problems.
    Returns:
      X_flat : (n_sets, 8)  -> [x_alt1 (4), x_alt2 (4)]
      y_set  : (n_sets,)    -> 1 if alt2 chosen else 0
      row_pairs : list[(i1,i2)] row indices into the provided arrays (X,y,...)
    """
    X = np.asarray(X); y = np.asarray(y, int)
    obs_ids = np.asarray(obs_ids); alts = np.asarray(alts, int)

    df = pd.DataFrame({'obs': obs_ids, 'alt': alts, 'y': y})
    for j in range(X.shape[1]):
        df[f'x{j}'] = X[:, j]

    X_flat, y_set, row_pairs = [], [], []
    cols = [f'x{j}' for j in range(X.shape[1])]
    for _, g in df.groupby('obs', sort=False):
        if len(g) != 2:
            continue
        g = g.sort_values('alt')
        if not np.array_equal(g['alt'].to_numpy(), [1, 2]):
            continue
        if g['y'].sum() != 1:
            continue

        i1, i2 = g.index.to_numpy()
        x1 = g[cols].iloc[0].to_numpy(float)
        x2 = g[cols].iloc[1].to_numpy(float)

        X_flat.append(np.concatenate([x1, x2], axis=0))        # 8-dim
        y_set.append(int(g['y'].iloc[1] == 1))                 # 1 if alt2 chosen
        row_pairs.append((i1, i2))
    if not X_flat:
        return np.empty((0, 8)), np.empty((0,), int), []
    return np.vstack(X_flat), np.asarray(y_set, int), row_pairs

def row_probs_to_pairs(obs_ids, alts, y, p_row):
    """Convert per-row probs -> (y_pair, p_alt2) per set for metrics."""
    df = pd.DataFrame({'obs': obs_ids, 'alt': alts, 'y': y, 'p': p_row})
    y_pair, p_alt2 = [], []
    for _, g in df.groupby('obs', sort=False):
        if len(g) != 2:
            continue
        g = g.sort_values('alt')
        if not np.array_equal(g['alt'].to_numpy(), [1, 2]):
            continue
        yy = g['y'].to_numpy()
        if yy.sum() != 1:
            continue
        y_pair.append(int(yy[1] == 1))
        p_alt2.append(float(g['p'].to_numpy()[1]))
    return np.asarray(y_pair, int), np.asarray(p_alt2, float)


# ---------- 4) FC-DNN implementation ----------
class FullyConnectedDNN:
    """
    Fully Connected Deep Neural Network for discrete choice modeling.
    """

    def __init__(self, hidden_units=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001):
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.history = None

    def _build_model(self, input_dim):
        """Build the FC-DNN architecture"""

        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='flattened_features')

        # Hidden layers with dropout
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, activation='relu', name=f'hidden_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='choice_probability')(x)

        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='FC_DNN')

        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def fit(self, X, y, obs_ids, alternatives, validation_data=None, epochs=100, batch_size=32, verbose=0):
        """
        Train the FC-DNN model.

        Parameters:
        -----------
        X : array-like
            Feature matrix (individual alternatives)
        y : array-like
            Choice indicators
        obs_ids : array-like
            Observation identifiers
        alternatives : array-like
            Alternative identifiers
        validation_data : tuple, optional
            Validation data in same format (X_val, y_val, obs_ids_val, alts_val)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
        """
        print("Training Fully Connected DNN...")

        # Convert to FC format (choice sets with concatenated alternatives)
        X_fc, y_fc = self._convert_to_fc_format(X, y, obs_ids, alternatives)

        # Feature scaling
        self.scaler = StandardScaler()
        X_fc_scaled = self.scaler.fit_transform(X_fc)

        print(f"- Training data shape: {X_fc_scaled.shape}")
        print(f"- Features per choice set: {X_fc_scaled.shape[1]} (4 features × 2 alternatives)")
        print(f"- Number of choice sets: {len(X_fc_scaled)}")
        print(f"- Choice distribution: {np.bincount(y_fc.astype(int))}")

        # Build model
        self.model = self._build_model(X_fc_scaled.shape[1])

        print(f"- Model architecture: {[X_fc_scaled.shape[1]] + self.hidden_units + [1]}")
        print(f"- Total parameters: {self.model.count_params():,}")

        # Prepare validation data if provided
        if validation_data is not None:
            X_val, y_val, obs_ids_val, alts_val = validation_data
            X_val_fc, y_val_fc = self._convert_to_fc_format(X_val, y_val, obs_ids_val, alts_val)
            X_val_fc_scaled = self.scaler.transform(X_val_fc)
            validation_data = (X_val_fc_scaled, y_val_fc)
            print(f"- Validation data shape: {X_val_fc_scaled.shape}")

        # Set up callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )

        # Train model
        print("- Starting training...")
        self.history = self.model.fit(
            X_fc_scaled, y_fc,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )

        # Training summary
        final_epoch = len(self.history.history['loss'])
        final_loss = self.history.history['loss'][-1]
        final_accuracy = self.history.history['accuracy'][-1]

        print(f"FC-DNN training complete!")
        print(f"- Final epoch: {final_epoch}/{epochs}")
        print(f"- Final training loss: {final_loss:.4f}")
        print(f"- Final training accuracy: {final_accuracy:.4f}")

        if validation_data is not None:
            final_val_loss = self.history.history['val_loss'][-1]
            final_val_accuracy = self.history.history['val_accuracy'][-1]
            print(f"- Final validation loss: {final_val_loss:.4f}")
            print(f"- Final validation accuracy: {final_val_accuracy:.4f}")

        return self
    
    def _convert_to_fc_format(self, X, y, obs_ids, alternatives):
        """Convert individual alternative format to choice set format for FC-DNN"""
        
        # Create dataframe for easier manipulation
        df = pd.DataFrame({
            'obs_id': obs_ids,
            'alt': alternatives,
            'choice': y
        })
        
        # Add feature columns
        for i in range(X.shape[1]):
            df[f'feature_{i}'] = X[:, i]
        
        # Group by observation and create choice sets
        choice_sets = []
        choice_labels = []
        
        for obs_id, group in df.groupby('obs_id'):
            if len(group) == 2:  # Valid choice set
                # Sort by alternative to ensure consistent order
                group = group.sort_values('alt')
                
                # Get features for both alternatives
                feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                alt1_features = group.iloc[0][feature_cols].values
                alt2_features = group.iloc[1][feature_cols].values
                
                # Concatenate features
                choice_set_features = np.concatenate([alt1_features, alt2_features])
                choice_sets.append(choice_set_features)
                
                # Determine which alternative was chosen (1 if alt2, 0 if alt1)
                chosen_alts = group[group['choice'] == 1]['alt'].values
                if len(chosen_alts) == 1:
                    choice_label = 1 if chosen_alts[0] == group.iloc[1]['alt'] else 0
                else:
                    # If no choice or multiple choices, skip this choice set
                    choice_sets.pop()  # Remove the features we just added
                    continue
                choice_labels.append(choice_label)
        
        return np.array(choice_sets), np.array(choice_labels)

    def predict_proba(self, X, obs_ids, alternatives):
        """
        Predict choice probabilities.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        obs_ids : array-like
            Observation identifiers
        alternatives : array-like
            Alternative identifiers
            
        Returns:
        --------
        array : Choice probabilities in original format (one per alternative)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Create a dataframe to help with mapping
        df_temp = pd.DataFrame({
            'obs_id': obs_ids,
            'alternative': alternatives
        }, index=range(len(X)))
        
        # Create mask for valid 2-alternative choice sets
        mask = df_temp.groupby('obs_id')['alternative'].transform('count') == 2

        # Filter to only valid choice sets
        valid_indices = np.where(mask)[0]
        X_valid = X[valid_indices]
        df_valid = df_temp.iloc[valid_indices].copy()

        # Prepare data in FC format (pairwise comparisons)
        choice_sets = []
        mapping = []  # Track which original indices correspond to each choice set

        for obs_id, group in df_valid.groupby('obs_id'):
            if len(group) == 2:
                # Sort by alternative to ensure consistent order
                group = group.sort_values('alternative')
                
                # Get feature vectors for both alternatives
                alt1_idx = group.iloc[0].name
                alt2_idx = group.iloc[1].name
                
                features_alt1 = X_valid[alt1_idx - valid_indices[0]]  # Adjust index
                features_alt2 = X_valid[alt2_idx - valid_indices[0]]  # Adjust index
                
                # Create choice set feature vector (concatenate)
                choice_set_features = np.concatenate([features_alt1, features_alt2])
                choice_sets.append(choice_set_features)
                
                # Store mapping
                mapping.append((alt1_idx, alt2_idx))

        if not choice_sets:
            # No valid choice sets found
            return np.zeros(len(X))

        X_fc = np.array(choice_sets)
        X_fc_scaled = self.scaler.transform(X_fc)

        # Get predictions (probability of choosing alternative 2)
        prob_alt2 = self.model.predict(X_fc_scaled, verbose=0).flatten()
        prob_alt1 = 1 - prob_alt2

        # Convert back to original format (one probability per alternative)
        probabilities = np.zeros(len(X))

        # Assign probabilities using the mapping
        for i, (alt1_idx, alt2_idx) in enumerate(mapping):
            probabilities[alt1_idx] = prob_alt1[i]
            probabilities[alt2_idx] = prob_alt2[i]

        return probabilities

    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise ValueError("Model must be fitted first")

        print("\nFC-DNN Architecture Summary:")
        self.model.summary()

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("Model must be trained first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss', color='#FF6B6B')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='#4ECDC4')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy', color='#FF6B6B')
        if 'val_accuracy' in self.history.history:
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='#4ECDC4')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_choice_probabilities(self, variable_name, variable_range, baseline_data, 
                                 alternative_names=None, figsize=(10, 6)):
        """
        Plot choice probabilities as a function of a varying attribute.
        
        Parameters:
        -----------
        variable_name : str
            Name of the variable to vary ('price', 'time', 'change', or 'comfort')
        variable_range : array-like
            Range of values for the variable to plot
        baseline_data : dict
            Dictionary with baseline values for all variables. 
            Keys should be: 'price', 'time', 'change', 'comfort'
            Example: {'price': 2400, 'time': 130, 'change': 1, 'comfort': 1}
        alternative_names : list, optional
            Names for the alternatives. If None, uses 'Alt 1', 'Alt 2'.
        figsize : tuple
            Figure size (width, height)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before plotting.")
        
        import matplotlib.pyplot as plt
        
        # FC-DNN is a binary choice model, so we have 2 alternatives
        if alternative_names is None:
            alternative_names = ['Alt 1', 'Alt 2']
        elif len(alternative_names) != 2:
            raise ValueError(f"FullyConnectedDNN is a binary model, expected 2 alternative names, got {len(alternative_names)}")
        
        # Define feature order as expected by the model
        feature_order = ['price', 'time', 'change', 'comfort']
        
        # Validate inputs
        if variable_name not in feature_order:
            raise ValueError(f"variable_name must be one of {feature_order}")
        
        for feature in feature_order:
            if feature not in baseline_data:
                raise ValueError(f"baseline_data must contain '{feature}'")
        
        # Create synthetic data for prediction
        n_points = len(variable_range)
        
        # For FC-DNN, we need to create choice sets with 2 alternatives each
        X_plot = []
        
        for var_value in variable_range:
            # Create feature vectors for both alternatives
            alt1_features = []  # Baseline alternative
            alt2_features = []  # Variable alternative
            
            for feature_name in feature_order:
                if feature_name == variable_name:
                    alt1_features.append(baseline_data[feature_name])  # Alt 1: baseline value
                    alt2_features.append(var_value)                   # Alt 2: varying value
                else:
                    alt1_features.append(baseline_data[feature_name]) # Both: baseline value
                    alt2_features.append(baseline_data[feature_name])
            
            # For FC-DNN, we create the flattened feature vector (alt1 + alt2 features)
            full_features = alt1_features + alt2_features  # 8 features total (4 per alternative)
            X_plot.append(full_features)
        
        X_plot = np.array(X_plot)
        
        # Get probabilities for alternative 2 (alt 1 probability = 1 - alt 2 probability)
        probs_alt2 = self.model.predict(X_plot, verbose=0).flatten()
        probs_alt1 = 1 - probs_alt2
        
        # Create probability matrix
        prob_matrix = np.column_stack([probs_alt1, probs_alt2])
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        colors = ['#FF6B6B', '#4ECDC4']
        
        for j in range(2):
            if j == 0:
                label = f"{alternative_names[j]} (baseline)"
            else:
                label = f"{alternative_names[j]} (varying {variable_name})"
            plt.plot(variable_range, prob_matrix[:, j], 
                    label=label, 
                    linewidth=2.5,
                    color=colors[j])
        
        plt.xlabel(f"{variable_name.capitalize()}")
        plt.ylabel("Choice Probability")
        plt.title(f"Choice Probabilities vs {variable_name.capitalize()} (FC-DNN Model)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add baseline information to the plot
        baseline_str = ", ".join([f"{k}={v}" for k, v in baseline_data.items()])
        plt.figtext(0.02, 0.02, f"Baseline: {baseline_str}", fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        return prob_matrix
        
        plt.tight_layout()
        plt.show()
        
        return prob_matrix
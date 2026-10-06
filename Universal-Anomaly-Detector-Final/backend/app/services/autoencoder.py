from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_autoencoder(input_dim: int) -> keras.Model:
    if input_dim <= 8:
        h1, h2, latent = max(8, input_dim * 2), max(4, input_dim), max(2, input_dim // 2)
    elif input_dim <= 64:
        h1, h2, latent = 64, 32, max(8, min(16, input_dim // 4))
    else:
        h1, h2, latent = 128, 64, max(16, min(32, input_dim // 4))

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(h1, activation="relu"),
            keras.layers.Dropout(0.05),
            keras.layers.Dense(h2, activation="relu"),
            keras.layers.Dense(latent, activation="relu", name="latent"),
            keras.layers.Dense(h2, activation="relu"),
            keras.layers.Dense(h1, activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ],
        name="universal_autoencoder",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def detect_anomalies(
    matrix: np.ndarray,
    training_matrix: np.ndarray | None = None,
    epochs: int = 15,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Train on a manageable normal-data sample and score the full dataset."""
    tf.random.set_seed(42)
    np.random.seed(42)

    if training_matrix is None:
        training_matrix = matrix

    model = build_autoencoder(training_matrix.shape[1])
    validation_split = 0.2 if len(training_matrix) >= 30 else 0.0

    history = model.fit(
        training_matrix,
        training_matrix,
        epochs=epochs,
        batch_size=min(batch_size, max(1, len(training_matrix))),
        validation_split=validation_split,
        shuffle=True,
        verbose=0,
    )

    # Score the complete uploaded dataset in batches to keep memory usage low.
    reconstructed = model.predict(matrix, batch_size=1024, verbose=0)
    errors = np.mean(np.square(matrix - reconstructed), axis=1)

    # The threshold is learned from training reconstruction errors, not from
    # the complete dataset. This avoids automatically labelling exactly 5%.
    train_reconstructed = model.predict(training_matrix, batch_size=1024, verbose=0)
    train_errors = np.mean(np.square(training_matrix - train_reconstructed), axis=1)
    threshold = float(np.percentile(train_errors, 99))

    labels = errors > threshold
    losses = [float(x) for x in history.history["loss"]]
    return labels, errors, threshold, losses

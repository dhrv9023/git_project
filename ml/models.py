"""
ml/models.py — Deep learning model builders for LSTM, GRU, and Transformer.

Engineering decisions:
  - Pure builder functions: take input_shape + config, return compiled model.
  - No global CONFIG references — all hyperparams passed as arguments with defaults.
  - Extracted from app.py to break circular sys.modules coupling in trainer.py.
  - Type hints on all signatures.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers

log = logging.getLogger(__name__)

ModelType = Literal["LSTM", "GRU", "Transformer"]


def build_lstm(
    input_shape: tuple[int, int],
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Stacked LSTM with dropout and L2 regularisation.

    Architecture: Input → LSTM(128) → Dropout → LSTM(64) → Dropout
                  → Dense(64, relu) → Dropout → Dense(1, float32)

    Engineering decision: Huber loss (delta=1.0) is more robust to outlier
    log-return spikes than MSE, while still providing gradient signal for
    in-range predictions.
    """
    reg = tf.keras.regularizers.l2(5e-5)
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(128, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=reg),
            layers.Dropout(0.4),
            layers.LSTM(64, recurrent_dropout=0.3, kernel_regularizer=reg),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu", kernel_regularizer=reg),
            layers.Dropout(0.4),
            layers.Dense(1, dtype="float32"),
        ],
        name="LSTM",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def build_gru(
    input_shape: tuple[int, int],
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Stacked GRU — same topology as LSTM but faster on CPU."""
    reg = tf.keras.regularizers.l2(5e-5)
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.GRU(128, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=reg),
            layers.Dropout(0.4),
            layers.GRU(64, recurrent_dropout=0.3, kernel_regularizer=reg),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu", kernel_regularizer=reg),
            layers.Dropout(0.4),
            layers.Dense(1, dtype="float32"),
        ],
        name="GRU",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def build_transformer(
    input_shape: tuple[int, int],
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Two-block Transformer encoder with learnable positional embeddings.

    Engineering decision: positional embedding via tf.keras.layers.Embedding
    rather than sinusoidal, because the sequence lengths here (≤90) are short
    enough that learned embeddings converge reliably.
    """
    inp = layers.Input(shape=input_shape)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embed = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    x = inp + pos_embed

    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(input_shape[1])(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout + 0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, dtype="float32")(x)

    model = models.Model(inputs=inp, outputs=out, name="Transformer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def build_model(
    model_type: ModelType,
    input_shape: tuple[int, int],
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Factory: returns a compiled model by name.

    Engineering decision: single entry point for model construction allows
    callers (TrainingService) to be agnostic to the specific architecture,
    satisfying the Open/Closed principle — new architectures are added here
    without changing callers.
    """
    builders = {
        "LSTM": build_lstm,
        "GRU": build_gru,
        "Transformer": build_transformer,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(builders)}")
    log.debug("Building model: %s  input_shape=%s", model_type, input_shape)
    return builders[model_type](input_shape, learning_rate=learning_rate)


def make_callbacks(
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    use_early_stopping: bool = True,
) -> list:
    """Build standard training callbacks: EarlyStopping + LR warmup/cosine decay.

    Engineering decision: cosine decay with linear warmup prevents large
    gradient updates at the start of training, which is especially important
    for the Transformer architecture.
    """
    cb_list = []

    if use_early_stopping:
        cb_list.append(
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        )
        cb_list.append(
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        )

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_schedule(step: int) -> float:
        step_t = tf.cast(step, tf.float32)
        lr_base = tf.constant(learning_rate, tf.float32)
        lr_min = lr_base * 0.1
        progress = (step_t - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = lr_min + (lr_base - lr_min) * 0.5 * (
            1 + tf.cos(np.pi * tf.clip_by_value(progress, 0.0, 1.0))
        )
        return float(tf.where(step_t < warmup_steps, lr_base * (step_t / warmup_steps), cosine).numpy())

    cb_list.append(callbacks.LearningRateScheduler(lr_schedule, verbose=0))
    return cb_list

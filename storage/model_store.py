"""
storage/model_store.py — Disk-based model artifact storage.

Production pattern: all model weights + scalers are persisted to a versioned
directory tree. This makes the ML system restart-safe: a server restart does
not invalidate trained models. Matches how MLflow, BentoML, and Vertex AI
handle artifact storage, but with zero additional dependencies.

Directory layout:
    model_artifacts/
    └── {ticker}/
        └── {version}/
            ├── metadata.json     ← training config, metrics, timestamps
            ├── lstm.keras        ← Keras SavedModel format
            ├── gru.keras
            ├── transformer.keras
            ├── scaler_X.pkl      ← MinMaxScaler for features
            └── scaler_y.pkl      ← MinMaxScaler for targets
"""

import os
import json
import pickle
import datetime
import logging
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


class ModelStore:
    """
    Handles all file I/O for trained model artifacts.

    Each (ticker, version) pair maps to one directory under base_dir.
    Callers never hard-code paths — they ask ModelStore for them.
    """

    def __init__(self, base_dir: str = "model_artifacts"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def version_dir(self, ticker: str, version: str) -> str:
        """Return the directory for a specific (ticker, version) pair."""
        d = os.path.join(self.base_dir, ticker.upper(), version)
        os.makedirs(d, exist_ok=True)
        return d

    def model_path(self, ticker: str, version: str, model_name: str) -> str:
        return os.path.join(self.version_dir(ticker, version), f"{model_name.lower()}.keras")

    def scaler_path(self, ticker: str, version: str, name: str) -> str:
        return os.path.join(self.version_dir(ticker, version), f"{name}.pkl")

    def metadata_path(self, ticker: str, version: str) -> str:
        return os.path.join(self.version_dir(ticker, version), "metadata.json")

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_model(self, keras_model, ticker: str, version: str, model_name: str) -> str:
        """Persist a Keras model to disk in SavedModel format."""
        path = self.model_path(ticker, version, model_name)
        keras_model.save(path)
        log.info(f"Saved {model_name} → {path}")
        return path

    def save_scaler(self, scaler, ticker: str, version: str, name: str) -> str:
        """Persist a scikit-learn scaler via pickle."""
        path = self.scaler_path(ticker, version, name)
        with open(path, "wb") as f:
            pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved scaler {name} → {path}")
        return path

    def save_metadata(self, ticker: str, version: str, metadata: Dict[str, Any]) -> str:
        """Write training metadata (config, metrics, timestamps) as JSON."""
        path = self.metadata_path(ticker, version)
        metadata["saved_at"] = datetime.datetime.utcnow().isoformat()
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        log.info(f"Saved metadata → {path}")
        return path

    def save_artifacts(self, ticker: str, version: str,
                       models_dict: dict,
                       scaler_X, scaler_y,
                       metadata: Dict[str, Any]) -> str:
        """
        One-shot: save all models + scalers + metadata for a training run.
        Returns the version directory path.
        """
        vdir = self.version_dir(ticker, version)

        for name, mdl in models_dict.items():
            if name.endswith("_history"):
                continue
            try:
                self.save_model(mdl, ticker, version, name)
            except Exception as e:
                log.error(f"Failed to save model {name}: {e}")

        self.save_scaler(scaler_X, ticker, version, "scaler_X")
        self.save_scaler(scaler_y, ticker, version, "scaler_y")
        self.save_metadata(ticker, version, metadata)
        return vdir

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_model(self, ticker: str, version: str, model_name: str):
        """Load a Keras model from disk. Returns None if file not found."""
        import tensorflow as tf
        path = self.model_path(ticker, version, model_name)
        if not os.path.exists(path):
            log.warning(f"Model file not found: {path}")
            return None
        try:
            mdl = tf.keras.models.load_model(path)
            log.info(f"Loaded {model_name} ← {path}")
            return mdl
        except Exception as e:
            log.error(f"Failed to load {model_name}: {e}")
            return None

    def load_scaler(self, ticker: str, version: str, name: str):
        """Load a pickled scaler. Returns None if not found."""
        path = self.scaler_path(ticker, version, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_metadata(self, ticker: str, version: str) -> Optional[Dict]:
        """Load training metadata JSON. Returns None if missing."""
        path = self.metadata_path(ticker, version)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def load_artifacts(self, ticker: str, version: str,
                       model_names: list = None) -> Optional[Dict]:
        """
        Load all artifacts for a (ticker, version) pair.
        Returns dict with keys: models, scaler_X, scaler_y, metadata
        Returns None if the version directory doesn't exist.
        """
        vdir = os.path.join(self.base_dir, ticker.upper(), version)
        if not os.path.isdir(vdir):
            return None

        if model_names is None:
            model_names = ["LSTM", "GRU", "Transformer"]

        models_dict = {}
        for name in model_names:
            mdl = self.load_model(ticker, version, name)
            if mdl is not None:
                models_dict[name] = mdl

        scaler_X = self.load_scaler(ticker, version, "scaler_X")
        scaler_y = self.load_scaler(ticker, version, "scaler_y")
        metadata = self.load_metadata(ticker, version)

        if not models_dict or scaler_X is None or scaler_y is None:
            log.warning(f"Incomplete artifacts for {ticker}/{version}")
            return None

        return {
            "models":   models_dict,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "metadata": metadata,
        }

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_versions(self, ticker: str) -> list:
        """Return all version directory names for a ticker."""
        ticker_dir = os.path.join(self.base_dir, ticker.upper())
        if not os.path.isdir(ticker_dir):
            return []
        return sorted([
            d for d in os.listdir(ticker_dir)
            if os.path.isdir(os.path.join(ticker_dir, d))
        ])

    def artifact_exists(self, ticker: str, version: str) -> bool:
        """Check if a complete artifact set exists on disk."""
        vdir = os.path.join(self.base_dir, ticker.upper(), version)
        return (
            os.path.isdir(vdir)
            and os.path.exists(self.metadata_path(ticker, version))
            and os.path.exists(self.scaler_path(ticker, version, "scaler_X"))
        )

    def disk_usage_bytes(self) -> int:
        """Return total bytes used by all stored artifacts."""
        total = 0
        for root, dirs, files in os.walk(self.base_dir):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total

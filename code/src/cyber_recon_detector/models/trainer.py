"""
 * author Antonio Sirignano
 * created on 10-09-2026
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score

from cyber_recon_detector.utils.terminal import TerminalLogger


class ReconModelTrainer:
    """Model Trainer for Reconnaissance Attack Detection."""

    def __init__(self, logger: TerminalLogger, models_dir: str | Path = "models") -> None:
        """
        Args:
            logger (TerminalLogger): Logger for terminal output.
            models_dir (str | Path): Directory to save/load models.
        """

        self.logger = logger
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, suffix: str = ""):
        """
        Train the LightGBM model for reconnaissance attack detection.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.
            suffix (str): Suffix for the model filename.

        """

        scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)   # Weighting for class imbalance
        
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        model_file = self.models_dir / f"lightgbm_recon_model{suffix}.joblib"
        joblib.dump(self.model, model_file)
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model on the test set and print classification metrics.

        Args:
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.

        Returns:
            Dict[str, float]: Dictionary containing PR-AUC and ROC-AUC scores.
        """

        if self.model is None:
            raise ValueError("No model has been trained. Please train the model before evaluation.")

        self.logger.log_info("Evaluating the model on the test set...")
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        self.logger.log_info(f"PR-AUC Score: {pr_auc:.4f}")
        self.logger.log_info(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=["BENIGN", "RECON"]))

        return {"pr_auc": pr_auc, "roc_auc": roc_auc}

    def save_model(self, path: Path) -> None:
        """
        Save the trained model to disk. 

        Args:
            path (Path): Path where the model will be saved.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  
        joblib.dump(self.model, path)
        self.logger.log_info(f"Model saved successfully to {path}")


    def load_model(self, path: Path) -> None:
        """
        Load a previously trained model from disk.

        Args:
            path (Path): Path to the model file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self.logger.log_info(f"Model loaded successfully from {path}")
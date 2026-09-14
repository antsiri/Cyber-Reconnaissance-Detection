"""
 * author Antonio Sirignano
 * created on 11-09-2026-11h-26m
 * github: https://github.com/antsiri
 * copyright 2026
"""

from pathlib import Path
import polars as pl
import numpy as np

from cyber_recon_detector.utils.terminal import TerminalLogger
from cyber_recon_detector.ingestion.downloader import DatasetDownloader
from cyber_recon_detector.ingestion.dataset_loader  import DatasetLoader
from cyber_recon_detector.features.preprocessor import ReconDataPreprocessor
from cyber_recon_detector.models.trainer import ReconModelTrainer

USE_CORRELATION = True
FORCE_RETRAIN = True

def run_pipeline(force_reatrain: bool = False) -> None:
    """
        Run the complete pipeline for cyber reconnaissance detection, including dataset downloading,
        preprocessing, model training, and evaluation.

        Args:
            force_reatrain (bool): If True, forces retraining of the model even if a trained model exists.
    """
    logger = TerminalLogger()

    logger.log_info("=============================")
    logger.log_info("Early Cyber-Recon Detector")
    logger.log_info("=============================")

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    models_dir = Path("models")

    suffix = "_corr" if USE_CORRELATION else "_base"

    cic_parquet = processed_dir / "cicids2017_full.parquet"
    unswnb_parquet = processed_dir / "unswnb15_full.parquet"
    npz_cache = processed_dir / f"cicids2017_arrays{suffix}.npz"
    model_path = models_dir / f"lightgbm_recon_model{suffix}.joblib"

    logger.log_info("--- Phase 1: Verify and Download Datasets ---")
    downloader = DatasetDownloader(raw_dir=raw_dir, logger=logger)
    downloader.fetch_dataset("CIC-IDS2017")
    downloader.fetch_dataset("UNSW-NB15")

    logger.log_info("--- Phase 2: Ingesetion and Parquet Optimization ---")
    if not cic_parquet.exists() or not unswnb_parquet.exists():
        loader = DatasetLoader(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            logger=logger
        )
        loader.process_cicids20217()
        loader.process_unswnb15()
        logger.log_info("Ingestion completed with success!")
    else:
        logger.log_info("Parquet founded in cache!")

    logger.log_info("--- Phase 3: Preprocessing & Scaling Feature ---")
    if not npz_cache.exists() or force_reatrain:
        df_cic = pl.read_parquet(cic_parquet)
        preprocessor = ReconDataPreprocessor(logger=logger)
        X_train, X_test, y_train, y_test, features = preprocessor.process_cicids2017(df_cic, USE_CORRELATION)

        np.savez_compressed(
            npz_cache,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        logger.log_info(f"Preprocessed matrix saved in cache: {npz_cache}")
    else:
        logger.log_info("Loading scaled matrix from cache '.npz'...")
        data = np.load(npz_cache)
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

    logger.log_info("--- Phase 4: Training & Evalutation ---")
    trainer = ReconModelTrainer(logger=logger,
                                models_dir=models_dir)

    if model_path.exists() and not force_reatrain:
        logger.log_info("Model already trained founded! Loading...")
        trainer.load_model(model_path)
        trainer.evaluate(X_test, y_test)
    else:
        logger.log_info("Training the new model LightGBM...")
        trainer.train(X_train, y_train, X_test, y_test, suffix=suffix)


if __name__ == "__main__":
    run_pipeline(FORCE_RETRAIN)
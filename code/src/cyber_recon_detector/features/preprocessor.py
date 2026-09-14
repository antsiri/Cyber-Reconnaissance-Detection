"""
 * author Antonio Sirignano
 * created on 10-09-2026-10h-38m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import numpy as np
import polars as pl

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from cyber_recon_detector.utils.terminal import TerminalLogger

class ReconDataPreprocessor:
    """
        Class to preprocess the CIC-IDS2017 dataset for cyber reconnaissance detection.
    """

    def __init__(self, logger: TerminalLogger):
        """
            Initialize the preprocessor with a logger and a scaler.

            Args:
                logger (TerminalLogger): An instance of TerminalLogger for logging messages.           
        """

        self.logger = logger
        self.scaler = RobustScaler()                # Initialize the robust scaler

        self.skewed_features = [                    # List of features that are skewed and will be log-transformed
            "Flow Duration",
            "Total Fwd Packets",
            "Total Lenght of Fwd Packets",
            "Flow Packets/s",
            "Flow Bytes/s",
        ]

    def extract_sequential_correlation_features(self, df: pl.DataFrame, window_size: int = 50) -> pl.DataFrame:
        """
            Extract sequential correlation features from the DataFrame using a rolling window.

            Args:
                df (pl.DataFrame): The input DataFrame containing the data.
                window_size (int): The size of the rolling window for feature extraction.

            Returns:
                pl.DataFrame: A new DataFrame with the extracted sequential correlation features.
        """

        df_work = df.with_columns(pl.int_range(0, pl.len()).alias("_row_idx"))

        rolling_stats = df_work.rolling(
            index_column="_row_idx",
            period=f"{window_size}i"
        ).agg([                                                         # Aggregation of rolling statistics for feature extraction
            pl.col("Destination Port")
            .n_unique()                                                 # Count unique destination ports in the rolling window
            .alias(f"corr_unique_dst_ports_w{window_size}"),

            pl.col("Flow Duration")
            .mean()
            .alias(f"corr_avg_duration_w{window_size}"),

            pl.col("Fwd Packet Length Std")
            .mean()
            .alias(f"corr_avg_fwd_pkt_std_w{window_size}"),

            pl.col("Flow Packets/s")
            .max()
            .alias(f"corr_max_pkt_rate_w{window_size}"),
        ])

        return df_work.join(rolling_stats, on="_row_idx", how="left").drop("_row_idx")

    def process_cicids2017(self, df: pl.DataFrame, use_correlation: bool = False,
                           test_size: float = 0.2, seed: int = 42, 
                           window_size: int = 50) -> Tuple:
        """
            Preprocess the CIC-IDS2017 dataset for cyber reconnaissance detection.

            Args:
                df (pl.DataFrame): The input DataFrame containing the CIC-IDS2017 dataset
                use_correlation (bool): Whether to extract sequential correlation features
                test_size (float): The proportion of the dataset to include in the test split
                seed (int): Random seed for reproducibility
                window_size (int): The size of the rolling window for sequential correlation features

            Returns:
                Tuple: A tuple containing the preprocessed training and testing data 
        """

        self.logger.log_info("Starting Preprocessing for CIC-IDS2017...")

        df_work = df.filter(                        # Filter the DataFrame to include only benign, portscan, and recon samples
            pl.col("Label").str.to_lowercase().str.contains("benign|portscan|recon")
        ).clone()


        if use_correlation:
            self.logger.log_info(f"Sequential correlation enabled (Window Size: {window_size})")
            df_work = self.extract_sequential_correlation_features(df_work, window_size=window_size)

        numeric_cols = [col 
                        for col, dtype in zip(df_work.columns, df_work.dtypes)
                        if dtype in [pl.Float64, pl.Float32, pl.Int16, pl.Int32] and col not in  ["Label", "target"]
                        ]

        sanitize_exprs = [                          # Sanitize numeric columns by replacing null, NaN, and infinite values with 0.0
            pl.when(pl.col(col).is_null()
                    | pl.col(col).is_nan()
                    | pl.col(col).is_infinite()
                    )
                    .then(0.0)
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in numeric_cols
        ]
        df_clean = df_work.with_columns(sanitize_exprs)

        df_clean = df_clean.with_columns(
            pl.when(pl.col("Label").str.to_lowercase().str.contains("portscan|recon"))
            .then(1)
            .otherwise(0)
            .alias("target")
        )

        log_exprs = []
        for col in self.skewed_features:            # Apply log transformation to skewed features, ensuring no negative values are present
            if col in df_clean.columns:
                log_exprs.append(
                    (
                        pl.when(pl.col(col) < 0)
                        .then(0.0)
                        .otherwise(pl.col(col))
                        + 1.0
                    )
                    .log10()
                    .alias(col)
                )
        df_clean = df_clean.with_columns(log_exprs)

        feature_cols = [col for col in numeric_cols if col != "target"]
        X = df_clean.select(feature_cols).to_numpy()
        y = df_clean.select("target").to_numpy().ravel()

        self.logger.log_info(f"Stratified data splitting ({int((1 - test_size) * 100)}/{int(test_size * 100)})...")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

        self.logger.log_info("RobustScaler on the features...")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.logger.log_info(f"Preprocessing completed -> Train: {X_train.shape}, Test: {X_test.shape}")

        return X_train, X_test, y_train, y_test, feature_cols
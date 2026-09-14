"""
 * author Antonio Sirignano
 * created on 09-09-2026-10h-22m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import polars as pl

from pathlib import Path

from cyber_recon_detector.utils.terminal import TerminalLogger

class DatasetLoader:
    """
        Class to load and preprocess datasets for cyber reconnaissance detection.
    """

    def __init__(self,
                 raw_dir: str | Path,
                 processed_dir: str | Path,
                 logger: TerminalLogger
                 ) -> None:
        """
            Initialize the DatasetLoader with directories and a logger.

            Args:
                raw_dir (str | Path): The directory where raw datasets are stored.
                processed_dir (str | Path): The directory where processed datasets will be saved.
                logger (TerminalLogger): An instance of TerminalLogger for logging messages.
        """

        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.logger = logger

        processed_dir.mkdir(parents=True, exist_ok=True)


    def process_cicids20217(self) -> Path:
        """
            Process the CIC-IDS2017 dataset by reading CSV files, filtering relevant records, and
            saving the cleaned data as a Parquet file. 

            Returns:
                Path: The path to the saved Parquet file containing the processed dataset.
        """

        output_file = self.processed_dir / "cicids2017_full.parquet"
        csv_file = list((self.raw_dir / "CIC-IDS2017").glob("*.csv"))

        if not csv_file:
            self.logger.log_warning("No CSV file founded for CIC-IDS2017")
            return output_file

        self.logger.log_info(f"Start parsing of {len(csv_file)} file for CIC-IDS2017")

        lazy_dfs = []

        for file in self.logger.track_iterable(csv_file, "Parsing CSV CIC-IDS2017"):        
            df_lazy = (                                                                 # LazyFrame to read CSV in streaming mode
                pl.scan_csv(file, ignore_errors=True, infer_schema=10000)
                .rename(lambda col: col.strip())
                .with_columns([
                    pl.col("Flow Bytes/s").cast(pl.String, strict=False),               # Cast "Flow Bytes/s" to string to handle potential non-numeric values
                    pl.col("Flow Packets/s").cast(pl.String, strict=False),             # Cast "Flow Packets/s" to string to handle potential non-numeric values
                ])
                .filter(
                    pl.col("Label").str.contains("(?i)portscan|recon|bot|probing") | (pl.col("Label") == "BENIGN")
                )
            )
            lazy_dfs.append(df_lazy)

        pbar = self.logger.progress_bar("Execution Polar Streaming Query", total=2) 
        combined_lazy = pl.concat(lazy_dfs, how="diagonal_relaxed")                     # Concatenate all LazyFrames into a single LazyFrame for further processing
        pbar.update(1)

        cleaned_lazy = combined_lazy.with_columns([
            pl.when(pl.col("Flow Bytes/s").str.contains("(?i)infinity|nan"))
            .then(None)
            .otherwise(pl.col("Flow Bytes/s"))
            .cast(pl.Float64, strict=False)
            .alias("Flow Bytes/s"),
            pl.when(pl.col("Flow Packets/s").str.contains("(?i)infinity|nan"))
            .then(None)
            .otherwise(pl.col("Flow Packets/s"))
            .cast(pl.Float64, strict=False)
            .alias("Flow Packets/s"),
        ])

        df = cleaned_lazy.collect(streaming=True)
        pbar.update(1)
        pbar.close()

        self.logger.log_info(f"Writing parquet: {df.height} records saved in {output_file}")
        df.write_parquet(output_file, compression="zstd")
        return output_file

    def process_unswnb15(self) -> Path:
        """ 
            Process the UNSW-NB15 dataset by reading CSV files, filtering relevant records, and
            saving the cleaned data as a Parquet file.

            Returns:
                Path: The path to the saved Parquet file containing the processed dataset.
        """

        output_file = self.processed_dir / "unswnb15_full.parquet"
        csv_files = [
            f for f in (self.raw_dir / "UNSW-NB15").glob("*.csv")
            if "training-set" in f.name.lower() or "testing-set" in f.name.lower()
        ]

        if not csv_files:
            self.logger.log_warning("No CSV founded for UNSW-NB15")
            return output_file

        lazy_dfs = []
        for file in self.logger.track_iterable(csv_files, "Parsing CSV UNSW-NB15"):
            df_lazy = (
                pl.scan_csv(file, ignore_errors=True, infer_schema_length=10000)
                .rename(lambda col: col.strip().lower())
                )
            lazy_dfs.append(df_lazy)

        pbar = self.logger.progress_bar("Execution Polar Streaming Query", total=2)
        combined_lazy = pl.concat(lazy_dfs)
        pbar.update(1)

        df = combined_lazy.collect(streaming=True)
        pbar.update(1)
        pbar.close()

        self.logger.log_info(f"Writing parquet: {df.height} records saved in {output_file}")
        df.write_parquet(output_file, compression="zstd")
        return output_file


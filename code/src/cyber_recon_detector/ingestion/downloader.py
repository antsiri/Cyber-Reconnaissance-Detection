"""
 * author Antonio Sirignano
 * created on 08-09-2026-14h-20m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import os
import zipfile
import requests

from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from cyber_recon_detector.utils.terminal import TerminalLogger

class DatasetDownloader:
    """
        Download manager for remote datasets with local caching.
    """

    DATASET_URLS: Dict[str, List[str]] = {
        "CIC-IDS2017": [
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Monday-WorkingHours.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Tuesday-WorkingHours.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Wednesday-workingHours.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        ],
        "UNSW-NB15": [
            "https://www.kaggle.com/api/v1/datasets/download/mrwellsdavid/unsw-nb15",
        ], 
    }

    def __init__(self, raw_dir: str | Path, logger: TerminalLogger) -> None:
        """
            Initialize the downloader with a raw data directory and a logger.

            Args:
                raw_dir (str | Path): The directory where raw datasets will be stored.
                logger (TerminalLogger): An instance of TerminalLogger for logging messages.
        """

        self.raw_dir = Path(raw_dir)
        self.logger = logger

    def download_file(self, url: str, destination: Path) -> Path:
        """
            Download the single file in streaming mode

            Args:
                url (str): The URL of the file to download.
                destination (Path): The local path where the downloaded file will be saved.

            Returns:
                Path: The path to the downloaded file.
        """

        response = requests.get(url=url, stream=True, timeout=90)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024    #1MB Chunk

        with open(destination, "wb") as file, tqdm(
            desc=f"\033[1;34m[DOWNLOAD]\033[0m {destination.name}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True, 
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        if destination.stat().st_size < 100_000:
            destination.unlink()
            raise ValueError(f"Downloaded file not valid (dimesion too little): {url}")

        return destination

    def fetch_dataset(self, dataset_name: str) -> Path:
        """
            Download dataset if it's alreay presented
            
            Args:
                dataset_name (str): The name of the dataset to download.

            Returns:
                Path: The path to the directory containing the downloaded dataset.
        """

        target_dir = self.raw_dir / dataset_name
        target_dir.mkdir(parents=True, exist_ok=True)

        urls = self.DATASET_URLS.get(dataset_name, [])

        for idx, url in enumerate(urls, 1):
            file_name = url.split("/")[-1]
            if not file_name.endswith(".csv"):
                file_name = f"{dataset_name}_{idx}.csv"

            final_path = target_dir / file_name

            if final_path.exists() and final_path.stat().st_size > 1_000_000:
                continue

            temp_file = target_dir / f"download_temp_{file_name}"
            try:
                self.logger.log_info(f"Missing file downloading: {file_name}")
                downloaded_path = self.download_file(url, temp_file)

                if zipfile.is_zipfile(downloaded_path):
                    self.logger.log_info(
                        f"ZIP file extraction for '{dataset_name}'"
                    )
                    with zipfile.ZipFile(downloaded_path, "r") as zip_ref:
                        zip_ref.extractall(target_dir)
                    downloaded_path.unlink()

                else:
                    downloaded_path.rename(final_path)

            except Exception as e:
                self.logger.log_error(
                    f"Error during '{dataset_name}' download: {e}"
                )
                if temp_file.exists():
                    temp_file.unlink()

        return target_dir
                


    


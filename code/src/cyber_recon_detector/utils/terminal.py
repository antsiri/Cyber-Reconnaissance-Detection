"""
 * author Antonio Sirignano
 * created on 08-09-2026-13h-53m
 * github: https://github.com/antsiri
 * copyright 2026
"""

import sys 

from typing import Optional, Iterable, TypeVar
from tqdm import tqdm

T = TypeVar("T")

class TerminalLogger:
    """
        Utility class to manage terminal logging and progress bars using `tqdm` 
    """

    @staticmethod
    def log_info(message: str) -> None:
        """
            Print info message  

            Args:
                message (str): The message to be printed   
        """

        tqdm.write(f"\033[92m[INFO]\033[0m {message}]]")

    @staticmethod
    def log_warning(message: str) -> None:
        """
            Print advise message

            Args:
                message (str): The message to be printed
        """

        tqdm.write(f"\033[93m[WRN]\033[0m {message}]]")

    @staticmethod
    def log_error(message: str) -> None:
        """
            Print error message 

            Args:
                message (str): The message to be printed
        """

        tqdm.write(f"\033[91m[ERROR]\033[0m {message}]]")

    @staticmethod
    def track_iterable(
        iterable: Iterable[T],
        description: str, 
        total: Optional[int] = None,
    ) -> Iterable:
        """
            Wrapper for general iterable item for `tqdm`

            Args:
                iterable (Iterable[T]): The iterable to be tracked
                description (str): The description for the progress bar
                total (Optional[int]): The total number of items

            Returns:
                Iterable: The tqdm progress bar
        """

        return tqdm(
            iterable,
            desc = f"\033[1;34m[PROGRESS]\033[0m {description}]]",
            total=total,
            unit="item",
            leave=True,
            dynamic_ncols=True
        )

    @staticmethod
    def progress_bar(description: str, total: int) -> tqdm:
        """
            Return manual progress bar with `tqdm` with step-by-step management

            Args:
                description (str): The description for the progress bar
                total (int): The total number of steps

            Returns:
                tqdm: The tqdm progress bar
        """

        return tqdm(
            total=total,
            desc=f"\033[1;34m[PROGRESS]\033[0m {description}]]",
            unit="step",
            leave=True,
            dynamic_ncols=True
        )


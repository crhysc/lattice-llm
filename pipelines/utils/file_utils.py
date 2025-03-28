"""
file_utils.py

Provides utilities for file/directory operations and CSV handling.
"""

import os
import sys
import csv
import logging


logger = logging.getLogger(__name__)


def create_output_directory(directory: str) -> None:
    """
    Creates the specified output directory if it does not exist.

    Args:
        directory (str): Path to the output directory.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info("Created output directory '%s'.", directory)
        except OSError as exc:
            logger.error("Error creating directory '%s': %s", directory, exc)
            sys.exit(1)
    else:
        logger.info("Output directory '%s' already exists.", directory)


def write_csv(csv_path: str, rows: list) -> None:
    """
    Writes rows to a CSV file.

    Args:
        csv_path (str): Path to CSV file to write.
        rows (list): A list of lists, where each sub-list is one row of data.
    """
    try:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)
        logger.info("id_prop.csv written at: %s", csv_path)
    except IOError as exc:
        logger.error("Failed to write CSV at '%s': %s", csv_path, exc)


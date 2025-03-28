"""
data_preprocessor.py

Coordinates the scanning of input directories, reading Tc.dat and CIFs, 
and generating POSCAR files along with an id_prop.csv.
"""

import os
import logging
import glob

from utils.file_utils import create_output_directory, write_csv
from utils.structure_utils import parse_tc_from_file, load_structure, write_poscar, sanitize_filename


logger = logging.getLogger(__name__)


def process_superconducting_dataset(input_dir: str, output_dir: str, filename_prefix: str, tc_key: str) -> None:
    """
    Main data preprocessing function to generate POSCAR files and 
    an id_prop.csv for a superconducting dataset.

    Args:
        input_dir (str): Directory containing 'batch-*' subdirectories.
        output_dir (str): Directory where generated files and CSV are saved.
        filename_prefix (str): Prefix for POSCAR filenames.
        tc_key (str): T_c key in 'Tc.dat' (e.g. 'Tc_OPT' or 'Tc_AD').
    """
    create_output_directory(output_dir)
    csv_path = os.path.join(output_dir, "id_prop.csv")

    batch_dirs = glob.glob(os.path.join(input_dir, "batch-*"))
    if not batch_dirs:
        logger.error("No 'batch-*' directories found in '%s'. Exiting.", input_dir)
        return

    all_rows = []  # We'll accumulate rows to write to id_prop.csv.

    for batch_dir in sorted(batch_dirs):
        if not os.path.isdir(batch_dir):
            continue
        logger.info("Processing batch directory: %s", batch_dir)

        structure_dirs = os.listdir(batch_dir)
        for structure_dir in sorted(structure_dirs):
            structure_path = os.path.join(batch_dir, structure_dir)
            if not os.path.isdir(structure_path):
                continue

            tc_dat_path = os.path.join(structure_path, "Tc.dat")
            cif_path = os.path.join(structure_path, "geo_opt.cif")

            # Skip if required files do not exist
            if not (os.path.exists(tc_dat_path) and os.path.exists(cif_path)):
                logger.warning("Skipping %s because Tc.dat or geo_opt.cif was not found.", structure_path)
                continue

            # Parse T_c
            chosen_tc = parse_tc_from_file(tc_dat_path, tc_key)
            if chosen_tc is None:
                logger.warning("Skipping %s due to missing/invalid T_c for %s.", structure_path, tc_key)
                continue

            # Load structure from CIF
            structure = load_structure(cif_path)
            if structure is None:
                logger.warning("Skipping structure at %s because the CIF could not be loaded.", cif_path)
                continue

            # Generate sanitized POSCAR filename
            sanitized_name = sanitize_filename(structure_dir)
            poscar_filename = f"{filename_prefix}{sanitized_name}.vasp"
            poscar_path = os.path.join(output_dir, poscar_filename)

            # Write the POSCAR
            if not write_poscar(structure, poscar_path):
                logger.warning("Skipping structure %s due to POSCAR write failure.", structure_path)
                continue

            # Accumulate row for CSV: [T_c, path_to_POSCAR]
            all_rows.append([chosen_tc, poscar_path])

    # Write out CSV
    write_csv(csv_path, all_rows)
    logger.info("All done! Your id_prop.csv is at: %s", csv_path)


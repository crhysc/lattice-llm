"""
structure_utils.py

Provides utilities for working with material structures (POSCAR, CIF, etc.).
"""

import logging
import os
from pymatgen.core import Structure


logger = logging.getLogger(__name__)


def parse_tc_from_file(tc_file_path: str, tc_key: str) -> float:
    """
    Parses the specified T_c from a Tc.dat file.

    Args:
        tc_file_path (str): Path to 'Tc.dat' file.
        tc_key (str): Key to look for (e.g., 'Tc_OPT', 'Tc_AD').

    Returns:
        float: The parsed T_c value, or None if not found / parse error.
    """
    if not os.path.exists(tc_file_path):
        logger.warning("Tc.dat file not found at %s", tc_file_path)
        return None

    with open(tc_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Example line: "Tc_OPT: 16.453±2.125"
            if line.startswith(tc_key + ":"):
                try:
                    # Remove the "Tc_OPT:" portion
                    raw_value = line.split(":", 1)[1].strip()  # e.g. "16.453±2.125"
                    # If there's an "±", split on it and use the left part
                    if "±" in raw_value:
                        raw_value = raw_value.split("±", 1)[0].strip()
                    return float(raw_value)
                except ValueError as exc:
                    logger.error("Error parsing T_c from line '%s': %s", line, exc)
                    return None

    logger.warning("Could not find a line starting with '%s:' in %s", tc_key, tc_file_path)
    return None


def load_structure(cif_path: str) -> Structure:
    """
    Loads a structure from a CIF file via pymatgen.

    Args:
        cif_path (str): Path to the CIF file.

    Returns:
        Structure: A pymatgen Structure object, or None if failure.
    """
    if not os.path.exists(cif_path):
        logger.warning("CIF file not found at %s", cif_path)
        return None

    try:
        structure = Structure.from_file(cif_path)
        return structure
    except Exception as exc:
        logger.error("Error reading CIF at %s: %s", cif_path, exc)
        return None


def write_poscar(structure: Structure, poscar_path: str) -> bool:
    """
    Writes a Structure to a VASP POSCAR file.

    Args:
        structure (Structure): The pymatgen Structure to write.
        poscar_path (str): The output POSCAR path.

    Returns:
        bool: True if written successfully, False otherwise.
    """
    try:
        structure.to(fmt="poscar", filename=poscar_path)
        logger.info("Generated POSCAR: %s", poscar_path)
        return True
    except Exception as exc:
        logger.error("Error writing POSCAR to %s: %s", poscar_path, exc)
        return False


def sanitize_filename(name: str) -> str:
    """
    Sanitizes the filename by removing or replacing invalid characters.

    Args:
        name (str): Original filename.

    Returns:
        str: Sanitized filename.
    """
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)


import os
import csv
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
from datasets import load_dataset


class ChemInfoStrategy(ABC):
    """Interface for building chemical info prefixes."""

    @abstractmethod
    def get_prefix(self, chem: str) -> str:
        """Return a string prefix based on chemical info type."""
        pass


class NoChemInfoStrategy(ChemInfoStrategy):
    """No chemical info is appended."""

    def get_prefix(self, chem: str) -> str:
        return ""


class ElementListChemInfoStrategy(ChemInfoStrategy):
    """Prefix strategy for listing elements."""

    def get_prefix(self, chem: str) -> str:
        return f"The chemical elements are {chem} . "


class FormulaChemInfoStrategy(ChemInfoStrategy):
    """Prefix strategy for specifying the formula."""

    def get_prefix(self, chem: str) -> str:
        return f"The chemical formula is {chem} . "


class ChemInfoStrategyFactory:
    """Factory to create appropriate ChemInfoStrategy objects."""

    @staticmethod
    def create(chem_info_type: str) -> ChemInfoStrategy:
        """Return the correct strategy class based on 'chem_info' setting."""
        if chem_info_type == "none":
            return NoChemInfoStrategy()
        elif chem_info_type == "element_list":
            return ElementListChemInfoStrategy()
        elif chem_info_type == "formula":
            return FormulaChemInfoStrategy()
        else:
            raise ValueError(f"Unknown chem_info type: {chem_info_type}")


def get_input(config, chem: str, val: float) -> str:
    """
    Return the text prompt for each data row.

    This function uses a strategy-based approach to insert chemical info
    into the prompt. Additional 'chem_info' types can be added by creating
    new subclasses of ChemInfoStrategy.
    """
    strategy = ChemInfoStrategyFactory.create(config.chem_info)
    prefix = strategy.get_prefix(chem)
    inp = (
        prefix
        + f"The {config.prop} is {val}."
        + config.output_prompt
    )
    return inp


def make_alpaca_json(dataset: List[Dict[str, Any]],
                     jids: List[str],
                     config) -> List[Dict[str, Any]]:
    """
    Generate 'alpaca style' JSON from raw data rows.

    Only keeps entries whose property is not 'na' and is in jids.
    """
    mem = []
    for entry in dataset:
        if entry[config.prop] != "na" and entry[config.id_tag] in jids:
            atoms = Atoms.from_dict(entry["atoms"])

            if config.chem_info == "element_list":
                chem = atoms.composition.search_string
            elif config.chem_info == "formula":
                chem = atoms.composition.reduced_formula
            else:
                chem = ""

            info = {
                "instruction": config.instruction,
                "input": get_input(
                    config=config,
                    val=entry[config.prop],
                    chem=chem
                ),
                # This function presumably comes from your script.
                "output": get_crystal_string_t(atoms),
                "id": entry[config.id_tag],
            }
            mem.append(info)
    return mem


def formatting_prompts_func(examples: Dict[str, List[str]],
                            alpaca_prompt: str) -> Dict[str, List[str]]:
    """
    Apply the alpaca-style template.

    For each example in 'examples', constructs a text by using
    the provided alpaca prompt format string.
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    eos_token = "</s>"
    texts = []
    for instruction, inp, out in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, inp, out) + eos_token
        texts.append(text)
    return {"text": texts}


def load_id_prop_data(id_prop_path: str, config) -> List[Dict[str, Any]]:
    """
    Load data from CSV and merges with Atoms info.

    Returns a list of dicts with 'id', the user-defined property,
    and the 'atoms' dict. You can extend this to handle multiple
    file formats or additional data merges without modifying the
    existing logic (e.g., implement new file loaders).
    """
    run_path = os.path.dirname(id_prop_path)
    data_rows = []
    with open(id_prop_path, "r", encoding="utf-8") as file_in:
        reader = csv.reader(file_in)
        data_rows = [row for row in reader]

    data_list = []
    for row in data_rows:
        info = {"id": row[0]}

        # Merge property
        tmp_vals = [float(j) for j in row[1:]]
        if len(tmp_vals) == 1:
            info[config.prop] = str(tmp_vals[0])
        else:
            info[config.prop] = "\n".join(map(str, tmp_vals))

        # Load Atoms based on config.file_format
        pth = os.path.join(run_path, info["id"])
        # Example extension point:
        # if config.file_format == "json":
        #     atoms = Atoms.from_dict(loadjson(pth + ".json"))
        # elif config.file_format == "poscar":
        #     ...
        # else:
        #     ...
        # For now, assume you already have `atoms` from some logic:
        atoms = Atoms()  # placeholder
        info["atoms"] = atoms.to_dict()

        data_list.append(info)

    return data_list

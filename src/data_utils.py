# data_utils.py
import os
import csv
from typing import List, Dict, Any
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
from datasets import load_dataset

def get_input(config, chem: str, val: float) -> str:
    """Returns the text prompt for each data row."""
    # Same logic you currently have in get_input()
    if config.chem_info == "none":
        prefix = ""
    elif config.chem_info == "element_list":
        prefix = "The chemical elements are " + chem + " . "
    elif config.chem_info == "formula":
        prefix = "The chemical formula is " + chem + " . "
    inp = (
        prefix
        + "The " + config.prop + " is " + str(val)
        + "."
        + config.output_prompt
    )
    return inp

def make_alpaca_json(dataset: List[Dict], jids: List[str], config) -> List[Dict]:
    """Generates 'alpaca style' JSON from raw data rows."""
    mem = []
    for i in dataset:
        if i[config.prop] != "na" and i[config.id_tag] in jids:
            atoms = Atoms.from_dict(i["atoms"])
            info = {
                "instruction": config.instruction,
                "input": get_input(config=config,
                                   val=i[config.prop],
                                   chem=atoms.composition.search_string
                                       if config.chem_info == "element_list"
                                       else atoms.composition.reduced_formula
                                       if config.chem_info == "formula"
                                       else "")
            }
            info["output"] = get_crystal_string_t(atoms)  # from your script
            info["id"] = i[config.id_tag]
            mem.append(info)
    return mem

def formatting_prompts_func(examples, alpaca_prompt: str):
    """Applies the alpaca-style template."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    EOS_TOKEN = "</s>"
    texts = []
    for instruction, inp, out in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, inp, out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def load_id_prop_data(id_prop_path: str, config) -> List[Dict]:
    """Loads data from CSV and merges with Atoms info."""
    run_path = os.path.dirname(id_prop_path)
    dt = []
    with open(id_prop_path, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]

    data_list = []
    for row in dt:
        info = {"id": row[0]}
        # Merge property
        tmp_vals = [float(j) for j in row[1:]]
        if len(tmp_vals) == 1:
            info[config.prop] = str(tmp_vals[0])
        else:
            info[config.prop] = "\n".join(map(str, tmp_vals))
        # Load Atoms
        pth = os.path.join(run_path, info["id"])
        # Based on config.file_format, load Atoms
        ...
        info["atoms"] = atoms.to_dict()
        data_list.append(info)

    return data_list

# evaluate.py
import csv
from tqdm import tqdm
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import Atoms
from typing import List, Dict

def evaluate(
    test_set: List[Dict],
    model,
    tokenizer,
    csv_out: str,
    config
):
    """Evaluate a set of examples (test_set). Writes CSV with target/pred/prompt."""
    print("Testing set size:", len(test_set))
    with open(csv_out, "w") as f:
        f.write("id,target,prediction\n")
        for row in tqdm(test_set, total=len(test_set)):
            prompt = row["input"]
            gen_mat = gen_atoms(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                alpaca_prompt=config.alpaca_prompt,
                instruction=config.instruction,
            )
            target_mat = text2atoms("\n" + row["output"])
            line = (
                row["id"] + ","
                + Poscar(target_mat).to_string().replace("\n","\\n")
                + ","
                + Poscar(gen_mat).to_string().replace("\n","\\n")
                + "\n"
            )
            f.write(line)
    print("Evaluation complete. Output saved to", csv_out)

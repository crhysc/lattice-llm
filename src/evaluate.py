import csv
from typing import List, Dict

from tqdm import tqdm
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import Atoms


def evaluate(
    test_set: List[Dict],
    model,
    tokenizer,
    csv_out: str,
    config
):
    """
    Evaluate a set of examples (test_set). Writes CSV with target/pred/prompt.

    :param test_set: A list of dictionaries, each containing 'input', 'output', 'id', etc.
    :param model: The trained model ready for inference.
    :param tokenizer: A tokenizer compatible with the model.
    :param csv_out: Path to output CSV.
    :param config: A config object (containing prompt settings, etc.).
    """
    print("Testing set size:", len(test_set))
    with open(csv_out, "w", encoding="utf-8") as file_out:
        writer = csv.writer(file_out)
        writer.writerow(["id", "target", "prediction"])

        for row in tqdm(test_set, total=len(test_set)):
            prompt = row["input"]
            # 'gen_atoms' and 'text2atoms' are assumed to be defined somewhere else
            generated_atoms = gen_atoms(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                alpaca_prompt=config.alpaca_prompt,
                instruction=config.instruction,
            )
            target_atoms = text2atoms("\n" + row["output"])

            # Convert them to POSCAR strings
            target_str = Poscar(target_atoms).to_string().replace("\n", "\\n")
            generated_str = Poscar(generated_atoms).to_string().replace("\n", "\\n")

            writer.writerow([row["id"], target_str, generated_str])

    print("Evaluation complete. Output saved to", csv_out)

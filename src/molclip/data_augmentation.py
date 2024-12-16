import os
import random

import google.generativeai as genai
import polars as pl
from datasets import Dataset
from dotenv import load_dotenv
from rdkit import Chem

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def augment_smiles(smiles: str, factor: int = 10, max_iteration: int = 10000) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * factor

    cnt = 0
    random_smiles_set = set()
    while len(random_smiles_set) < factor and cnt < max_iteration:
        random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        if random_smiles not in random_smiles_set:
            random_smiles_set.add(random_smiles)
        cnt += 1
    return list(random_smiles_set)


def get_prompt(description: str, factor: int) -> str:
    return f"""
The following is a description of a molecule.
Please provide {factor} paraphrases of this text.
Note that:
- the text should be coherent and meaningful.
- the text should not be a verbatim copy of the original text.
- the text should not mean something completely different from the original text.
- {factor} different texts should be provided.

Original text:
{description}
"""


def augment_text(text: str, factor: int = 10) -> list[str]:
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        get_prompt(text, factor),
        generation_config=genai.GenerationConfig(response_mime_type="application/json", response_schema=list[str]),
    )
    output = eval(response.parts[0].text)
    if not isinstance(output, list):
        return augment_text(text, factor)
    if len(output) < factor:
        return augment_text(text, factor)
    return output[:factor]


def augment(dataset_url: str) -> pl.DataFrame:
    dataset = pl.read_csv(dataset_url, separator="\t")
    for idx in range(len(dataset)):
        smiles = dataset["SMILES"][idx]
        text = dataset["description"][idx]
        augmented_smiles = augment_smiles(smiles, factor=10)
        augmented_text = augment_text(text, factor=10)
        augmented_smiles = random.shuffle(augmented_smiles)
        augmented_text = random.shuffle(augmented_text)
        new_rows = pl.DataFrame(
            {
                "SMILES": augmented_smiles,
                "description": augmented_text,
            }
        )
        dataset = dataset.vstack(new_rows)
    return dataset


if __name__ == "__main__":
    train_dataset_url = "https://raw.githubusercontent.com/blender-nlp/MolT5/refs/heads/main/ChEBI-20_data/train.txt"
    validation_dataset_url = (
        "https://raw.githubusercontent.com/blender-nlp/MolT5/refs/heads/main/ChEBI-20_data/validation.txt"
    )
    test_dataset_url = "https://raw.githubusercontent.com/blender-nlp/MolT5/refs/heads/main/ChEBI-20_data/test.txt"

    train_dataset = augment(train_dataset_url)
    validation_dataset = augment(validation_dataset_url)
    test_dataset = augment(test_dataset_url)

    hf_path = "kohbanye/ChEBI-20-augmented"
    Dataset.from_polars(train_dataset).push_to_hub(hf_path, split="train")
    Dataset.from_polars(validation_dataset).push_to_hub(hf_path, split="validation")
    Dataset.from_polars(test_dataset).push_to_hub(hf_path, split="test")

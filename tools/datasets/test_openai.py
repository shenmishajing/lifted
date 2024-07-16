from string import Template
import openai
import yaml

import pandas as pd


def load_client(key_path="openai_key.yaml"):
    openai._reset_client()
    key = yaml.safe_load(open(key_path))
    for k, v in key.items():
        setattr(openai, k, v)
    return openai._load_client()


def main():
    data = pd.read_csv("data/labeling/phase_II_train.csv")
    row = data.iloc[2033]
    linearization = [
        "; ".join([f"{name}: {value}" for name, value in zip(row.index, row)])
    ]

    schema_definition = (
        "phase: the phase of the trial. phase I, or phase II, or phase III.\n"
        + "diseases: list of disease names.\n"
        + "icdcodes: list of icd-10 codes of diseases.\n"
        + "drugs: list of drug names.\n"
        + "smiless: list of SMILES of the drugs.\n"
        + "criteria: eligibility criteria."
    )

    chat_kwargs = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": Template(
                    "Here is the schema definition of the table:\n"
                    + "$schema_definition\n"
                    + "This is a sample from the table:\n"
                    + "$linearization\n"
                    + "Please describe the sample using natural language.",
                ).safe_substitute(
                    schema_definition=schema_definition, linearization=linearization
                ),
            },
        ],
    }
    client = load_client("openai_key.yaml")
    res = client.chat.completions.create(**chat_kwargs).to_dict()
    print(res)


if __name__ == "__main__":
    main()

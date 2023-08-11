import json
import os

import pandas as pd
import xmltodict
from rdkit import Chem
from tqdm import tqdm


def cache_drug_description(
    data_root="data/clinical-trial-outcome-prediction/data/drugbank",
    data_name="drugbank_database.xml",
):
    if os.path.exists(os.path.join(data_root, "drug_description.json")):
        return

    data = xmltodict.parse(open(os.path.join(data_root, data_name)).read())
    data = data["drugbank"]["drug"]
    data = {
        drug["name"]: drug["description"]
        for drug in tqdm(data, desc="drug description")
    }
    json.dump(
        data, open(os.path.join(data_root, "drug_description.json"), "w"), indent=4
    )


def cache_drug_smiles_from_csv(
    data_root="data/clinical-trial-outcome-prediction/data/drugbank",
    data_name="drugbank_structures.csv",
):
    if os.path.exists(os.path.join(data_root, "drug_smiless.json")):
        return

    data = pd.read_csv(os.path.join(data_root, data_name))
    result = {}

    for _, row in tqdm(data.iterrows(), desc="drug smiles"):
        result[row["Name"]] = row["SMILES"]

    json.dump(result, open(os.path.join(data_root, "drug_smiless.json"), "w"), indent=4)


def cache_drug_smiles_from_sdf(
    data_root="data/clinical-trial-outcome-prediction/data/drugbank",
    data_name="drugbank_structures.sdf",
):
    if os.path.exists(os.path.join(data_root, "drug_smiless.json")):
        return

    data = Chem.SDMolSupplier(os.path.join(data_root, data_name))
    result = {
        d.GetProp("GENERIC_NAME"): d.GetProp("SMILES")
        for d in tqdm(data, desc="drug smiles")
        if d is not None
    }

    json.dump(result, open(os.path.join(data_root, "drug_smiless.json"), "w"), indent=4)


def main():
    cache_drug_description()
    cache_drug_smiles_from_csv()


if __name__ == "__main__":
    main()

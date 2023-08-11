import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from psutil import cpu_count
from tqdm import tqdm


def get_icd_from_nih(disease_name):
    prefix = (
        "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code,name&terms="
    )
    url = prefix + disease_name
    response = requests.get(url)
    text = response.text
    if text == "[0,[],null,[]]":
        return None
    text = text[1:-1]
    idx1 = text.find("[")
    idx2 = text.find("]")
    codes = text[idx1 + 1 : idx2].split(",")
    codes = [i[1:-1] for i in codes]
    return codes


def xmlfile2results(data_root, xml_file, drug_to_smiles_data_path):
    icd_code_dict = json.load(open(os.path.join(data_root, "icd_code_dict.json")))
    drug_to_smiles = json.load(open(drug_to_smiles_data_path))
    tree = ET.parse(os.path.join(data_root, xml_file))
    root = tree.getroot()
    nctid = root.find("id_info").find("nct_id").text  ### nctid: 'NCT00000102'
    drugs = [i for i in root.findall("intervention")]
    drugs = [
        i.find("intervention_name").text
        for i in drugs
        if i.find("intervention_type").text == "Drug"
    ]
    if len(drugs) == 0:
        return None

    try:
        status = root.find("overall_status").text
    except:
        status = ""

    try:
        why_stop = root.find("why_stopped").text
    except:
        why_stop = ""

    try:
        phase = root.find("phase").text
    except:
        phase = ""
    diseases = [i.text for i in root.findall("condition")]  ### disease

    icd_codes = set()
    for disease in diseases:
        icd_codes.update(
            icd_code_dict[disease]
            if disease in icd_code_dict and icd_code_dict[disease]
            else []
        )
    icd_codes = sorted(icd_codes)

    try:
        criteria = root.find("eligibility").find("criteria").find("textblock").text
    except:
        criteria = ""

    smiless = []

    for drug in drugs:
        if drug in drug_to_smiles:
            smiless.append(drug_to_smiles[drug])
        else:
            print(f"drug not found: {drug}")

    try:
        brief_summary = root.find("brief_summary").find("textblock").text
    except:
        brief_summary = ""

    return [
        nctid,
        status,
        why_stop,
        phase,
        diseases,
        icd_codes,
        drugs,
        smiless,
        criteria,
        brief_summary,
    ]


def get_disease_name(data_root, xml_file):
    root = ET.parse(os.path.join(data_root, xml_file)).getroot()

    return set([i.text for i in root.findall("condition")])


def get_all_disease_names(data_root, num_process=None, chunksize=100):
    if os.path.exists(os.path.join(data_root, "disease_names.json")):
        return

    data_path = os.path.join(data_root, "data.xml")
    data_names = open(data_path).read().split("\n")
    data_names = [name for name in data_names if name]

    disease_names = set()

    if num_process is None:
        num_process = cpu_count(False)

    if num_process > 1:
        num_process = min(num_process, len(data_names))
        with ProcessPoolExecutor(num_process) as executor:
            for names in tqdm(
                executor.map(
                    get_disease_name,
                    [data_root] * len(data_names),
                    data_names,
                    chunksize=chunksize,
                ),
                total=len(data_names),
                desc="get_all_disease_names",
            ):
                disease_names.update(names)
    else:
        for data_name in tqdm(data_names, desc="get_all_disease_names"):
            disease_names.update(get_disease_name(data_root, data_name))

    json.dump(
        sorted(disease_names),
        open(os.path.join(data_root, "disease_names.json"), "w"),
        indent=4,
    )


def get_icd_code_dict(data_root, num_process=None):
    if os.path.exists(os.path.join(data_root, "icd_code_dict.json")):
        return

    disease_names = json.load(open(os.path.join(data_root, "disease_names.json")))
    icd_code_dict = {}

    if num_process is None:
        num_process = cpu_count(False)

    if num_process > 1:
        num_process = min(num_process, len(disease_names))
        with ThreadPoolExecutor(num_process) as executor:
            for name, icd_code in zip(
                disease_names,
                tqdm(
                    executor.map(get_icd_from_nih, disease_names),
                    total=len(disease_names),
                    desc="get_icd_code_dict",
                ),
            ):
                icd_code_dict[name] = icd_code
    else:
        for name in tqdm(disease_names, desc="get_icd_code_dict"):
            icd_code_dict[name] = get_icd_from_nih(name)

    json.dump(
        icd_code_dict,
        open(os.path.join(data_root, "icd_code_dict.json"), "w"),
        indent=4,
    )


def get_data(data_root, num_process=None, chunksize=100):
    if os.path.exists(os.path.join(data_root, "data.csv")):
        return

    data_path = os.path.join(data_root, "data.xml")
    data_names = open(data_path).read().split("\n")
    data_names = [name for name in data_names if name]

    if num_process is None:
        num_process = cpu_count(False)

    if num_process > 1:
        num_process = min(num_process, len(data_names))
        with ProcessPoolExecutor(num_process) as executor:
            data = list(
                tqdm(
                    executor.map(
                        xmlfile2results,
                        [data_root] * len(data_names),
                        data_names,
                        [
                            "data/clinical-trial-outcome-prediction/data/drugbank/drug_smiless.json"
                        ]
                        * len(data_names),
                        chunksize=chunksize,
                    ),
                    total=len(data_names),
                    desc="data",
                )
            )
    else:
        data = []
        for data_name in tqdm(data_names, desc="data"):
            data.append(
                xmlfile2results(
                    data_root,
                    data_name,
                    "data/clinical-trial-outcome-prediction/data/drugbank/drug_smiless.json",
                )
            )

    data = [i for i in data if i]

    data = pd.DataFrame(
        data,
        columns=[
            "nctid",
            "status",
            "why_stop",
            "phase",
            "diseases",
            "icd_codes",
            "drugs",
            "smiless",
            "criteria",
            "brief_summary",
        ],
    )
    data.to_csv(os.path.join(data_root, "data.csv"), index=False)


def main():
    data_root = "data/clinical_trials_gov"

    get_all_disease_names(data_root)
    get_icd_code_dict(data_root)
    get_data(data_root)


if __name__ == "__main__":
    main()

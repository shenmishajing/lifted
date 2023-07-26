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


def xmlfile2results(data_root, xml_file):
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

    icd_codes = None

    try:
        criteria = root.find("eligibility").find("criteria").find("textblock").text
    except:
        criteria = ""

    return [nctid, status, why_stop, phase, diseases, icd_codes, drugs, criteria]


def get_data_without_icd_codes(data_root, num_process=None):
    if not os.path.exists(os.path.join(data_root, "data_without_icd_codes.json")):
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
                            chunksize=len(data_names) // (num_process * 4),
                        ),
                        total=len(data_names),
                        desc="data_without_icd_codes",
                    )
                )
        else:
            data = []
            for data_name in tqdm(data_names, desc="data_without_icd_codes"):
                data.append(xmlfile2results(data_root, data_name))

        data = [i for i in data if i]

        json.dump(
            data,
            open(os.path.join(data_root, "data_without_icd_codes.json"), "w"),
            indent=4,
        )


def get_icd(row):
    icd_codes = set()
    for disease in row[4]:
        icd_codes.update(get_icd_from_nih(disease))
    row[5] = sorted(icd_codes)
    return row


def get_data(data_root, num_process=None):
    if not os.path.exists(os.path.join(data_root, "data.csv")):
        data = json.load(open(os.path.join(data_root, "data_without_icd_codes.json")))

        if num_process is None:
            num_process = cpu_count()

        if num_process > 1:
            num_process = min(num_process, len(data))
            with ThreadPoolExecutor(num_process) as executor:
                data = list(
                    tqdm(executor.map(get_icd, data), total=len(data), desc="data")
                )
        else:
            for row in tqdm(data, desc="data"):
                get_icd(row)

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
                "criteria",
            ],
        )
        data.to_csv(os.path.join(data_root, "data.csv"), index=False)


def main():
    data_root = "data/clinical_trials_gov"

    get_data_without_icd_codes(data_root)
    get_data(data_root)


if __name__ == "__main__":
    main()

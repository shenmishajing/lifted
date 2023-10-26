import csv
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def clean_protocol(protocol):
    protocol = protocol.lower()
    protocol_split = protocol.split("\n")
    filter_out_empty_fn = lambda x: len(x.strip()) > 0
    strip_fn = lambda x: x.strip()
    protocol_split = list(filter(filter_out_empty_fn, protocol_split))
    protocol_split = list(map(strip_fn, protocol_split))
    return protocol_split


def split_protocol(protocol):
    protocol_split = clean_protocol(protocol)
    inclusion_idx, exclusion_idx = len(protocol_split), len(protocol_split)
    for idx, sentence in enumerate(protocol_split):
        if "inclusion" in sentence:
            inclusion_idx = idx
            break
    for idx, sentence in enumerate(protocol_split):
        if "exclusion" in sentence:
            exclusion_idx = idx
            break
    if inclusion_idx + 1 < exclusion_idx + 1 < len(protocol_split):
        inclusion_criteria = protocol_split[inclusion_idx:exclusion_idx]
        exclusion_criteria = protocol_split[exclusion_idx:]
        if not (len(inclusion_criteria) > 0 and len(exclusion_criteria) > 0):
            print(len(inclusion_criteria), len(exclusion_criteria), len(protocol_split))
            exit()
        return inclusion_criteria, exclusion_criteria  ## list, list
    else:
        return (protocol_split,)


def get_all_protocols(input_file="data/raw_data.csv"):
    with open(input_file, "r") as csvfile:
        rows = list(csv.reader(csvfile, delimiter=","))[1:]
    protocols = [row[9] for row in rows]
    return protocols


def collect_cleaned_sentence_set(input_file="data/raw_data.csv"):
    protocol_lst = get_all_protocols(input_file)
    cleaned_sentence_lst = []
    for protocol in protocol_lst:
        result = split_protocol(protocol)
        cleaned_sentence_lst.extend(result[0])
        if len(result) == 2:
            cleaned_sentence_lst.extend(result[1])
    return set(cleaned_sentence_lst)


def save_sentence_bert_dict_pkl(
    input_file="data/raw_data.csv", output_file="data/sentence2embedding.pkl"
):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").cuda()

    cleaned_sentence_set = collect_cleaned_sentence_set(input_file)

    # from biobert_embedding.embedding import BiobertEmbedding

    # biobert = BiobertEmbedding()

    # def text2vec(text):
    #     return biobert.sentence_vector(text)

    protocol_sentence_2_embedding = dict()
    for sentence in tqdm(cleaned_sentence_set):
        protocol_sentence_2_embedding[sentence] = (
            model(**tokenizer(sentence, return_tensors="pt").to("cuda"))
            .pooler_output.detach()
            .cpu()
        )
    pickle.dump(protocol_sentence_2_embedding, open(output_file, "wb"))
    return protocol_sentence_2_embedding


def prepare_criteria_feature(
    data_path="data/clinical-trial-outcome-prediction/data",
    embedding_path="sentence2embedding.pkl",
    output_path="criteria",
):
    os.makedirs(os.path.join(data_path, output_path), exist_ok=True)
    sentence2vec = pickle.load(open(os.path.join(data_path, embedding_path), "rb"))

    for k in sentence2vec:
        sentence2vec[k] = sentence2vec[k].cpu()

    for phase in tqdm(["I", "II", "III"], desc="phase"):
        for split in tqdm(["train", "valid", "test"], desc="split"):
            data = pd.read_csv(
                open(os.path.join(data_path, f"phase_{phase}_{split}.csv"))
            )

            data_list = []
            for i, row in data.iterrows():
                if "criteria" in row and isinstance(row["criteria"], str):
                    inclusion_feature, exclusion_feature = protocol2feature(
                        row["criteria"], sentence2vec
                    )
                    inclusion_feature = inclusion_feature.mean(0).view(1, -1)
                    exclusion_feature = exclusion_feature.mean(0).view(1, -1)
                    data_list.append(
                        torch.cat([inclusion_feature, exclusion_feature], 1)
                    )
                else:
                    data_list.append(torch.zeros(1, 768 * 2))
            data_list = torch.cat(data_list, 0)
            np.save(
                os.path.join(data_path, output_path, f"phase_{phase}_{split}.npy"),
                data_list.numpy(),
            )


def load_sentence_2_vec(data_path="data/sentence2embedding.pkl"):
    sentence_2_vec = pickle.load(open(data_path, "rb"))
    return sentence_2_vec


def protocol2feature(protocol, sentence_2_vec):
    result = split_protocol(protocol)
    inclusion_criteria, exclusion_criteria = result[0], result[-1]
    inclusion_feature = [
        sentence_2_vec[sentence].view(1, -1)
        for sentence in inclusion_criteria
        if sentence in sentence_2_vec
    ]
    exclusion_feature = [
        sentence_2_vec[sentence].view(1, -1)
        for sentence in exclusion_criteria
        if sentence in sentence_2_vec
    ]
    if inclusion_feature == []:
        inclusion_feature = torch.zeros(1, 768)
    else:
        inclusion_feature = torch.cat(inclusion_feature, 0)
    if exclusion_feature == []:
        exclusion_feature = torch.zeros(1, 768)
    else:
        exclusion_feature = torch.cat(exclusion_feature, 0)
    return inclusion_feature, exclusion_feature


def main():
    # save_sentence_bert_dict_pkl(
    #     "data/clinical-trial-outcome-prediction/data/raw_data.csv",
    #     "data/clinical-trial-outcome-prediction/data/sentence2embedding.pkl",
    # )
    prepare_criteria_feature()


if __name__ == "__main__":
    main()

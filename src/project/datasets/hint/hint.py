import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from mmengine.dataset import BaseDataset
from transformers import AutoTokenizer

Tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
SMILESTokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

MaxLength = dict(
    table=1625,
    summarization=265,
    description=7,
    diseases=38,
    diseases_concat=461,
    diseases_summarization=560,
    drugs=54,
    drugs_concat=193,
    drugs_summarization=285,
    smiless=1627,
    smiless_concat=1075,
    smiless_summarization=1182,
    smiless_transformer=512,
    smiless_transformer_concat=512,
    smiless_transformer_summarization=512,
)


def tokenize(tokenizer, text, max_length=None):
    return tokenizer(
        text,
        # padding=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=False,
    )


class HINTDataset(BaseDataset):
    """HINT dataset."""

    def __init__(
        self,
        data_prefix=None,
        ann_file_name=None,
        augment=False,
        input_parts=None,
        **kwargs,
    ):
        self.ann_file_name = ann_file_name
        if data_prefix is None:
            data_prefix = dict(
                data_path="",
                table_path="text_description/processed",
                summarization_path="brief_summary/processed",
                drug_description_path="drugbank/drug_description.json",
                criteria_path="criteria",
            )

        self.augment = augment
        if input_parts is None:
            input_parts = [
                "table",
                "summarization",
                "description",
                "criteria",
            ] + [
                f"{k}{p}"
                for k in ["smiless", "smiless_transformer", "drugs", "diseases"]
                for p in ["", "_concat", "_summarization"]
            ]
        self.input_parts = input_parts

        super().__init__(data_prefix=data_prefix, **kwargs)

    @staticmethod
    def collate_fn(batch):
        def collate(batch):
            res = {}

            for name in ["label", "criteria"]:
                if name in batch[0]:
                    res[name] = torch.stack([b[name] for b in batch], dim=0)

            for name in ["table", "summarization"] + [
                f"{k}_{p}"
                for k in ["smiless", "smiless_transformer", "drugs", "diseases"]
                for p in ["concat", "summarization"]
            ]:
                if name in batch[0]:
                    res[name] = tokenize(
                        Tokenizer
                        if "smiless_transformer" not in name
                        else SMILESTokenizer,
                        [b[name] for b in batch],
                        MaxLength[name],
                    )

            for name in [
                "smiless",
                "smiless_transformer",
                "drugs",
                "diseases",
                "description",
            ]:
                if name in batch[0]:
                    res[name] = [
                        tokenize(
                            Tokenizer
                            if "smiless_transformer" not in name
                            else SMILESTokenizer,
                            b[name],
                            MaxLength[name],
                        )
                        for b in batch
                    ]

            res["idx"] = torch.tensor([b["idx"] for b in batch])
            return res

        res = collate(batch)
        if "augment" in batch[0]:
            res["augment"] = collate([b["augment"] for b in batch])

        return res

    def load_table_data(self):
        if "table" in self.input_parts and os.path.exists(
            os.path.join(self.data_prefix["table_path"], f"{self.ann_file_name}.json")
        ):
            return json.load(
                open(
                    os.path.join(
                        self.data_prefix["table_path"], f"{self.ann_file_name}.json"
                    ),
                )
            )

    def load_summarization_data(self):
        if "summarization" in self.input_parts and os.path.exists(
            os.path.join(
                self.data_prefix["summarization_path"], f"{self.ann_file_name}.json"
            )
        ):
            return json.load(
                open(
                    os.path.join(
                        self.data_prefix["summarization_path"],
                        f"{self.ann_file_name}.json",
                    ),
                )
            )

    def load_drug_description_data(self):
        if "description" in self.input_parts and os.path.exists(
            self.data_prefix["drug_description_path"]
        ):
            return json.load(open(self.data_prefix["drug_description_path"]))

    def load_criteria_data(self):
        if "criteria" in self.input_parts and os.path.exists(
            self.data_prefix["criteria_path"]
        ):
            return torch.from_numpy(
                np.load(
                    os.path.join(
                        self.data_prefix["criteria_path"], f"{self.ann_file_name}.npy"
                    )
                )
            )

    def add_list_data(self, name, list_data, summarization_data=None):
        data = {}
        if name in self.input_parts:
            data[name] = list_data

        if f"{name}_concat" in self.input_parts:
            data[f"{name}_concat"] = ",".join(list_data)

        if f"{name}_summarization" and summarization_data:
            data[
                f"{name}_summarization"
            ] = f"{name}: {','.join(list_data)}; summarization: {summarization_data}"
        return data

    def load_data_list(self):
        data = pd.read_csv(
            os.path.join(self.data_prefix["data_path"], f"{self.ann_file_name}.csv")
        )

        table_data = self.load_table_data()
        summarization_data = self.load_summarization_data()
        drug_description = self.load_drug_description_data()
        criteria_data = self.load_criteria_data()

        data_list = []
        for i, row in data.iterrows():
            cur_data = {"idx": i}

            if "label" in row:
                cur_data["label"] = torch.tensor(row["label"], dtype=torch.long)

            if criteria_data is not None:
                cur_data["criteria"] = criteria_data[i]
                if (cur_data["criteria"] == 0).all():
                    continue

            flag = True

            for name in ["smiless", "drugs", "diseases"]:
                if name in row:
                    d = sorted(set(eval(row[name])))
                    if not d:
                        flag = False
                        continue

                    cur_data.update(
                        self.add_list_data(
                            name,
                            d,
                            summarization_data[i] if summarization_data else None,
                        )
                    )

                    if name == "smiless":
                        cur_data.update(
                            self.add_list_data(
                                "smiless_transformer",
                                d,
                                summarization_data[i] if summarization_data else None,
                            )
                        )

            for name, name_data in zip(
                ["table", "summarization"], [table_data, summarization_data]
            ):
                if name_data:
                    cur_data[name] = name_data[i]

            if drug_description:
                cur_data["description"] = [
                    drug_description[drug_name]
                    if drug_name in drug_description and drug_description[drug_name]
                    else "This is a drug."
                    for drug_name in eval(row["drugs"])
                ]

            if flag:
                data_list.append(cur_data)

        return data_list

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if self.augment:
            data_info["augment"] = self.get_data_info(self._rand_another())
        return self.pipeline(data_info)


def main():
    # max_length = defaultdict(int)
    max_length = defaultdict(list)
    labels = {}
    for phase in ["I", "II", "III"]:
        labels[phase] = {}
        for split in ["train", "valid", "test"]:
            # for split in ["test"]:
            labels[phase][split] = [0, 0]
            dataset = HINTDataset(
                ann_file_name=f"phase_{phase}_{split}",
                data_root="data/clinical-trial-outcome-prediction/data",
            )
            for i in range(len(dataset)):
                data = dataset[i]
                if any(d != "This is a drug." for d in data["description"]):
                    data = {
                        k: v
                        for k, v in data.items()
                        if k
                        in [
                            "smiless",
                            "drugs",
                            "diseases",
                            "summarization",
                            "description",
                        ]
                    }
                    print(data)
                for name in data:
                    if name not in ["label", "sample_idx", "idx", "criteria"]:
                        tokenizer = (
                            Tokenizer
                            if "smiless_transformer" not in name
                            else SMILESTokenizer
                        )
                        max_length[name].append(
                            tokenizer(
                                data[name],
                                padding=True,
                                return_tensors="pt",
                                return_token_type_ids=False,
                            )
                            .data["input_ids"]
                            .shape[-1]
                        )
                labels[phase][split][data["label"].item()] += 1
            labels[phase][split] = labels[phase][split][1] / sum(labels[phase][split])
    for name in max_length:
        max_length[name] = sorted(max_length[name], reverse=True)
    print(labels)
    print(max_length)


if __name__ == "__main__":
    main()

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from mmengine.dataset import BaseDataset
from transformers import AutoTokenizer


class HINTDataset(BaseDataset):
    """
    HINT dataset.
    """

    def __init__(
        self,
        data_prefix=None,
        ann_file_name=None,
        max_lengths=None,
        tokenizer="bert-base-cased",
        augment=False,
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

        if max_lengths is None:
            max_lengths = dict(
                table=1625,
                summarization=265,
                smiless=1627,
                smiless_concat=1075,
                smiless_summarization=1182,
                drugs=54,
                drugs_concat=193,
                drugs_summarization=285,
                diseases=38,
                diseases_concat=461,
                diseases_summarization=560,
                description=381,
            )
        self.max_lengths = max_lengths
        self.augment = augment

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
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
                for k in ["smiless", "drugs", "diseases"]
                for p in ["concat", "summarization"]
            ]:
                if name in batch[0]:
                    res[name] = {}
                    for k in batch[0][name]:
                        res[name][k] = torch.cat([b[name][k] for b in batch], dim=0)

            for name in ["smiless", "drugs", "diseases", "description", "idx"]:
                if name in batch[0]:
                    res[name] = [b[name] for b in batch]
            return res

        res = collate(batch)
        if "augment" in batch[0]:
            res["augment"] = collate([b["augment"] for b in batch])

        return res

    def load_data_list(self):
        def tokenize(text, max_length):
            return self.tokenizer(
                text,
                # padding=True,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )

        data = pd.read_csv(
            os.path.join(self.data_prefix["data_path"], f"{self.ann_file_name}.csv")
        )

        data_path = os.path.join(
            self.data_prefix["table_path"], f"{self.ann_file_name}.json"
        )
        if os.path.exists(data_path):
            table_data = json.load(
                open(
                    os.path.join(
                        self.data_prefix["table_path"], f"{self.ann_file_name}.json"
                    ),
                )
            )
        else:
            table_data = None

        data_path = os.path.join(
            self.data_prefix["summarization_path"], f"{self.ann_file_name}.json"
        )
        if os.path.exists(data_path):
            summarization_data = json.load(
                open(
                    os.path.join(
                        self.data_prefix["summarization_path"],
                        f"{self.ann_file_name}.json",
                    ),
                )
            )
        else:
            summarization_data = None

        if os.path.exists(self.data_prefix["drug_description_path"]):
            drug_description = json.load(
                open(self.data_prefix["drug_description_path"])
            )
        else:
            drug_description = None

        if os.path.exists(self.data_prefix["criteria_path"]):
            print("load criteria")
            start = time.time()
            criteria_data = torch.from_numpy(
                np.load(
                    os.path.join(
                        self.data_prefix["criteria_path"], f"{self.ann_file_name}.npy"
                    )
                )
            )
            print(f"load criteria cost {time.time()-start} seconds")
        else:
            criteria_data = None

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
                    d = list(set(eval(row[name])))
                    if not d:
                        flag = False
                        continue
                    cur_data[name] = tokenize(d, self.max_lengths[name])
                    d = ",".join(d)
                    cur_data[f"{name}_concat"] = tokenize(
                        d, self.max_lengths[f"{name}_concat"]
                    )
                    if summarization_data:
                        cur_data[f"{name}_summarization"] = tokenize(
                            f"{name}: {d}; summarization: {summarization_data[i]}",
                            self.max_lengths[f"{name}_summarization"],
                        )

            for name, name_data in zip(
                ["table", "summarization"], [table_data, summarization_data]
            ):
                if name_data:
                    cur_data[name] = tokenize(name_data[i], self.max_lengths[name])

            if drug_description:
                cur_data["description"] = tokenize(
                    [
                        drug_description[drug_name]
                        if drug_name in drug_description and drug_description[drug_name]
                        else "This is a drug."
                        for drug_name in eval(row["drugs"])
                    ],
                    self.max_lengths["description"],
                )

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
        # for split in ["train", "valid", "test"]:
        for split in ["test"]:
            labels[phase][split] = [0, 0]
            dataset = HINTDataset(
                ann_file_name=f"phase_{phase}_{split}",
                data_root="data/clinical-trial-outcome-prediction/data",
            )
            for i in range(len(dataset)):
                data = dataset[i]
                for name in data:
                    if name in ["label", "sample_idx"]:
                        continue
                    # max_length[name] = max(
                    #     data[name].data["input_ids"].shape[-1], max_length[name]
                    # )
                    max_length[name].append(data[name].data["input_ids"].shape[-1])
                labels[phase][split][data["label"].item()] += 1
            labels[phase][split] = labels[phase][split][1] / sum(labels[phase][split])
    for name in max_length:
        max_length[name] = sorted(max_length[name], reverse=True)
    print(labels)
    print(max_length)


if __name__ == "__main__":
    main()

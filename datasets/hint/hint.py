import json
import os

import pandas as pd
import torch
from mmengine.dataset import BaseDataset

# Load model directly
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
        **kwargs,
    ):
        self.ann_file_name = ann_file_name
        if data_prefix is None:
            data_prefix = dict(
                data_path="",
                table_path="cache/processed",
                drug_description_path="drugbank_mini.csv",
            )

        if max_lengths is None:
            max_lengths = dict(
                table=1024, smiless=1024, drugs=64, disease=64, description=1024
            )
        self.max_lengths = max_lengths

        self.tokenizer = tokenizer
        super().__init__(data_prefix=data_prefix, **kwargs)

    @staticmethod
    def collate_fn(batch):
        res = {}

        for name in ["label"]:
            res[name] = torch.stack([b[name] for b in batch], dim=0)

        for name in ["table"]:
            res[name] = {}
            for k in batch[0][name]:
                res[name][k] = torch.cat([b[name][k] for b in batch], dim=0)

        for name in ["smiless", "drugs", "disease", "description"]:
            res[name] = [b[name] for b in batch]

        return res

    def load_data_list(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        data = pd.read_csv(
            os.path.join(self.data_prefix["data_path"], f"{self.ann_file_name}.csv")
        )
        table_data = json.load(
            open(
                os.path.join(
                    self.data_prefix["table_path"], f"{self.ann_file_name}.json"
                ),
            )
        )
        drug_description = pd.read_csv(self.data_prefix["drug_description_path"])
        drug_description["name"] = drug_description["name"].str.lower()
        drug_description = drug_description.set_index("name").to_dict()["description.1"]

        data_list = []
        for i, row in data.iterrows():
            if table_data[i]:
                data_list.append(
                    {
                        "label": torch.tensor(row["label"], dtype=torch.long),
                        "smiless": tokenizer(
                            eval(row["smiless"]),
                            padding="max_length",
                            max_length=self.max_lengths["smiless"],
                            return_tensors="pt",
                            return_token_type_ids=False,
                        ),
                        "table": tokenizer(
                            table_data[i],
                            padding="max_length",
                            max_length=self.max_lengths["table"],
                            return_tensors="pt",
                            return_token_type_ids=False,
                        ),
                        "drugs": tokenizer(
                            eval(row["drugs"].lower()),
                            padding="max_length",
                            max_length=self.max_lengths["drugs"],
                            return_tensors="pt",
                            return_token_type_ids=False,
                        ),
                        "disease": tokenizer(
                            eval(row["diseases"].lower()),
                            padding="max_length",
                            max_length=self.max_lengths["disease"],
                            return_tensors="pt",
                            return_token_type_ids=False,
                        ),
                        "description": tokenizer(
                            [
                                drug_description[drug_name]
                                if drug_name in drug_description
                                else "This is a drug."
                                for drug_name in eval(row["drugs"].lower())
                            ],
                            padding="max_length",
                            max_length=self.max_lengths["description"],
                            return_tensors="pt",
                            return_token_type_ids=False,
                        ),
                    }
                )

        return data_list


def main():
    HINTDataset(
        ann_file_name="phase_I_train",
        data_root="data/clinical-trial-outcome-prediction/data",
    )


if __name__ == "__main__":
    main()

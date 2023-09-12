import json
import os

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
            )

        if max_lengths is None:
            max_lengths = dict(
                table=1024,
                summarization=1024,
                smiless=1024,
                drugs=64,
                disease=64,
                description=1024,
            )
        self.max_lengths = max_lengths
        self.augment = augment

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        super().__init__(data_prefix=data_prefix, **kwargs)

    @staticmethod
    def collate_fn(batch):
        def collate(batch):
            res = {}

            for name in ["label"]:
                if name in batch[0]:
                    res[name] = torch.stack([b[name] for b in batch], dim=0)

            for name in ["table", "summarization"]:
                res[name] = {}
                for k in batch[0][name]:
                    res[name][k] = torch.cat([b[name][k] for b in batch], dim=0)

            for name in ["smiless", "drugs", "disease", "description"]:
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

        data_list = []
        for i, row in data.iterrows():
            cur_data = {}

            if "label" in row:
                cur_data["label"] = torch.tensor(row["label"], dtype=torch.long)

            for name in ["smiless", "drugs", "disease"]:
                if name in row:
                    cur_data[name] = tokenize(eval(row[name]), self.max_lengths[name])

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

            data_list.append(cur_data)

        return data_list

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if self.augment:
            data_info["augment"] = self.get_data_info(self._rand_another())
        return self.pipeline(data_info)


def main():
    HINTDataset(
        ann_file_name="phase_I_train",
        data_root="data/clinical-trial-outcome-prediction/data",
    )


if __name__ == "__main__":
    main()

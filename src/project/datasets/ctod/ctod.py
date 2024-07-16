from collections import defaultdict

from transformers import AutoTokenizer
from project.datasets.hint.hint import HINTDataset

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


class CTODataset(HINTDataset):
    """CTO dataset."""

    def __init__(
        self,
        data_prefix=None,
        ann_file_name=None,
        augment=False,
        input_parts=None,
        **kwargs,
    ):
        super().__init__(
            data_prefix=data_prefix,
            ann_file_name=ann_file_name,
            augment=augment,
            input_parts=input_parts,
            **kwargs,
        )


def main():
    # max_length = defaultdict(int)
    max_length = defaultdict(list)
    labels = {}
    for phase in ["I", "II", "III"]:
        labels[phase] = {}
        for split in ["train", "valid"]:
            # for split in ["test"]:
            labels[phase][split] = [0, 0]
            dataset = CTODataset(
                ann_file_name=f"phase_{phase}_{split}",
                data_root="data/labeling",
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

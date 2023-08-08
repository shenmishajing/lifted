from torch import nn
from transformers import AutoModelForSequenceClassification


class BioBert(nn.Module):
    def __init__(
        self,
        model="dmis-lab/biobert-base-cased-v1.1-mnli",
        num_labels=1,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=num_labels
        )

    def forward(self, data):
        output = self.model(**data["table"], labels=data["label"].float())

        return {
            "log_dict": {"loss": output.loss},
            "pred": output.logits.sigmoid().squeeze(-1),
            "target": data["label"],
        }

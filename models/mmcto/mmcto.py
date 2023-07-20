import math

import torch
from torch import nn


class MMCTO(nn.Module):
    def __init__(
        self,
        encoders: nn.Module,
        vocab_size: int = 28996,
        model_dim: int = 768,
    ):
        super().__init__()
        self.encoders = encoders

        self.vocab_size = vocab_size
        self.model_dim = model_dim

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.cls_tokens = nn.Parameter(torch.empty(5, model_dim))

        self.gate_fc = nn.Linear(2 * model_dim, 3)
        self.final_fc = nn.Linear(model_dim, 1)

        self.sigmod = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.cls_tokens)

    def add_embedding(self, input_ids, attention_mask=None, embedding_index=0):
        embedding = self.embedding(input_ids)
        cls_tokens = self.cls_tokens[embedding_index][None, None, ...].expand(
            (embedding.shape[0], -1, -1)
        )
        embedding = torch.cat([cls_tokens, embedding], dim=-2)

        _, max_len, d_model = embedding.shape

        # Compute the positional encodings once in log space.
        pos_embedding = embedding.new_zeros(max_len, d_model)
        position = torch.arange(0, max_len, device=embedding.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=embedding.device)
            * -(math.log(10000.0) / d_model)
        ).unsqueeze(0)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pos_embedding[:, 1::2] = torch.cos(position * div_term)  # 奇数下标的位置

        embedding = embedding + pos_embedding[None, ...]

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask.new_ones(attention_mask.shape[0], 1),
                    attention_mask,
                ],
                dim=-1,
            )
            attention_mask = ~attention_mask.to(torch.bool)

            return embedding, attention_mask
        else:
            return embedding

    def forward(self, data):
        datas = {}

        for embedding_index, key in enumerate(
            ["smiless", "description", "drugs", "disease"]
        ):
            datas[key] = []
            for cur_data in data[key]:
                embedding, attetion_mask = self.add_embedding(
                    **cur_data, embedding_index=embedding_index
                )
                datas[key].append(
                    self.encoders[key](embedding, src_key_padding_mask=attetion_mask)[
                        :, 0, ...
                    ].mean(dim=0)
                )
            datas[key] = torch.stack(datas[key])

        embedding, attetion_mask = self.add_embedding(
            **data["table"], embedding_index=4
        )
        datas["table"] = self.encoders["table"](
            embedding, src_key_padding_mask=attetion_mask
        )[:, 0, ...]

        gate_data = self.gate_fc(torch.cat([datas["drugs"], datas["disease"]], dim=-1))
        gate_data = torch.softmax(gate_data, dim=-1)
        gate_data = (
            gate_data[:, None]
            * torch.stack(
                [datas["table"], datas["smiless"], datas["description"]], dim=-1
            )
        ).sum(-1)

        pred = self.sigmod(self.final_fc(gate_data)).squeeze(-1)

        return {
            "loss": self.loss(pred, data["label"].float()),
            "acc": (pred.round() == data["label"]).float().mean(),
        }

import math

import torch
from torch import nn
from transformers import AutoModel


class EarlyFusion(nn.Module):
    def __init__(
        self,
        encoders: nn.Module,
        smiless_transformer_encoder="seyonec/ChemBERTa-zinc-base-v1",
        final_input_parts=None,
        gate_input_parts=None,
        vocab_size: int = 28996,
        model_dim: int = 768,
    ):
        super().__init__()
        self.encoders = encoders

        if final_input_parts is None:
            final_input_parts = ["table", "summarization", "smiless", "description"]
        if gate_input_parts is None:
            gate_input_parts = ["drugs", "diseases"]
        self.input_parts = set(final_input_parts + gate_input_parts)
        self.final_input_parts = final_input_parts
        self.gate_input_parts = gate_input_parts

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.cls_tokens = nn.Parameter(torch.empty(len(self.input_parts), model_dim))

        for key in list(self.encoders):
            if key not in self.input_parts:
                del self.encoders[key]

        self.smiless_transformer_encoder = {
            k: nn.ModuleList(
                [
                    AutoModel.from_pretrained(smiless_transformer_encoder),
                    nn.Linear(768, model_dim),
                ]
            )
            for k in [
                f"smiless_transformer{p}" for p in ["", "_concat", "_summarization"]
            ]
            if k in self.input_parts
        }

        if self.smiless_transformer_encoder:
            self.smiless_transformer_encoder = nn.ModuleDict(
                self.smiless_transformer_encoder
            )

        if "criteria" in self.input_parts:
            self.encoders["criteria"] = nn.Sequential(
                nn.Linear(768 * 2, model_dim), nn.ReLU(), nn.LayerNorm(model_dim)
            )

        self.fc = nn.Linear(model_dim, 1)

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

    def encode(self, embedding, attetion_mask, key):
        feature = self.encoders[key](embedding, src_key_padding_mask=attetion_mask)
        return {
            "logits": feature[:, 0],
            "importance_loss": feature.new_zeros([]),
            "importances": feature.new_zeros([]),
        }

    def forward(self, data):
        features = {}

        embedding_index = -1

        key = "criteria"
        if key in self.input_parts:
            features[key] = self.encoders[key](data[key])

        for key in [f"smiless_transformer_{p}" for p in ["concat", "summarization"]]:
            if key in self.input_parts:
                feature = self.smiless_transformer_encoder[key][0](**data[key])
                feature = self.smiless_transformer_encoder[key][1](
                    feature.pooler_output
                )
                features[key] = feature

        for key in ["table", "summarization"] + [
            f"{k}_{p}"
            for k in ["smiless", "drugs", "diseases"]
            for p in ["concat", "summarization"]
        ]:
            if key not in self.input_parts:
                continue
            embedding_index += 1

            embedding, attetion_mask = self.add_embedding(
                **data[key], embedding_index=embedding_index
            )
            features[key] = embedding

        for key in ["smiless", "description", "drugs", "diseases"]:
            if key not in self.input_parts:
                continue
            embedding_index += 1

            features[key] = []
            for batch_idx, cur_data in enumerate(data[key]):
                embedding, attetion_mask = self.add_embedding(
                    **cur_data, embedding_index=embedding_index
                )
                embedding = embedding.mean(dim=0, keepdim=True)
                features[key].append(embedding)

            features[key] = torch.cat(features[key])

        # encode_result = self.encode(embedding, attetion_mask, key)
        preds = self.sigmod(
            self.fc(torch.stack(list(features.values()), dim=-1).mean(-1))
        ).squeeze(-1)

        losses = {"loss": self.loss(preds, data["label"].float())}
        metrics = {
            "preds": preds,
            "target": data["label"],
        }

        return {"loss_dict": losses, "metric_dict": metrics}

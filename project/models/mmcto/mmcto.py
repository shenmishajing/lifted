import math
import random

import torch
from torch import nn


class MMCTO(nn.Module):
    def __init__(
        self,
        encoders: nn.Module,
        final_input_parts=None,
        gate_input_parts=None,
        moe_method="weighted",
        vocab_size: int = 28996,
        model_dim: int = 768,
        num_labels: int = 1,
        augment_prob=None,
        augment_eps=0.1,
    ):
        super().__init__()
        self.encoders = encoders

        if final_input_parts is None:
            final_input_parts = ["table", "summarization", "smiless", "description"]
        if gate_input_parts is None:
            gate_input_parts = ["drugs", "disease"]
        self.input_parts = final_input_parts + gate_input_parts
        self.final_input_parts = final_input_parts
        self.gate_input_parts = gate_input_parts
        self.moe_method = moe_method
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.augment_prob = {} if augment_prob is None else augment_prob
        self.augment_eps = augment_eps

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.cls_tokens = nn.Parameter(torch.empty(len(self.input_parts), model_dim))

        for key in list(self.encoders):
            if key not in self.input_parts:
                del self.encoders[key]

        if moe_method == "weighted":
            self.gate_fc = nn.Linear(
                len(self.gate_input_parts) * model_dim, len(self.final_input_parts)
            )
            self.final_fc = nn.Linear(model_dim, num_labels)
        elif moe_method == "mean":
            self.gate_fc = None
            self.final_fc = nn.Linear(model_dim, num_labels)
        elif moe_method == "concat":
            self.gate_fc = None
            self.final_fc = nn.Linear(
                model_dim * len(self.final_input_parts), num_labels
            )

        self.sigmod = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.consistency_loss = nn.L1Loss()

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
        losses = {}

        embedding_index = -1

        for key in ["table", "summarization"]:
            if key not in self.input_parts:
                continue
            embedding_index += 1

            embedding, attetion_mask = self.add_embedding(
                **data[key], embedding_index=embedding_index
            )
            feature = self.encoders[key](embedding, src_key_padding_mask=attetion_mask)[
                :, 0, ...
            ]
            datas[key] = feature

            if key in self.augment_prob:
                mask = (
                    torch.rand(
                        embedding.shape[0],
                        device=embedding.device,
                        dtype=embedding.dtype,
                    )
                    < self.augment_prob[key]
                )
                cur_embedding, _ = self.add_embedding(
                    **data["augment"][key], embedding_index=embedding_index
                )
                lamb = (
                    torch.rand(
                        embedding.shape[0],
                        device=embedding.device,
                        dtype=embedding.dtype,
                    )
                    * self.augment_eps
                )
                lamb = lamb.where(mask, torch.zeros_like(lamb))[:, None, None]
                aug_embedding = lamb * cur_embedding + (1 - lamb) * embedding
                aug_feature = self.encoders[key](
                    aug_embedding, src_key_padding_mask=attetion_mask
                )[:, 0, ...]
                losses[key] = self.consistency_loss(feature, aug_feature)

        for key in ["smiless", "description", "drugs", "disease"]:
            if key not in self.input_parts:
                continue
            embedding_index += 1

            datas[key] = []
            losses[key] = []
            for batch_idx, cur_data in enumerate(data[key]):
                embedding, attetion_mask = self.add_embedding(
                    **cur_data, embedding_index=embedding_index
                )
                feature = self.encoders[key](
                    embedding, src_key_padding_mask=attetion_mask
                )[:, 0, ...].mean(dim=0)
                datas[key].append(feature)

                if key in self.augment_prob:
                    aug_embedding = embedding.clone()
                    for idx in range(embedding.shape[0]):
                        if torch.rand(1) < self.augment_prob[key]:
                            cur_embedding, _ = self.add_embedding(
                                **data["augment"][key][batch_idx],
                                embedding_index=embedding_index,
                            )
                            aug_idx = random.choice(range(cur_embedding.shape[0]))
                            lamb = (
                                torch.rand(
                                    1, device=embedding.device, dtype=embedding.dtype
                                )
                                * self.augment_eps
                            )
                            aug_embedding[idx] = (
                                lamb * cur_embedding[aug_idx]
                                + (1 - lamb) * embedding[idx]
                            )
                    aug_feature = self.encoders[key](
                        aug_embedding, src_key_padding_mask=attetion_mask
                    )[:, 0, ...].mean(dim=0)
                    losses[key].append(self.consistency_loss(feature, aug_feature))
            datas[key] = torch.stack(datas[key])
            if losses[key]:
                losses[key] = torch.stack(losses[key]).mean()
            else:
                del losses[key]

        if self.moe_method == "weighted":
            gate_data = self.gate_fc(
                torch.cat([datas[p] for p in self.gate_input_parts], dim=-1)
            )
            gate_data = torch.softmax(gate_data, dim=-1)
            gate_data = (
                gate_data[:, None]
                * torch.stack([datas[p] for p in self.final_input_parts], dim=-1)
            ).sum(-1)
        elif self.moe_method == "mean":
            gate_data = torch.stack(
                [datas[p] for p in self.final_input_parts], dim=-1
            ).mean(dim=-1)
        elif self.moe_method == "concat":
            gate_data = torch.cat([datas[p] for p in self.final_input_parts], dim=-1)

        pred = self.sigmod(self.final_fc(gate_data)).squeeze(-1)

        losses = {f"{k}_consistency_loss": v for k, v in losses.items()}
        losses["classification_loss"] = self.loss(pred, data["label"].float())

        return {
            "loss_dict": losses,
            "metric_dict": {"preds": pred, "target": data["label"]},
        }

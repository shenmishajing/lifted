import math

import torch
import torch.nn.functional as F
from torch import nn


class MMCTO(nn.Module):
    def __init__(
        self,
        encoders: nn.Module,
        final_input_parts=None,
        gate_input_parts=None,
        aux_loss_parts=None,
        aux_loss_share_fc=False,
        weighted_aux_loss=False,
        moe_method="weighted",
        pretrain=False,
        vocab_size: int = 28996,
        model_dim: int = 768,
        num_labels: int = 1,
        augment_prob=0.0,
        augment_eps=0.1,
        piror_init=0,
        multiply_disturb=False,
        contrastive_loss=False,
        inverse_consistency_loss=False,
        use_cosin_simiarity_loss=False,
    ):
        super().__init__()
        self.encoders = encoders

        if final_input_parts is None:
            final_input_parts = ["table", "summarization", "smiless", "description"]
        if gate_input_parts is None:
            gate_input_parts = ["drugs", "diseases"]
        if aux_loss_parts is None:
            aux_loss_parts = []
        else:
            aux_loss_parts = [p for p in aux_loss_parts if p in final_input_parts]
        self.input_parts = set(final_input_parts + gate_input_parts)
        self.final_input_parts = final_input_parts
        self.gate_input_parts = gate_input_parts
        self.aux_loss_share_fc = aux_loss_share_fc
        self.weighted_aux_loss = weighted_aux_loss
        self.moe_method = moe_method
        self.pretrain = pretrain
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.augment_prob = augment_prob
        self.augment_eps = augment_eps
        self.contrastive_loss = contrastive_loss
        self.inverse_consistency_loss = inverse_consistency_loss
        self.use_cosin_simiarity_loss = use_cosin_simiarity_loss

        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.cls_tokens = nn.Parameter(torch.empty(len(self.input_parts), model_dim))
        self.piror = nn.Parameter(torch.empty(len(self.final_input_parts)))
        self.piror_init = piror_init
        self.multiply_disturb = multiply_disturb

        for key in list(self.encoders):
            if key not in self.input_parts:
                del self.encoders[key]

        if "criteria" in self.input_parts:
            self.encoders["criteria"] = nn.Sequential(
                nn.Linear(768 * 2, model_dim), nn.ReLU(), nn.LayerNorm(model_dim)
            )

        if aux_loss_share_fc:
            aux_loss_fc = nn.Linear(model_dim, num_labels)
            self.aux_loss_fc = nn.ModuleDict(
                {part: aux_loss_fc for part in aux_loss_parts}
            )
        else:
            self.aux_loss_fc = nn.ModuleDict(
                {part: nn.Linear(model_dim, num_labels) for part in aux_loss_parts}
            )
        if not self.pretrain:
            if moe_method == "weighted":
                self.gate_fc = nn.Linear(
                    len(self.gate_input_parts) * model_dim, len(self.final_input_parts)
                )
            else:
                self.gate_fc = None

            if moe_method == "concat":
                self.final_fc = nn.Linear(
                    model_dim * len(self.final_input_parts), num_labels
                )
            else:
                self.final_fc = nn.Linear(model_dim, num_labels)

        self.sigmod = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.aux_loss = nn.BCELoss(reduction="none")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.cls_tokens)
        nn.init.normal_(self.piror, self.piror_init, 0.1)

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

    def similarity(self, x, y):
        if self.use_cosin_simiarity_loss:
            return (
                1
                + (F.normalize(x.flatten(1)) * F.normalize(y.flatten(1)))
                .sum(dim=-1)
                .mean()
            ) / 2
        else:
            return F.l1_loss(x, y)

    def generate_random_lambda(self, embedding):
        lamb = (
            torch.rand(
                embedding.shape,
                device=embedding.device,
                dtype=embedding.dtype,
            )
            * self.augment_eps
        )
        lamb = lamb.where(
            torch.rand(
                embedding.shape,
                device=embedding.device,
                dtype=embedding.dtype,
            )
            < self.augment_prob,
            torch.zeros_like(lamb),
        )
        return lamb

    def disturb_embedding(self, embedding, lamb, disturb_embedding=None):
        if self.multiply_disturb:
            disturb_embedding = (
                torch.rand(
                    embedding.shape,
                    device=embedding.device,
                    dtype=embedding.dtype,
                )
                * 2
                - 1
            ).exp()
            disturb_embedding = disturb_embedding.where(
                lamb, torch.ones_like(disturb_embedding)
            )
            return embedding * disturb_embedding
        else:
            return (1 - lamb) * embedding + lamb * disturb_embedding

    def encode(self, embedding, attetion_mask, key):
        return self.encoders[key](embedding, src_key_padding_mask=attetion_mask)[
            :, 0, ...
        ]

    def calculate_consistency_and_contrastive_loss(
        self,
        key,
        feature,
        disturbed_feature,
        aug_feature=None,
        aug_disturbed_feature=None,
    ):
        losses = {}
        losses[f"{key}_consistency_loss"] = self.similarity(feature, disturbed_feature)

        if self.contrastive_loss:
            losses[f"{key}_contrastive_loss"] = (
                4
                - self.similarity(feature, aug_feature)
                - self.similarity(feature, aug_disturbed_feature)
                - self.similarity(disturbed_feature, aug_feature)
                - self.similarity(disturbed_feature, aug_disturbed_feature)
            )

        if self.inverse_consistency_loss:
            losses[f"{key}_inverse_consistency_loss"] = self.similarity(
                aug_feature, aug_disturbed_feature
            )
        return losses

    def forward(self, data):
        datas = {}
        losses = {}

        embedding_index = -1

        if "criteria" in self.input_parts:
            key = "criteria"
            feature = self.encoders[key](data[key])
            datas[key] = feature

            if self.augment_prob > 0:
                lamb = self.generate_random_lambda(data[key])

                if self.contrastive_loss or self.inverse_consistency_loss:
                    aug_feature = self.encoders[key](data["augment"][key])
                    aug_disturbed_feature = self.encoders[key](
                        self.disturb_embedding(data["augment"][key], lamb, data[key])
                    )
                else:
                    aug_feature = None
                    aug_disturbed_feature = None

                disturbed_feature = self.encoders[key](
                    self.disturb_embedding(data[key], lamb, data["augment"][key])
                )
                losses.update(
                    self.calculate_consistency_and_contrastive_loss(
                        key,
                        feature,
                        disturbed_feature,
                        aug_feature,
                        aug_disturbed_feature,
                    )
                )

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
            feature = self.encode(embedding, attetion_mask, key)
            datas[key] = feature

            if self.augment_prob > 0:
                lamb = self.generate_random_lambda(embedding)

                aug_embedding, aug_attetion_mask = self.add_embedding(
                    **data["augment"][key], embedding_index=embedding_index
                )
                if self.contrastive_loss or self.inverse_consistency_loss:
                    aug_feature = self.encode(aug_embedding, aug_attetion_mask, key)
                    aug_disturbed_feature = self.encode(
                        self.disturb_embedding(aug_embedding, lamb, embedding),
                        aug_attetion_mask,
                        key,
                    )
                else:
                    aug_feature = None
                    aug_disturbed_feature = None

                del aug_attetion_mask

                disturbed_feature = self.encode(
                    self.disturb_embedding(embedding, lamb, aug_embedding),
                    attetion_mask,
                    key,
                )
                del aug_embedding, embedding, attetion_mask, lamb

                losses.update(
                    self.calculate_consistency_and_contrastive_loss(
                        key,
                        feature,
                        disturbed_feature,
                        aug_feature,
                        aug_disturbed_feature,
                    )
                )

        for key in ["smiless", "description", "drugs", "diseases"]:
            if key not in self.input_parts:
                continue
            embedding_index += 1

            datas[key] = []
            losses[f"{key}_contrastive_loss"] = []
            losses[f"{key}_consistency_loss"] = []
            losses[f"{key}_inverse_consistency_loss"] = []
            for batch_idx, cur_data in enumerate(data[key]):
                embedding, attetion_mask = self.add_embedding(
                    **cur_data, embedding_index=embedding_index
                )
                feature = self.encode(embedding, attetion_mask, key).mean(dim=0)[None]
                datas[key].append(feature)

                if self.augment_prob > 0:
                    lamb = self.generate_random_lambda(embedding)

                    aug_embedding, aug_attetion_mask = self.add_embedding(
                        **data["augment"][key][batch_idx],
                        embedding_index=embedding_index,
                    )
                    aug_idx = torch.randint(
                        0, aug_embedding.shape[0], (embedding.shape[0],)
                    )
                    aug_embedding = aug_embedding[aug_idx]

                    if self.contrastive_loss or self.inverse_consistency_loss:
                        aug_attetion_mask = aug_attetion_mask[aug_idx]
                        aug_feature = self.encode(
                            aug_embedding, aug_attetion_mask, key
                        ).mean(dim=0)[None]
                        aug_disturbed_feature = self.encode(
                            self.disturb_embedding(aug_embedding, lamb, embedding),
                            aug_attetion_mask,
                            key,
                        ).mean(dim=0)[None]
                    else:
                        del aug_attetion_mask

                    disturbed_feature = self.encode(
                        self.disturb_embedding(embedding, lamb, aug_embedding),
                        attetion_mask,
                        key,
                    ).mean(dim=0)[None]
                    del aug_embedding, embedding, attetion_mask, lamb

                    for k, v in self.calculate_consistency_and_contrastive_loss(
                        key,
                        feature,
                        disturbed_feature,
                        aug_feature,
                        aug_disturbed_feature,
                    ).items():
                        losses[k].append(v)

            datas[key] = torch.cat(datas[key])

            for k in [
                "contrastive_loss",
                "consistency_loss",
                "inverse_consistency_loss",
            ]:
                if losses[f"{key}_{k}"]:
                    losses[f"{key}_{k}"] = torch.stack(losses[f"{key}_{k}"]).mean()
                else:
                    del losses[f"{key}_{k}"]

        if not self.pretrain:
            aux_losses = {
                part: self.aux_loss(
                    self.sigmod(fc(datas[part])).squeeze(-1), data["label"].float()
                )
                for part, fc in self.aux_loss_fc.items()
            }
            hidden_states = {
                "input_parts": self.final_input_parts,
                "aux_losses": aux_losses,
            }
            if self.weighted_aux_loss:
                if self.moe_method != "weighted":
                    aux_losses = {
                        k: aux_losses[k].mean() / len(self.aux_loss_fc)
                        for k in self.aux_loss_fc
                    }
            else:
                aux_losses = {k: aux_losses[k].mean() for k in self.aux_loss_fc}

            if self.moe_method == "weighted":
                gate_data = (
                    self.gate_fc(
                        torch.cat([datas[p] for p in self.gate_input_parts], dim=-1)
                    )
                    * self.piror[None]
                )
                gate_data = torch.softmax(gate_data, dim=-1)
                hidden_states["piror"] = self.piror
                hidden_states["moe_weights"] = gate_data
                if self.weighted_aux_loss:
                    aux_losses = {
                        k: (aux_losses[k] * gate_data[..., idx]).mean()
                        for idx, k in enumerate(self.final_input_parts)
                        if k in aux_losses
                    }
                gate_data = (
                    gate_data[:, None]
                    * torch.stack([datas[p] for p in self.final_input_parts], dim=-1)
                ).sum(-1)
            elif self.moe_method == "mean":
                gate_data = torch.stack(
                    [datas[p] for p in self.final_input_parts], dim=-1
                ).mean(dim=-1)
            elif self.moe_method == "concat":
                gate_data = torch.cat(
                    [datas[p] for p in self.final_input_parts], dim=-1
                )

            losses.update({f"{k}_aux_loss": v for k, v in aux_losses.items()})

            pred = self.sigmod(self.final_fc(gate_data)).squeeze(-1)

            losses["classification_loss"] = self.loss(pred, data["label"].float())
            metrics = {"preds": pred, "target": data["label"]}
        else:
            metrics = {}
            hidden_states = {}

        return {
            "loss_dict": losses,
            "metric_dict": metrics,
            "hidden_state_dict": hidden_states,
        }

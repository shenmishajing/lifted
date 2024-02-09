import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class MMCTO(nn.Module):
    def __init__(
        self,
        encoders: nn.Module,
        smoe_encoder: nn.Module = None,
        smiless_transformer_encoder="seyonec/ChemBERTa-zinc-base-v1",
        final_input_parts=None,
        gate_input_parts=None,
        aux_loss=False,
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

        self.smoe_encoder = smoe_encoder

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

        if aux_loss:
            if aux_loss_share_fc:
                aux_loss_fc = nn.Linear(model_dim, num_labels)
                self.aux_loss_fc = nn.ModuleDict(
                    {part: aux_loss_fc for part in final_input_parts}
                )
            else:
                self.aux_loss_fc = nn.ModuleDict(
                    {
                        part: nn.Linear(model_dim, num_labels)
                        for part in final_input_parts
                    }
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
        feature = self.encoders[key](embedding, src_key_padding_mask=attetion_mask)
        if self.smoe_encoder is not None:
            return self.smoe_encoder(feature[:, 0])
        return {
            "logits": feature[:, 0],
            "importance_loss": feature.new_zeros([]),
            "importances": feature.new_zeros([]),
        }

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
        hidden_states = {
            "input_parts": self.final_input_parts,
            "smoe_weights": {},
        }
        features = {}
        losses = {}

        embedding_index = -1

        key = "criteria"
        if key in self.input_parts:
            feature = self.encoders[key](data[key])
            if self.smoe_encoder is not None:
                feature = self.smoe_encoder(feature)
                losses[f"{key}_importance_loss"] = feature["importance_loss"]
                hidden_states["smoe_weights"][key] = feature["importances"]
                feature = feature["logits"]
            features[key] = feature

            if self.augment_prob > 0:
                lamb = self.generate_random_lambda(data[key])

                if self.contrastive_loss or self.inverse_consistency_loss:
                    aug_feature = self.encoders[key](data["augment"][key])
                    aug_disturbed_feature = self.encoders[key](
                        self.disturb_embedding(data["augment"][key], lamb, data[key])
                    )
                    if self.smoe_encoder is not None:
                        aug_feature = self.smoe_encoder(aug_feature)
                        losses[f"{key}_aug_importance_loss"] = aug_feature[
                            "importance_loss"
                        ]
                        hidden_states["smoe_weights"][f"{key}_aug"] = aug_feature[
                            "importances"
                        ]
                        aug_feature = aug_feature["logits"]
                        aug_disturbed_feature = self.smoe_encoder(aug_disturbed_feature)
                        losses[
                            f"{key}_aug_disturbed_importance_loss"
                        ] = aug_disturbed_feature["importance_loss"]
                        hidden_states["smoe_weights"][
                            f"{key}_aug_disturbed"
                        ] = aug_disturbed_feature["importances"]
                        aug_disturbed_feature = aug_disturbed_feature["logits"]

                else:
                    aug_feature = None
                    aug_disturbed_feature = None

                disturbed_feature = self.encoders[key](
                    self.disturb_embedding(data[key], lamb, data["augment"][key])
                )
                if self.smoe_encoder is not None:
                    disturbed_feature = self.smoe_encoder(disturbed_feature)
                    losses[f"{key}_disturbed_importance_loss"] = disturbed_feature[
                        "importance_loss"
                    ]
                    hidden_states["smoe_weights"][
                        f"{key}_disturbed"
                    ] = disturbed_feature["importances"]
                    disturbed_feature = disturbed_feature["logits"]
                losses.update(
                    self.calculate_consistency_and_contrastive_loss(
                        key,
                        feature,
                        disturbed_feature,
                        aug_feature,
                        aug_disturbed_feature,
                    )
                )

        for key in [f"smiless_transformer_{p}" for p in ["concat", "summarization"]]:
            if key in self.input_parts:
                feature = self.smiless_transformer_encoder[key][0](**data[key])
                feature = self.smiless_transformer_encoder[key][1](
                    feature.pooler_output
                )
                if self.smoe_encoder is not None:
                    feature = self.smoe_encoder(feature)
                    losses[f"{key}_importance_loss"] = feature["importance_loss"]
                    hidden_states["smoe_weights"][key] = feature["importances"]
                    feature = feature["logits"]
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
            encode_result = self.encode(embedding, attetion_mask, key)
            losses[f"{key}_importance_loss"] = encode_result["importance_loss"]
            features[key] = encode_result["logits"]
            hidden_states["smoe_weights"][key] = encode_result["importances"]

            if self.augment_prob > 0:
                lamb = self.generate_random_lambda(embedding)

                aug_embedding, aug_attetion_mask = self.add_embedding(
                    **data["augment"][key], embedding_index=embedding_index
                )
                if self.contrastive_loss or self.inverse_consistency_loss:
                    aug_encode_result = self.encode(
                        aug_embedding, aug_attetion_mask, key
                    )
                    losses[f"{key}_aug_importance_loss"] = aug_encode_result[
                        "importance_loss"
                    ]
                    hidden_states["smoe_weights"][f"{key}_aug"] = aug_encode_result[
                        "importances"
                    ]
                    aug_disturbed_encode_result = self.encode(
                        self.disturb_embedding(aug_embedding, lamb, embedding),
                        aug_attetion_mask,
                        key,
                    )
                    losses[
                        f"{key}_aug_disturbed_importance_loss"
                    ] = aug_disturbed_encode_result["importance_loss"]
                    hidden_states["smoe_weights"][
                        f"{key}_aug_disturbed"
                    ] = aug_disturbed_encode_result["importances"]
                else:
                    aug_encode_result = defaultdict(lambda: None)
                    aug_disturbed_encode_result = defaultdict(lambda: None)

                del aug_attetion_mask

                disturbed_encode_result = self.encode(
                    self.disturb_embedding(embedding, lamb, aug_embedding),
                    attetion_mask,
                    key,
                )
                losses[f"{key}_disturbed_importance_loss"] = disturbed_encode_result[
                    "importance_loss"
                ]
                hidden_states["smoe_weights"][
                    f"{key}_disturbed"
                ] = disturbed_encode_result["importances"]
                del aug_embedding, embedding, attetion_mask, lamb

                losses.update(
                    self.calculate_consistency_and_contrastive_loss(
                        key,
                        encode_result["logits"],
                        disturbed_encode_result["logits"],
                        aug_encode_result["logits"],
                        aug_disturbed_encode_result["logits"],
                    )
                )

        for key in ["smiless", "description", "drugs", "diseases"]:
            if key not in self.input_parts:
                continue
            embedding_index += 1

            features[key] = []
            cur_losses = defaultdict(list)
            cur_importances = defaultdict(list)
            for batch_idx, cur_data in enumerate(data[key]):
                embedding, attetion_mask = self.add_embedding(
                    **cur_data, embedding_index=embedding_index
                )
                encode_result = self.encode(embedding, attetion_mask, key)
                encode_result["logits"] = encode_result["logits"].mean(
                    dim=0, keepdim=True
                )
                features[key].append(encode_result["logits"])
                cur_losses[f"{key}_importance_loss"].append(
                    encode_result["importance_loss"]
                )
                cur_importances[key].append(encode_result["importances"])

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
                        aug_encode_result = self.encode(
                            aug_embedding, aug_attetion_mask, key
                        )
                        aug_encode_result["logits"] = aug_encode_result["logits"].mean(
                            dim=0, keepdim=True
                        )
                        cur_losses[f"{key}_aug_importance_loss"].append(
                            aug_encode_result["importance_loss"]
                        )
                        cur_importances[f"{key}_aug"].append(
                            aug_encode_result["importances"]
                        )

                        aug_disturbed_encode_result = self.encode(
                            self.disturb_embedding(aug_embedding, lamb, embedding),
                            aug_attetion_mask,
                            key,
                        )
                        aug_disturbed_encode_result[
                            "logits"
                        ] = aug_disturbed_encode_result["logits"].mean(
                            dim=0, keepdim=True
                        )
                        cur_losses[f"{key}_aug_disturbed_importance_loss"].append(
                            aug_disturbed_encode_result["importance_loss"]
                        )
                        cur_importances[f"{key}_aug_disturbed"].append(
                            aug_disturbed_encode_result["importances"]
                        )
                    else:
                        del aug_attetion_mask

                    disturbed_encode_result = self.encode(
                        self.disturb_embedding(embedding, lamb, aug_embedding),
                        attetion_mask,
                        key,
                    )
                    disturbed_encode_result["logits"] = disturbed_encode_result[
                        "logits"
                    ].mean(dim=0, keepdim=True)
                    cur_losses[f"{key}_disturbed_importance_loss"].append(
                        disturbed_encode_result["importance_loss"]
                    )
                    cur_importances[f"{key}_disturbed"].append(
                        disturbed_encode_result["importances"]
                    )
                    del aug_embedding, embedding, attetion_mask, lamb

                    for k, v in self.calculate_consistency_and_contrastive_loss(
                        key,
                        encode_result["logits"],
                        disturbed_encode_result["logits"],
                        aug_encode_result["logits"],
                        aug_disturbed_encode_result["logits"],
                    ).items():
                        cur_losses[k].append(v)

            features[key] = torch.cat(features[key])

            for k, v in cur_losses.items():
                if v:
                    cur_losses[k] = torch.stack(v).mean()
                else:
                    del cur_losses[k]
            losses.update(cur_losses)
            hidden_states["smoe_weights"].update(cur_importances)

        if not self.pretrain:
            aux_losses = {
                part: self.aux_loss(
                    self.sigmod(fc(features[part])).squeeze(-1), data["label"].float()
                )
                for part, fc in self.aux_loss_fc.items()
            }
            hidden_states["aux_losses"] = aux_losses
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
                        torch.cat([features[p] for p in self.gate_input_parts], dim=-1)
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
                    * torch.stack([features[p] for p in self.final_input_parts], dim=-1)
                ).sum(-1)
            elif self.moe_method == "mean":
                gate_data = torch.stack(
                    [features[p] for p in self.final_input_parts], dim=-1
                ).mean(dim=-1)
            elif self.moe_method == "concat":
                gate_data = torch.cat(
                    [features[p] for p in self.final_input_parts], dim=-1
                )

            losses.update({f"{k}_aux_loss": v for k, v in aux_losses.items()})

            pred = self.sigmod(self.final_fc(gate_data)).squeeze(-1)

            losses["classification_loss"] = self.loss(pred, data["label"].float())
            metrics = {"preds": pred, "target": data["label"]}
        else:
            metrics = {}

        return {
            "loss_dict": losses,
            "metric_dict": metrics,
            "hidden_state_dict": hidden_states,
        }

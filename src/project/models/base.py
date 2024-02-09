import os
import pickle
from collections import defaultdict

import pandas as pd
import seaborn as sns
import torch
from lightning_template import LightningModule as _LightningModule
from matplotlib import pyplot as plt


class LightningModule(_LightningModule):
    def __init__(self, bootstrap_num=20, predict_tasks=None, *args, **kwargs) -> None:
        if predict_tasks is None:
            predict_tasks = ["hidden_state"]
        super().__init__(predict_tasks=predict_tasks, *args, **kwargs)
        self.bootstrap_num = bootstrap_num

    def update_evaluator(self, evaluator, *args, metrics, **kwargs):
        if "results" not in evaluator:
            evaluator["results"] = {}
        for key, value in metrics.items():
            if key not in evaluator["results"]:
                evaluator["results"][key] = []
            evaluator["results"][key].append(value)

    def _compute_evaluator(self, evaluator, *args, **kwargs):
        for key, value in evaluator["results"].items():
            evaluator["results"][key] = torch.cat(value, dim=0)

        results = []
        if self.bootstrap_num > 1:
            for _ in range(self.bootstrap_num):
                cur_result = {}
                idx = torch.randint_like(
                    evaluator["results"]["target"],
                    0,
                    len(evaluator["results"]["target"]),
                )
                for key, value in evaluator["results"].items():
                    cur_result[key] = evaluator["results"][key][idx]

                evaluator["metric"].update(**cur_result)
                results.append(evaluator["metric"].compute())
                evaluator["metric"].reset()
        else:
            evaluator["metric"].update(**evaluator["results"])
            results.append(evaluator["metric"].compute())
            evaluator["metric"].reset()

        evaluator.pop("results", None)

        result = {}
        if len(results) == 1:
            result = results[0]
        elif len(results) > 1:
            for key, value in results[0].items():
                data = torch.stack([r[key] for r in results], dim=0).to(torch.float32)
                if "statscores" in key:
                    result[f"{key}_min"] = data.min()
                    result[f"{key}_max"] = data.max()
                result[f"{key}_mean"] = data.mean()
                result[f"{key}_std"] = data.std()

        return result

    def on_predict_epoch_start(self) -> None:
        super().on_predict_epoch_start()
        self.hidden_states = defaultdict(list)

    def predict_forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def predict_hidden_state(
        self, batch, *args, hidden_state_dict, metric_dict, **kwargs
    ):
        self.hidden_states["input_parts"] = hidden_state_dict["input_parts"]
        self.hidden_states["piror"] = hidden_state_dict["piror"].cpu()
        self.hidden_states["moe_weights"].append(hidden_state_dict["moe_weights"].cpu())
        self.hidden_states["preds"].append(metric_dict["preds"].cpu())
        self.hidden_states["target"].append(metric_dict["target"].cpu())
        self.hidden_states["error"].append(
            (metric_dict["preds"] - metric_dict["target"]).abs().cpu()
        )
        self.hidden_states["idx"].extend(batch["idx"].cpu())
        for k, v in hidden_state_dict["smoe_weights"].items():
            if isinstance(v, list):
                v = torch.stack([item[0] for item in v])
            hidden_state_dict["smoe_weights"][k] = v.cpu()

        self.hidden_states["smoe_weights"].extend(
            [
                {k: v[i] for k, v in hidden_state_dict["smoe_weights"].items()}
                for i in range(len(batch["idx"]))
            ]
        )

    def on_predict_epoch_end(self) -> None:
        for key in self.hidden_states:
            if key in ["idx", "input_parts", "piror", "smoe_weights"]:
                continue
            self.hidden_states[key] = torch.cat(self.hidden_states[key], dim=0)

        pickle.dump(
            self.hidden_states,
            open(
                os.path.join(self.predict_path, "hidden_state", "all.pkl"),
                "wb",
            ),
        )

        idx = self.hidden_states["error"].topk(10, largest=False, dim=0).indices
        for key in self.hidden_states:
            if key in ["input_parts", "piror"]:
                continue
            elif key in ["idx", "smoe_weights"]:
                self.hidden_states[key] = [self.hidden_states[key][i] for i in idx]
            else:
                self.hidden_states[key] = self.hidden_states[key][idx]

        pickle.dump(
            self.hidden_states,
            open(
                os.path.join(self.predict_path, "hidden_state", "top10.pkl"),
                "wb",
            ),
        )

        data = pd.read_csv(
            os.path.join(
                self.trainer.datamodule.dataset.data_prefix["data_path"],
                f"{self.trainer.datamodule.dataset.ann_file_name}.csv",
            )
        ).iloc[self.hidden_states["idx"]]

        data.to_csv(
            os.path.join(self.predict_path, "hidden_state", "top10.csv"),
            index=False,
        )

        smoe_weight_output_path = os.path.join(
            self.predict_path, "hidden_state", "smoe_weights"
        )
        os.makedirs(smoe_weight_output_path, exist_ok=True)

        moe_weight_output_path = os.path.join(
            self.predict_path, "hidden_state", "moe_weights"
        )
        os.makedirs(moe_weight_output_path, exist_ok=True)

        def plot_moe_piror(data, labels, output_path):
            plt.bar_label(
                plt.bar(range(len(data)), data, tick_label=labels),
                label_type="edge",
            )
            plt.savefig(output_path)
            plt.clf()

        def plot_moe_weights(data, labels, output_path):
            plt.pie(data, labels=labels, autopct="%1.1f%%")
            plt.axis("equal")
            plt.savefig(output_path)
            plt.clf()

        def plot_smoe_weights(data, output_path):
            df_smoe_weights = pd.DataFrame(data) * 100
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_smoe_weights, annot=True, fmt=".1f", cmap="YlGnBu")
            plt.ylabel("Expert")
            plt.xlabel("Modality")
            plt.savefig(output_path)
            plt.clf()

        plot_moe_piror(
            self.hidden_states["piror"],
            self.hidden_states["input_parts"],
            os.path.join(moe_weight_output_path, "piror.png"),
        )

        for i, moe_weights in enumerate(self.hidden_states["moe_weights"]):
            plot_moe_weights(
                moe_weights,
                self.hidden_states["input_parts"],
                os.path.join(moe_weight_output_path, f"{i}.png"),
            )

        for i, smoe_weights in enumerate(self.hidden_states["smoe_weights"]):
            keys = [
                "summarization",
                "drugs",
                "diseases",
                "criteria",
                "smiless_transformer_concat",
                "description",
            ]
            smoe_weights = {k: smoe_weights[k] for k in keys}
            if "smiless_transformer_concat" in smoe_weights:
                smoe_weights["smiless"] = smoe_weights.pop("smiless_transformer_concat")
            plot_smoe_weights(
                smoe_weights, os.path.join(smoe_weight_output_path, f"{i}.png")
            )

import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
from lightning_template import LightningModule as _LightningModule


class LightningModule(_LightningModule):
    def __init__(self, bootstrap_num=20, predict_tasks=None, *args, **kwargs) -> None:
        if predict_tasks is None:
            predict_tasks = ["hidden_state"]
        super().__init__(predict_tasks=predict_tasks, *args, **kwargs)
        self.bootstrap_num = bootstrap_num

    def update_evaluator(self, evaluator, *args, **kwargs):
        if "results" not in evaluator:
            evaluator["results"] = {}
        for key, value in kwargs.items():
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
        if self.trainer.ckpt_path:
            output_path = os.path.join(
                "results",
                "visualization",
                os.path.basename(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(self.trainer.ckpt_path))
                    )
                ),
            )
        else:
            output_path = None

        for name in self.predict_tasks:
            if self.predict_tasks[name] is None:
                if output_path is None:
                    raise ValueError(
                        "predict_path is None, please set predict_path or pass ckpt_path"
                    )

                self.predict_tasks[name] = output_path

            self.predict_tasks[name] = os.path.join(self.predict_tasks[name], name)
            self.rm_and_create(self.predict_tasks[name])

        self.hidden_states = defaultdict(list)

    def predict_forward(self, *args, **kwargs):
        return self(*args, **kwargs)

    def predict_hidden_state(
        self, batch, *args, hidden_state_dict, metric_dict, **kwargs
    ):
        self.hidden_states["input_parts"] = hidden_state_dict["input_parts"]
        self.hidden_states["moe_weights"].append(hidden_state_dict["moe_weights"].cpu())
        self.hidden_states["preds"].append(metric_dict["preds"].cpu())
        self.hidden_states["target"].append(metric_dict["target"].cpu())
        self.hidden_states["error"].append(
            (metric_dict["preds"] - metric_dict["target"]).abs().cpu()
        )
        self.hidden_states["idx"].extend(batch["idx"])

    def on_predict_epoch_end(self) -> None:
        for key in self.hidden_states:
            if key in ["idx", "input_parts"]:
                continue
            self.hidden_states[key] = torch.cat(self.hidden_states[key], dim=0)

        pickle.dump(
            self.hidden_states,
            open(
                os.path.join(self.predict_tasks["hidden_state"], "all.pkl"),
                "wb",
            ),
        )

        idx = self.hidden_states["error"].topk(10, dim=0).indices
        for key in self.hidden_states:
            if key == "idx":
                self.hidden_states[key] = [self.hidden_states[key][i] for i in idx]
            else:
                self.hidden_states[key] = self.hidden_states[key][idx]

        pickle.dump(
            self.hidden_states,
            open(
                os.path.join(self.predict_tasks["hidden_state"], "top10.pkl"),
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
            os.path.join(self.predict_tasks["hidden_state"], "top10.csv"),
            index=False,
        )

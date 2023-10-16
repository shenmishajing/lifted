import torch
from lightning_template import LightningModule as _LightningModule


class LightningModule(_LightningModule):
    def __init__(self, bootstrap_num=20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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

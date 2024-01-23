# load demo data
import json
import os
import pickle
import random
import shutil
from collections import defaultdict

import numpy as np
import torch
from lightning import seed_everything
from pytrial.data.demo_data import load_trial_outcome_data

# prepare the input data class for this task
from pytrial.tasks.trial_outcome import HINT, SPOT
from pytrial.tasks.trial_outcome.data import (
    TrialOutcomeDataset,
    TrialOutcomeDatasetBase,
)
from torchmetrics import AUROC, AveragePrecision, F1Score, MetricCollection, StatScores
from tqdm import trange


def get_random_seed():
    return random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)


def parse_results(data):
    result = {}
    result["seed"] = [d["seed"] for d in data]
    for split in ["valid", "test"]:
        result[split] = {}
        for metric in ["PR-AUC", "F1", "ROC-AUC"]:
            result[split][metric] = torch.stack([d[split][metric] for d in data])
            result[split][metric] = [
                result[split][metric].mean().item(),
                result[split][metric].std().item(),
            ]

    return result


def bootstrap_test(preds, target, metrics, bootstrap_num=20):
    results = []
    for _ in range(bootstrap_num):
        cur_result = {}
        idx = torch.randint_like(target, 0, len(target))
        cur_result["preds"] = preds[idx]
        cur_result["target"] = target[idx]

        metrics.update(**cur_result)
        results.append(metrics.compute())
        metrics.reset()

    result = {}
    if len(results) == 1:
        result = results[0]
    elif len(results) > 1:
        for key in results[0]:
            data = torch.stack([r[key] for r in results], dim=0).to(torch.float32)
            if "statscores" in key:
                result[f"{key}_min"] = data.min()
                result[f"{key}_max"] = data.max()
            result[f"{key}_mean"] = data.mean()
            result[f"{key}_std"] = data.std()
        for key in result:
            result[key] = result[key].item()
    return result


def fit_modal(
    model_name,
    phase,
    no_bootstrap_test,
    n=30,
    world_size=1,
    rank=0,
    output_path="results/details",
    *args,
    **kwargs,
):
    output_path = os.path.join(output_path, phase, model_name)
    os.makedirs(output_path, exist_ok=True)

    for i in trange(rank, n, world_size):
        seed = get_random_seed()
        seed_everything(seed)

        result = model_map[model_name](
            *args,
            **kwargs,
            no_bootstrap_test=no_bootstrap_test,
            seed=seed,
            model_name=model_name,
            output_path=os.path.join(output_path, str(i)),
        )
        result["seed"] = seed
        if no_bootstrap_test:
            pickle.dump(result, open(os.path.join(output_path, f"{i}.pkl"), "wb"))
        else:
            json.dump(result, open(os.path.join(output_path, f"{i}.json"), "w"))


def hint(datasets, metrics, data, no_bootstrap_test, *args, **kwargs):
    model = HINT(highway_num_layer=2, epoch=5, lr=1e-3)
    model.fit(datasets["train"], datasets["valid"])

    result = {}
    for split in ["valid", "test"]:
        preds = model.predict(datasets[split])
        target = data[split][["nctid", "label"]].set_index("nctid").to_dict("split")
        target = {k: v[0] for k, v in zip(target["index"], target["data"])}
        target = torch.tensor([target[k[0]] for k in preds])
        preds = torch.tensor([k[1] for k in preds])
        if no_bootstrap_test:
            result[split] = metrics(preds, target)
            metrics.reset()
        else:
            result[split] = bootstrap_test(preds, target, metrics)

    return result


def spot(
    datasets, metrics, model_name, no_bootstrap_test, seed, output_path, *args, **kwargs
):
    model_args = {"seed": seed, "output_dir": output_path}

    if "hint" in model_name:
        if "low_lr" in model_name:
            model_args["learning_rate"] = 3e-4
        else:
            model_args["learning_rate"] = 1e-3
        model_args["weight_decay"] = 0

    if "5e" in model_name:
        model_args["epochs"] = 5

    model = SPOT(**model_args)
    model.fit(datasets["train"], datasets["valid"])

    result = {}
    for split in ["valid", "test"]:
        print(split)
        preds = model.predict(datasets[split])
        target = torch.tensor(preds["label"])
        preds = torch.tensor(preds["pred"][:, 0])
        if no_bootstrap_test:
            result[split] = metrics(preds, target)
            metrics.reset()
        else:
            result[split] = bootstrap_test(preds, target, metrics)

    shutil.rmtree(output_path)

    return result


model_map = defaultdict(lambda: spot)
model_map["hint"] = hint


def argparse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spot_hint,spot_hint_5e,spot,hint")
    parser.add_argument("--phase", type=str, default="I,II,III")
    parser.add_argument(
        "--no-bootstrap-test",
        action="store_false",
        help="do not use bootstrap test",
    )
    parser.add_argument("--output-path", type=str, default="results")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    return parser.parse_args()


def main():
    metrics = MetricCollection(
        {
            "F1": F1Score("binary"),
            "ROC-AUC": AUROC("binary"),
            "PR-AUC": AveragePrecision("binary"),
            "STAT": StatScores("binary"),
        }
    )

    args = argparse()

    for phase in args.phase.split(","):
        data = {
            s: load_trial_outcome_data(phase=phase, split=s)["data"]
            for s in ["train", "valid", "test"]
        }

        for split in data:
            data[split] = data[split].dropna(axis=0, subset=["criteria"])

        for model in args.model.split(","):
            if model == "summary":
                data_path = os.path.join(args.output_path, "details", phase)
                output_path = os.path.join(args.output_path, "summary")
                os.makedirs(output_path, exist_ok=True)

                res_string = ""
                for model_name in os.listdir(data_path):
                    results = set(
                        [
                            int(res.removesuffix(".pkl"))
                            for res in os.listdir(os.path.join(data_path, model_name))
                        ]
                    )
                    results = set(range(args.n)) - results
                    if len(results) > 0:
                        print(f"{model_name} does not complete: {results}")
                        continue

                    results = [
                        pickle.load(
                            open(os.path.join(data_path, model_name, f"{i}.pkl"), "rb")
                        )
                        for i in range(args.n)
                    ]
                    results = parse_results(results)

                    res_string += f"{model_name} "
                    for metric in ["PR-AUC", "F1", "ROC-AUC"]:
                        res_string += f'& $ {results["test"][metric][0]:.3g} \pm {results["test"][metric][1]:.3g} $'
                    res_string += "\\\\\n"
                open(os.path.join(output_path, f"{phase}.txt"), "w").write(res_string)
            else:
                if model == "hint":
                    datasets = {k: TrialOutcomeDatasetBase(v) for k, v in data.items()}
                else:
                    datasets = {k: TrialOutcomeDataset(v) for k, v in data.items()}

                fit_modal(
                    model,
                    phase,
                    args.n,
                    args.world_size,
                    args.rank,
                    output_path=os.path.join(args.output_path, "details"),
                    datasets=datasets,
                    metrics=metrics,
                    data=data,
                )


if __name__ == "__main__":
    main()

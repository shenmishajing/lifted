# load demo data
import os
import pickle
import random

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


def fit_modal(
    model_name,
    phase,
    n=30,
    world_size=1,
    rank=0,
    output_path="results/details",
    *args,
    **kwargs,
):
    output_path = os.path.join(output_path, phase, model_name)

    for i in trange(rank, n, world_size):
        seed = get_random_seed()
        seed_everything(seed)

        result = model_map[model_name](
            *args,
            **kwargs,
            seed=seed,
            model_name=model_name,
            output_path=os.path.join(output_path, str(i)),
        )
        result["seed"] = seed
        pickle.dump(result, open(os.path.join(output_path, f"{i}.pkl"), "wb"))


def hint(datasets, metrics, datas, *args, **kwargs):
    model = HINT(highway_num_layer=2, epoch=1, lr=1e-3)
    model.fit(datasets["train"], datasets["valid"])

    result = {}
    for split in ["valid", "test"]:
        preds = model.predict(datasets[split])
        target = datas[split][["nctid", "label"]].set_index("nctid").to_dict("split")
        target = {k: v[0] for k, v in zip(target["index"], target["data"])}
        target = torch.tensor([target[k[0]] for k in preds])
        preds = torch.tensor([k[1] for k in preds])
        result[split] = metrics(preds, target)
        metrics.reset()

    return result


def spot(datasets, metrics, model_name, seed, output_path, *args, **kwargs):
    if model_name == "spot_hint":
        model = SPOT(
            epochs=5,
            learning_rate=1e-3,
            weight_decay=0,
            seed=seed,
            output_dir=os.path.join(output_path, "checkpoints"),
        )
    else:
        model = SPOT(seed=seed, output_dir=os.path.join(output_path, "checkpoints"))

    model.fit(datasets["train"], datasets["valid"])

    result = {}
    for split in ["valid", "test"]:
        print(split)
        preds = model.predict(datasets[split])
        target = torch.tensor(preds["label"])
        preds = torch.tensor(preds["pred"][:, 0])
        result[split] = metrics(preds, target)
        metrics.reset()

    return result


model_map = {"hint": hint, "spot": spot, "spot_hint": spot}


def argparse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spot_hint,spot,hint")
    parser.add_argument("--phase", type=str, default="I,II,III")
    parser.add_argument("--output-path", type=str, default="results")
    parser.add_argument("--n", type=int, default=30)
    parser.add_augment("--world-size", type=int, default=1)
    parser.add_augment("--rank", type=int, default=0)
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
        datas = {
            s: load_trial_outcome_data(phase=phase, split=s)["data"]
            for s in ["train", "valid", "test"]
        }

        for split in datas:
            datas[split] = datas[split].dropna(axis=0, subset=["criteria"])

        for model in args.model.split(","):
            if model == "summary":
                data_path = os.path.join(args.output_path, "details", phase)
                output_path = os.path.join(args.output_path, "summary")

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
                    datasets = {k: TrialOutcomeDatasetBase(v) for k, v in datas.items()}
                else:
                    datasets = {k: TrialOutcomeDataset(v) for k, v in datas.items()}

                fit_modal(
                    model,
                    phase,
                    args.n,
                    args.world_size,
                    args.rank,
                    output_path=os.path.join(args.output_path, "details"),
                    datasets=datasets,
                    metrics=metrics,
                    datas=datas,
                )


if __name__ == "__main__":
    main()

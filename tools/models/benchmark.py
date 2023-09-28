# load demo data
import os
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


def parse_results(results):
    for split in results:
        result = {}
        for metric in results[split][0]:
            result[metric] = torch.stack([k[metric] for k in results[split]])

        for metric in result:
            result[metric] = [result[metric].mean().item(), result[metric].std().item()]

        results[split] = result
    return results


def hint(datas, metrics, phase, n=30, output_path="results/details"):
    results = defaultdict(list)
    datasets = {k: TrialOutcomeDatasetBase(v) for k, v in datas.items()}

    os.makedirs(output_path, exist_ok=True)
    file = open(
        f"results/details/{phase}_hint.txt",
        "w",
    )

    for _ in trange(n):
        seed = get_random_seed()
        file.write(f"seed: {seed}\n")
        seed_everything(seed)

        model = HINT(highway_num_layer=2, epoch=1, lr=1e-3)
        model.fit(datasets["train"], datasets["valid"])

        for split in ["valid", "test"]:
            preds = model.predict(datasets[split])
            target = (
                datas[split][["nctid", "label"]].set_index("nctid").to_dict("split")
            )
            target = {k: v[0] for k, v in zip(target["index"], target["data"])}
            target = torch.tensor([target[k[0]] for k in preds])
            preds = torch.tensor([k[1] for k in preds])
            result = metrics(preds, target)
            results[split].append(result)
            metrics.reset()

            file.write(f"{split}\t{result}\n")

    file.close()

    return results


def spot(
    datas,
    metrics,
    phase,
    n=30,
    use_hint_hyperparameters=True,
    output_path="results/details",
):
    results = defaultdict(list)
    datasets = {k: TrialOutcomeDataset(v) for k, v in datas.items()}

    os.makedirs(output_path, exist_ok=True)

    for _ in trange(n):
        file = open(
            f"results/details/{phase}_spot{'_hint' if use_hint_hyperparameters else ''}.txt",
            "a",
        )
        seed = get_random_seed()
        file.write(f"seed: {seed}\n")
        seed_everything(seed)

        if use_hint_hyperparameters:
            model = SPOT(
                epochs=5,
                learning_rate=1e-3,
                weight_decay=0,
                seed=seed,
                output_dir=f"./checkpoints/spot/phase_{phase}",
            )
        else:
            model = SPOT(seed=seed, output_dir=f"./checkpoints/spot/phase_{phase}")

        model.fit(datasets["train"], datasets["valid"])

        for split in ["valid", "test"]:
            print(split)
            preds = model.predict(datasets[split])
            target = torch.tensor(preds["label"])
            preds = torch.tensor(preds["pred"][:, 0])
            result = metrics(preds, target)
            results[split].append(result)
            metrics.reset()

            file.write(f"{split}\t{result}\n")

        shutil.rmtree(f"./checkpoints/spot/phase_{phase}")

        file.close()

    return results


def argparse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spot_hint,spot,hint")
    parser.add_argument("--phase", type=str, default="I,II,III")
    parser.add_argument("--n", type=int, default=30)
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

    models = args.model.split(",")
    phases = args.phase.split(",")
    save_path = "results/summarization"
    os.makedirs(save_path, exist_ok=True)

    for phase in phases:
        datas = {
            s: load_trial_outcome_data(phase=phase, split=s)["data"]
            for s in ["train", "valid", "test"]
        }

        for split in datas:
            datas[split] = datas[split].dropna(axis=0, subset=["criteria"])

        for model in models:
            if model == "hint":
                results = hint(datas, metrics, phase, args.n)
            elif model == "spot":
                results = spot(datas, metrics, phase, args.n, False)
            elif model == "spot_hint":
                results = spot(datas, metrics, phase, n=args.n)

            results = parse_results(results)

            res_string = ""
            for metric in ["PR-AUC", "F1", "ROC-AUC"]:
                res_string += f'& $ {results["test"][metric][0]:.3g} \pm {results["test"][metric][1]:.3g} $'
            res_string += "\\\\\n"
            open(os.path.join(save_path, f"{phase}_{model}.txt"), "w").write(res_string)


if __name__ == "__main__":
    main()

# load demo data
import torch
from pytrial.data.demo_data import load_trial_outcome_data

# prepare the input data class for this task
from pytrial.tasks.trial_outcome import HINT, SPOT
from pytrial.tasks.trial_outcome.data import (
    TrialOutcomeDataset,
    TrialOutcomeDatasetBase,
)
from pytrial.tasks.trial_outcome.evaluation import find_the_best_threshold
from torchmetrics import AUROC, AveragePrecision, F1Score, MetricCollection, StatScores


def hint(datas, metrics):
    datasets = {k: TrialOutcomeDatasetBase(v) for k, v in datas.items()}

    model = HINT(highway_num_layer=2, epoch=5, lr=1e-3)
    model.fit(datasets["train"], datasets["valid"])

    for split in ["valid", "test"]:
        preds = model.predict(datasets[split])
        target = datas[split][["nctid", "label"]].set_index("nctid").to_dict("split")
        target = {k: v[0] for k, v in zip(target["index"], target["data"])}
        target = torch.tensor([target[k[0]] for k in preds])
        preds = torch.tensor([k[1] for k in preds])
        print(split, preds.shape, target.shape, metrics(preds, target))
        metrics.reset()


def spot(datas, metrics, phase):
    datasets = {k: TrialOutcomeDataset(v) for k, v in datas.items()}

    model = SPOT(
        epochs=5,
        learning_rate=1e-3,
        weight_decay=0,
        output_dir=f"./checkpoints/spot/phase_{phase}",
    )
    model.fit(datasets["train"], datasets["valid"])

    for split in ["valid", "test"]:
        print(split)
        preds = model.predict(datasets[split])
        print("shape:", preds["pred"].shape, preds["label"].shape)
        best_threshold = find_the_best_threshold(preds["pred"], preds["label"])
        print("best threshold:", best_threshold)
        target = torch.tensor(preds["label"])
        preds = torch.tensor(preds["pred"][:, 0])
        print("metrics", metrics(preds, target))
        metrics.reset()

        metrics_with_threshold = MetricCollection(
            {
                "F1": F1Score("binary", threshold=best_threshold),
                "ROC-AUC": AUROC("binary"),
                "PR-AUC": AveragePrecision("binary"),
                "STAT": StatScores("binary", threshold=best_threshold),
            }
        )

        print("metrics with threshold", metrics_with_threshold(preds, target))


def argparse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="spot")
    parser.add_argument("--phase", type=str, default="I")
    return parser.parse_args()


def main():
    args = argparse()
    metrics = MetricCollection(
        {
            "F1": F1Score("binary"),
            "ROC-AUC": AUROC("binary"),
            "PR-AUC": AveragePrecision("binary"),
            "STAT": StatScores("binary"),
        }
    )

    datas = {
        s: load_trial_outcome_data(phase=args.phase, split=s)["data"]
        for s in ["train", "valid", "test"]
    }

    for split in datas:
        datas[split] = datas[split].dropna(axis=0, subset=["criteria"])

    if args.model == "hint":
        hint(datas, metrics)
    elif args.model == "spot":
        spot(datas, metrics, args.phase)


if __name__ == "__main__":
    main()

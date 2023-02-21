import copy
import string
from abc import ABC
from collections.abc import Mapping, Sequence

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import \
    LightningDataModule as _LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

from utils import deep_update


class LightningDataModule(_LightningDataModule):
    SPLIT_NAMES = ["train", "val", "test", "predict"]

    def __init__(
        self,
        dataset_cfg: dict,
        dataloader_cfg: dict = None,
    ):
        super().__init__()

        self.dataset_cfg = self.get_split_config(dataset_cfg)
        self.dataloader_cfg = self.get_split_config(dataloader_cfg)

        self.datasets = {}
        self.dataset = None

    def get_split_config(self, config):
        if isinstance(config, Mapping):
            if all([config.get(name) is None for name in self.SPLIT_NAMES]):
                return {name: copy.deepcopy(config) for name in self.SPLIT_NAMES}
            else:
                res = {}
                last_name = None
                for name in self.SPLIT_NAMES:
                    if last_name is None:
                        res[name] = copy.deepcopy(config[name])
                    else:
                        res[name] = deep_update(
                            copy.deepcopy(res[last_name]), config.get(name, {})
                        )
                    last_name = name

                if "split_info" in config:
                    config = config["split_info"]

                    if not isinstance(config["split_format_to"], Sequence):
                        config["split_format_to"] = [config["split_format_to"]]

                    config["split_name_map"] = config.get("split_name_map", {})
                    for name in self.SPLIT_NAMES:
                        config["split_name_map"].setdefault(
                            name, name if name != "predict" else "test"
                        )

                    config.setdefault("split_attr_split_str", ".")

                    for name in self.SPLIT_NAMES:
                        if res[name].get("init_args") is None:
                            res[name]["init_args"] = {}
                        for split_attr in config["split_format_to"]:
                            cur_cfg = res[name]["init_args"]
                            split_attr = split_attr.split(
                                config["split_attr_split_str"]
                            )
                            for s in split_attr[:-1]:
                                cur_cfg = cur_cfg[s]
                            split_attr = split_attr[-1]
                            cur_cfg[split_attr] = string.Template(
                                cur_cfg.get(split_attr, "$split")
                            ).safe_substitute(split=config["split_name_map"][name])
                return res
        else:
            return {
                name: copy.deepcopy(config) if config else {}
                for name in self.SPLIT_NAMES
            }

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = ["train", "val", "test", "predict"]
        elif stage == "fit":
            split_names = ["train", "val"]
        else:
            split_names = [stage.lower()]
        return split_names

    def _setup_dataset(self, split_name):
        self.datasets[split_name] = self._build_data_set(
            self.split_name_map[split_name]
        )

    def setup(self, stage=None):
        split_names = self._get_split_names(stage)

        for name in split_names:
            self._setup_dataset(name)
        self.dataset = self.datasets[split_names[0]]

    def _dataloader(self, split_name, **kwargs):
        return self._build_data_loader(
            self.datasets[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")

    def _build_data_set(self, split):
        return instantiate_class(tuple(), self.dataset_cfg[split])

    def _build_data_loader(self, dataset, split="train"):
        if isinstance(dataset, Mapping):
            return {
                key: self._build_data_loader(
                    ds,
                    split=split,
                )
                for key, ds in dataset.items()
            }
        elif isinstance(dataset, Sequence):
            return [
                self._build_data_loader(
                    ds,
                    split=split,
                )
                for ds in dataset
            ]
        else:
            collate_fn = self.collate_fn if hasattr(self, "collate_fn") else None
            return DataLoader(
                dataset=dataset, collate_fn=collate_fn, **self.dataloader_cfg[split]
            )


class KFoldLightningDataModule(LightningDataModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_folds = None
        self.folds = {}
        self.splits = []

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = ["train", "val", "test", "predict"]
        elif stage == "fit":
            split_names = ["train"]
        else:
            split_names = [stage.lower()]
        return split_names

    def setup(self, stage=None):
        super().setup(stage)
        split_names = self._get_split_names(stage)
        if "train" in split_names or "val" in split_names:
            self.setup_folds(2)
            self.setup_fold_index(0)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.dataset)))
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        for indices, fold_name in zip(self.splits[fold_index], ["train", "val"]):
            self.folds[fold_name] = Subset(self.dataset, indices)

    def _fold_dataloader(self, split_name, **kwargs):
        return self._build_data_loader(
            self.folds[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._fold_dataloader("train")

    def val_dataloader(self):
        return self._fold_dataloader("val")

## Introduction

## Installation

See [installation docs](docs/installation/installation.md) for details.

## Prepare

### Openai

```bash
echo <your-key> > openai_api_key.txt
```

### Wandb

```bash
wandb login
### paste your api key
```

### Data

```bash
mkdir data
```

#### Hint

```bash
git clone https://github.com/futianfan/clinical-trial-outcome-prediction.git data/clinical-trial-outcome-prediction
```

#### Description

upload text_data.zip to data/clinical-trial-outcome-prediction

Unzip text_data.zip and put them in data/clinical-trial-outcome-prediction

```bash
unzip data/clinical-trial-outcome-prediction/text_data.zip
mv data/clinical-trial-outcome-prediction/text_data/* data/clinical-trial-outcome-prediction
rm -rf data/clinical-trial-outcome-prediction/text_data
```

## Experiment

### Adjust batch size

Modify `batch_size` in `configs/datasets/hint/hint_phase_I.yaml` data.dataloader_cfg.batch_size
Modify `accumulate_grad_batches` in `configs/datasets/hint/accumulate_grad_batches_patch.yaml` trainer.accumulate_grad_batches to ensure that `batch_size * accumulate_grad_batches` is 32.

Run

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization_augment_5e.yaml
```

to check if the batch size is optimal.

### Run

Run experiments for phase I

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> shell_cmd_launcher 'cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization_augment_5e.yaml' --num 10
CUDA_VISIBLE_DEVICES=<gpu_id> shell_cmd_launcher 'cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_no-summarization_augment_5e.yaml' --num 10
CUDA_VISIBLE_DEVICES=<gpu_id> shell_cmd_launcher 'cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization_no-table_augment_5e.yaml' --num 10
CUDA_VISIBLE_DEVICES=<gpu_id> shell_cmd_launcher 'cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization_no-augment_5e.yaml' --num 10
CUDA_VISIBLE_DEVICES=<gpu_id> shell_cmd_launcher 'cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_no-summarization_no-augment_5e.yaml' --num 10
CUDA_VISIBLE_DEVICES=<gpu_id> shell_cmd_launcher 'cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization_no-table_no-augment_5e.yaml' --num 10
```

Replace `phase_I` with `phase_II` and `phase_III` to run experiments for phase II and III.

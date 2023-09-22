## Experiment

### Adjust batch size

In `configs/datasets/ct-gov/ct-gov.yaml`, set data.dataloader_cfg.batch_size to 32, or bigger.

Run

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_summarization-no-table_ct-gov_1x.yaml
```

to check if the batch size is optimal.

### Run

#### Pretrain

Run the pretrain experiment.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_summarization-no-table_ct-gov_1x.yaml
```

record the run id from which wandb reports. For example, wandb may report the save dir is `work_dirs/mmcto_summarization-no-table_ct-gov_1x/2023-09-22_21-42-00.789666/wandb/run-20230922_214202-2023-09-22_21-42-00.789666`, then the run id is `2023-09-22_21-42-00.789666`.

#### Load pretrain checkpoint

In `configs/models/mmcto/mmcto_summarization-no-table_augment_aux-loss_load-pretrained-ckpt` change the `ckpt_path` to `work_dirs/mmcto_summarization-no-table_ct-gov_1x/<run_id>/checkpoints/latest.pth`.


#### Run

Run experiments for phase I, II and III on different cards parallelly:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_II_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_III_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml
```

Or run experiments for phase I, II and III on the same card sequentially:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_${phase}_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml' --arg_dict.phase 'I,II,III'
```

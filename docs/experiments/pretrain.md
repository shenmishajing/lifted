## Experiment

### Pretrain

Run the pretrain experiment.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/pretrain/mmcto_summarization-no-table_ct-gov_1x.yaml
```

record the run id from which wandb reports. For example, wandb may report the save dir is `work_dirs/mmcto_summarization-no-table_ct-gov_1x/2023-09-22_21-42-00.789666/wandb/run-20230922_214202-2023-09-22_21-42-00.789666`, then the run id is `2023-09-22_21-42-00.789666`.

### Modify pretrain checkpoint patd and batch size

In `configs/models/mmcto/pretrain/mmcto_summarization-no-table_augment_aux-loss_load-pretrained-ckpt.yaml` change the `model.init_args.ckpt_path` to `work_dirs/mmcto_summarization-no-table_ct-gov_1x/<run_id>/checkpoints/latest.pth`.

### Run

Run experiments for phase I, II and III on different gpus parallelly:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/pretrain/mmcto_hint_phase_I_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml' --num 30
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/pretrain/mmcto_hint_phase_II_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml' --num 30
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/pretrain/mmcto_hint_phase_III_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml' --num 30
```

Or run experiments for phase I, II and III on the same gpu sequentially:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/pretrain/mmcto_hint_phase_${phase}_summarization-no-table_augment_aux-loss_load-pretrained-ckpt_5e.yaml' --num 30 --arg_dict.phase 'I,II,III'
```

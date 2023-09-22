## Experiment

### Adjust batch size

In `configs/datasets/hint/augment_batch_size_patch.yaml`, set data.dataloader_cfg.batch_size to 32, set trainer.accumulate_grad_batches to 1

Run

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization_augment_5e.yaml
```

to check if there is an OOM error.

### Run

Run experiments for phase I, II and III on different cards parallelly:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_I_summarization-no-table_augment${run}_5e.yaml' --num 10 --arg_dict.run ',_aux-loss,_aux-loss-share-fc,_aux-loss-weighted,_aux-loss-weighted-share-fc'
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_II_summarization-no-table_augment${run}_5e.yaml' --num 10 --arg_dict.run ',_aux-loss,_aux-loss-share-fc,_aux-loss-weighted,_aux-loss-weighted-share-fc'
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_III_summarization-no-table_augment${run}_5e.yaml' --num 10 --arg_dict.run ',_aux-loss,_aux-loss-share-fc,_aux-loss-weighted,_aux-loss-weighted-share-fc'
```

Or run experiments for phase I, II and III on the same card sequentially:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/mmcto_hint_phase_${phase}_summarization-no-table_augment${run}_5e.yaml' --num 10 --arg_dict.run ',_aux-loss,_aux-loss-share-fc,_aux-loss-weighted,_aux-loss-weighted-share-fc' --arg_dict.phase 'I,II,III'
```

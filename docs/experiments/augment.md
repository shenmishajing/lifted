## Experiment

### Data augment probability ablation study

Run experiments for phase I, II and III on different gpus parallelly:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_I_summarization-no-table_augment_5e.yaml --model.init_args.model.init_args.augment_prob ${run}' --num 30 --arg_dict.run '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_II_summarization-no-table_augment_5e.yaml --model.init_args.model.init_args.augment_prob ${run}' --num 30 --arg_dict.run '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_III_summarization-no-table_augment_5e.yaml --model.init_args.model.init_args.augment_prob ${run}' --num 30 --arg_dict.run '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'
```

Or run experiments for phase I, II and III on the same gpu sequentially:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_${phase}_summarization-no-table_augment_5e.yaml --model.init_args.model.init_args.augment_prob ${run}' --num 30 --arg_dict.run '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0' --arg_dict.phase 'I,II,III'
```

### Augment, aux loss and consistency loss ablation study

Run experiments for phase I, II and III on different gpus parallelly:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_I_summarization-no-table_augment${run}_5e.yaml' --num 30 --arg_dict.run ',_aux-loss,-cos-loss,-cos-loss_aux-loss'
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_II_summarization-no-table_augment${run}_5e.yaml' --num 30 --arg_dict.run ',_aux-loss,-cos-loss,-cos-loss_aux-loss'
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_III_summarization-no-table_augment${run}_5e.yaml' --num 30 --arg_dict.run ',_aux-loss,-cos-loss,-cos-loss_aux-loss'
```

Or run experiments for phase I, II and III on the same gpu sequentially:

```bash
shell_command_launcher 'CUDA_VISIBLE_DEVICES=<gpu_id> cli fit --config configs/runs/mmcto/augment/mmcto_hint_phase_${phase}_summarization-no-table_augment${run}_5e.yaml' --num 30 --arg_dict.run ',_aux-loss,-cos-loss,-cos-loss_aux-loss' --arg_dict.phase 'I,II,III'
```

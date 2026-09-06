# RoboCasa-GR1 Tabletop Example

This folder contains WALA training and evaluation files for RoboCasa-GR1
tabletop tasks. The training directory provides the example configuration and
launch script for RoboCasa policy training.

## Training

Train the Qwen3-VL policy:

```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wala.sh
```

Train the Wan2.2 policy:

```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wala_wan.sh
```

The corresponding configurations are `wala_robocasa.yaml` and
`wala_wan_robocasa.yaml`. Update the dataset path and
`trainer.pretrained_checkpoint` before launching distributed training.

## Environment Setup

Follow the official [RoboCasa-GR1 tabletop task setup](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
to create the `RoboCasa` simulator environment and prepare the required assets.
WALA uses a two-environment workflow: the policy server runs in the `WALA`
environment, while the simulator runs in the `RoboCasa` environment.

Install the evaluation-side bridge dependencies in the `RoboCasa` environment:

```bash
conda activate RoboCasa
pip install tyro websockets msgpack rich omegaconf av imageio imageio-ffmpeg
```

## Evaluation

Run the RoboCasa batch evaluator:

```bash
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh
```

For the PPU runtime launcher:

```bash
bash examples/Robocasa_tabletop/eval_files/run_robocasa_eval_ppu.sh \
    results/Checkpoints/<run_id>/checkpoints/steps_xxx_pytorch_model.pt \
    1 720 32
```

Modify the checkpoint path, Python paths, GPU count, episode count, and rollout
settings directly in the corresponding shell script before running. Both
scripts start policy servers, dispatch all RoboCasa-GR1 tabletop tasks, and save
logs and rollout videos under the corresponding `results/Checkpoints/<run_id>/`
directory. The checkpoint's saved `config.yaml` selects either the Qwen3-VL or
Wan2.2 policy automatically.

# WALA

Anonymous code release accompanying the paper.

**WALA** stands for **World- and Action-supervised Latent Actions**. This repository contains the project code for **WALA: Learning Executable Latent Actions from Action-Labeled Demonstrations and Action-Free Videos**.

This repository includes:

- the semantic-geometric latent action model built on DINOv3 features and dense depth supervision;
- policy training with robot action prediction, latent action target matching, and future dynamics supervision;
- Qwen3-VL-4B-Instruct and Wan2.2-TI2V-5B policy backbones;
- RoboCasa-GR1 tabletop and RoboTwin training and evaluation examples;
- WebSocket policy-server utilities for local and split-machine evaluation.

## Model Variants

WALA provides two policy implementations that share the same latent action
supervision and MLP action head:

- **WALA (`framework.name: WALA`)** uses Qwen3-VL-4B-Instruct as the
  vision-language backbone. Latent actions are read from the hidden states of
  learned action-query tokens.
- **WALA-Wan (`framework.name: WALA_WAN`)** uses the Diffusers release of
  Wan2.2-TI2V-5B as a video-generation backbone. A learnable action-query
  resampler cross-attends to Wan hidden tokens. Robot state can be injected as
  text or through a separate state encoder and cross-attention path.

The provided Wan benchmark configs use one current observation frame. RoboTwin
forms a fixed-size mosaic from the head and two wrist cameras, while RoboCasa
uses its single egocentric view. Future observations supervise training only;
action prediction is conditioned on the current observation, instruction, and
robot state.

## Repository Layout

```text
wala/                         Core Python package
  model/framework/policies/   Qwen3-VL and Wan2.2 WALA policies
  model/modules/latent_action_model/  Latent action model modules
  model/modules/vision_encoder/       DINOv3 wrapper
  training/                   Training entrypoints
deployment/model_server/      WebSocket policy server used by simulation evaluation
examples/
  Robocasa_tabletop/          RoboCasa-GR1 tabletop training and evaluation
  Robotwin/                   RoboTwin training and evaluation
scripts/                      Depth-cache preprocessing utility and documentation
checkpoints/                  Local directory for third-party pretrained weights
results/                      Training checkpoints, logs, and rollout videos
```

Datasets, simulator assets, pretrained LAM and policy weights, third-party pretrained weights, and training outputs are not included. Set the dataset and pretrained LAM paths in the example configurations before training. Evaluation requires a policy checkpoint with its saved `config.yaml` and normalization statistics.

## Environment Setup

WALA uses one Python environment for model training and policy serving. RoboCasa and RoboTwin evaluation should be installed in separate simulator environments.

### 1. Create the WALA environment

Install the supported CUDA 12.4 PyTorch build:

```bash
conda create -n WALA python=3.10 -y
conda activate WALA

pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install WALA and the core Python dependencies:

```bash
git clone https://github.com/WALA-Anonymous/WALA.git
cd WALA

pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the Diffusers release used by the Wan2.2 policy.

### 2. Depth Anything V2

WALA uses the bundled Depth Anything V2 implementation for dense depth
supervision. The source code is already included under
`wala/model/third_party/Depth-Anything-V2`, and WALA adds this directory to
Python's module search path at runtime. Do not clone another copy or run
`pip install -e .` in that directory. Its inference dependencies are included
in WALA's main requirements. The upstream Depth Anything requirements also
install Gradio packages for its standalone demo; those packages are not needed
for WALA training or evaluation.

For the original Depth Anything V2 usage notes and checkpoint links, see
`wala/model/third_party/Depth-Anything-V2/README.md`.

### 3. Prepare third-party pretrained weights

Download the required third-party checkpoints according to their original
licenses and place them under `checkpoints/`. The example configurations use
the following local layout:

```text
checkpoints/
  qwen/Qwen3-VL-4B-Instruct/
  wan/Wan2.2-TI2V-5B-Diffusers/
  dinov3/dinov3-vitb16-pretrain-lvd1689m/
  depth_anything/depth_anything_v2_vitl.pth
```

If your checkpoints are stored elsewhere, update the paths in the corresponding
example YAML files before training or evaluation.

### 4. Prepare datasets

WALA supports LeRobot v2.1 and v3.0 datasets. Each dataset must provide
`meta/modality.json` matching the video, state, action, and language keys used
by its data registry. Prepare the datasets required by each benchmark and
update `data_root_dir`, `data_mix`, and the registry entries in:

```text
examples/Robocasa_tabletop/train_files/wala_robocasa.yaml
examples/Robocasa_tabletop/train_files/wala_wan_robocasa.yaml
examples/Robocasa_tabletop/train_files/data_registry/data_config.py
examples/Robotwin/train_files/wala_robotwin.yaml
examples/Robotwin/train_files/wala_wan_robotwin.yaml
examples/Robotwin/train_files/data_registry/data_config.py
```

Dataset statistics and step indices are cached under `cache/datasets/` by
default, so read-only source datasets are supported without writing generated
metadata into their original directories.

### 5. RoboCasa-GR1 tabletop simulator environment

RoboCasa evaluation uses two processes:

- the WALA policy server, launched in the `WALA` environment;
- the RoboCasa simulator, launched in a separate `RoboCasa` environment.

First install the simulator following the official
[RoboCasa-GR1 tabletop task setup](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).
WALA serves the policy in the `WALA` environment, while the RoboCasa simulator runs in its own
`RoboCasa` environment.

After the simulator is installed, install the small set of evaluation-side
Python packages used by the WALA bridge:

```bash
conda activate RoboCasa
pip install tyro websockets msgpack rich omegaconf av imageio imageio-ffmpeg
```

See `examples/Robocasa_tabletop/README.md` for the full details.

### 6. RoboTwin simulator environment

RoboTwin evaluation also uses two processes:

- the WALA policy server, launched in the `WALA` environment;
- the RoboTwin evaluator, launched in a separate `RoboTwin` environment.

Use a current RoboTwin checkout and its bundled installer. The policy server
runs in the `WALA` environment, while rendering and task evaluation run in the
separate `RoboTwin` environment:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

cd third_party/RoboTwin
bash script/_install.sh
python script/update_embodiment_config_path.py
```

The current evaluator expects the cuRobo `v0.7.8` API. If the RoboTwin
installer fetched a newer cuRobo release, replace it with `v0.7.8` and rebuild
the native extension for the local GPU:

```bash
cd /absolute/path/to/third_party/RoboTwin
rm -rf envs/curobo
git clone --branch v0.7.8 --depth 1 \
    https://github.com/NVlabs/curobo.git envs/curobo

# Set this to the compute capability of the target GPU (for example, 8.0).
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
pip install -e envs/curobo --no-build-isolation
```

The separate Facebook PyTorch3D build requested by RoboTwin's installer is
optional for WALA's RoboTwin evaluation and may be skipped in the `RoboTwin`
environment. The WALA environment still installs `pipablepytorch3d` from its
own `requirements.txt` because WALA's state/action transforms use it.

Then install WALA's evaluation bridge in the simulator environment and point
WALA to the RoboTwin checkout:

```bash
cd /absolute/path/to/WALA
conda activate RoboTwin
pip install -r examples/Robotwin/eval_files/requirements.txt
export ROBOTWIN_PATH=/absolute/path/to/third_party/RoboTwin
```

WALA's evaluator passes the checkpoint path to RoboTwin through
`--policy_ckpt_path`. Some RoboTwin revisions already support this argument;
older revisions require the compatibility patch documented in
`examples/Robotwin/README.md`.

See `examples/Robotwin/README.md` for the full details.

## Training

Run commands from the repository root. The scripts write checkpoints and logs under `results/Checkpoints/`.

Train the Qwen3-VL policy on RoboCasa-GR1 tabletop:

```bash
conda activate WALA
bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wala.sh
```

Train the Wan2.2 policy on RoboCasa-GR1 tabletop:

```bash
conda activate WALA
bash examples/Robocasa_tabletop/train_files/run_robocasa_train_wala_wan.sh
```

Train the Qwen3-VL policy on RoboTwin:

```bash
conda activate WALA
bash examples/Robotwin/train_files/run_robotwin_train_wala.sh
```

Train the Wan2.2 policy on RoboTwin:

```bash
conda activate WALA
bash examples/Robotwin/train_files/run_robotwin_train_wala_wan.sh
```

The benchmark YAML files expect a pretrained latent action model through
`trainer.pretrained_checkpoint` and load its `transition_bottleneck` weights.
Update that path before training. Before launching distributed training, also
check the machine count, process count, batch size, NCCL interface, and
`accelerate` settings in each script. The launchers use the DeepSpeed ZeRO-2
configuration under `wala/config/deepseeds/`.

## Evaluation

### RoboCasa-GR1 tabletop

Edit checkpoint paths and Python paths directly in the shell script, then run:

```bash
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh
```

For the provided PPU runtime launcher:

```bash
bash examples/Robocasa_tabletop/eval_files/run_robocasa_eval_ppu.sh \
    results/Checkpoints/<run_id>/checkpoints/steps_xxx_pytorch_model.pt \
    1 720 32
```

Both scripts start WALA policy servers, dispatch the RoboCasa tasks, and save logs/videos under the corresponding `results/Checkpoints/<run_id>/` directory.

### RoboTwin

Set `ROBOTWIN_PATH`, then launch:

```bash
conda activate WALA
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin

bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean \
    -n full_eval \
    -s 42 \
    -j 2 \
    -c results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt \
    -p 6666 \
    all
```

Use `demo_clean` for the Easy setting and `demo_randomized` for the Hard setting. See `examples/Robotwin/README.md` for task lists, logging paths, and optional flags.

For split evaluation, where PPU devices run policy inference and an H20
machine renders RoboTwin environments, first launch a pool of policy servers
on the PPU machine:

```bash
GPU_IDS=0,1,2,3 JOBS_PER_GPU=2 \
bash examples/Robotwin/eval_files/run_robotwin_policy_servers_ppu.sh \
    results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt \
    6666
```

The servers use consecutive ports beginning at `6666`.

On the H20 machine, launch the tunnel and RoboTwin clients with one command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/Robotwin/eval_files/run_robotwin_eval_h20.sh \
    --ppu-host <PPU_SSH_HOST> \
    --ppu-ssh-port <PPU_SSH_PORT> \
    --num-servers 8 \
    -m demo_clean \
    -n full_eval \
    -s 42 \
    -j 4 \
    -c /path/on/h20/to/steps_xxx_pytorch_model.pt \
    -p 6666 \
    all
```

The H20 launcher forwards the policy-server ports over SSH, schedules the
RoboTwin clients, and closes the tunnel on exit. See
`examples/Robotwin/README.md` for the complete cross-machine procedure.

## Notes

- The example training launchers disable W&B by default. To enable it, set
  `WANDB_MODE=online` and configure your own `wandb_entity`. For direct Python
  launches, set `WANDB_MODE=disabled` explicitly if no online logging is needed.
- Machine-specific paths are placeholders or environment-variable overrides.
  Set `WALA_PYTHON`, `ROBOTWIN_PYTHON`, or `ROBOCASA_PYTHON` to your environment's
  Python executable when using the evaluation launchers.
- Training outputs are written under `results/`.
- Third-party pretrained model files can be placed under `checkpoints/`.
- The policy server reads the framework type and model settings from the
  checkpoint's saved `config.yaml`, so the same evaluation scripts support
  both Qwen3-VL and Wan2.2 checkpoints.
- RoboCasa and RoboTwin are separate projects; install and update them independently from WALA.

## Acknowledgement

We sincerely thank the great open-source [StarVLA](https://github.com/starVLA/starVLA) project. If you encounter environment installation issues, you may also refer to the StarVLA environment setup and benchmark-specific installation instructions. The environments used in this project are built on top of the StarVLA setup. We thank StarVLA again for its valuable open-source contribution.

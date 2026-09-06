# RoboTwin Example

This folder contains WALA training and evaluation files for RoboTwin 2.0.
The training directory provides the example configuration and launch script for
RoboTwin policy training.

## Training

Train the Qwen3-VL policy:

```bash
bash examples/Robotwin/train_files/run_robotwin_train_wala.sh
```

Train the Wan2.2 policy:

```bash
bash examples/Robotwin/train_files/run_robotwin_train_wala_wan.sh
```

The corresponding configurations are `wala_robotwin.yaml` and
`wala_wan_robotwin.yaml`. Update the dataset path and
`trainer.pretrained_checkpoint` before launching distributed training. The Wan
configuration constructs a `384 x 320` mosaic from the head, left-wrist, and
right-wrist views and uses cross-attention state injection by default.

## Environment Setup

Use a current RoboTwin checkout and its bundled installation script. WALA uses
a two-environment workflow: the policy server runs in the `WALA` environment,
and the RoboTwin evaluator runs in the `RoboTwin` environment.

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

cd RoboTwin
bash script/_install.sh
python script/update_embodiment_config_path.py
```

The current evaluator expects the cuRobo `v0.7.8` API. If the RoboTwin
installer fetched a newer release, replace it and rebuild the extension:

```bash
cd /path/to/RoboTwin
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
own requirements because WALA's state/action transforms use it.

After RoboTwin is installed, prepare the evaluation-side dependencies:

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
cd /path/to/WALA
pip install -r examples/Robotwin/eval_files/requirements.txt
```

WALA passes the checkpoint path to RoboTwin through `--policy_ckpt_path`.
Check whether your RoboTwin revision already supports this argument:

```bash
grep -n "policy_ckpt_path" "$ROBOTWIN_PATH/script/eval_policy.py"
```

If matches are printed, no source change is required. Otherwise, apply the
following compatibility patch to `script/eval_policy.py`. This patch follows
the RoboTwin setup used by starVLA and is documented here because RoboTwin is a
separate third-party repository.

```diff
diff --git a/script/eval_policy.py b/script/eval_policy.py
--- a/script/eval_policy.py
+++ b/script/eval_policy.py
@@
     policy_name = usr_args["policy_name"]
     instruction_type = usr_args["instruction_type"]
+    policy_ckpt_path = usr_args["policy_ckpt_path"]
     save_dir = None
@@
     args['task_name'] = task_name
     args["task_config"] = task_config
     args["ckpt_setting"] = ckpt_setting
+    args["policy_ckpt_path"] = policy_ckpt_path
@@
 def parse_args_and_config():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, required=True)
     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
@@
     with open(args.config, "r", encoding="utf-8") as f:
         config = yaml.safe_load(f)
+    config["policy_ckpt_path"] = args.policy_ckpt_path
```

Without this argument, RoboTwin cannot forward the selected WALA checkpoint to
`model2robotwin_interface_joint.py`.

## Evaluation

Use `start_eval.sh` as the main RoboTwin evaluation launcher. It starts the
WALA policy server, waits for the server to become ready, launches RoboTwin
evaluation jobs, writes logs, and cleans up child processes on exit.

Example for evaluating all 50 tasks in the Easy setting:

```bash
conda activate WALA
export ROBOTWIN_PATH=/path/to/RoboTwin
bash examples/Robotwin/eval_files/start_eval.sh -m demo_clean -n full_eval -s 42 -j 2 -c results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt -p 6666 all
```

Example for evaluating one task:

```bash
bash examples/Robotwin/eval_files/start_eval.sh -m demo_clean -n single_task -s 42 -j 1 -c results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt -p 6666 adjust_bottle
```

Use `demo_clean` for the Easy setting and `demo_randomized` for the Hard
setting. `start_eval.sh` schedules jobs over the GPUs visible to the current
process. Set `CUDA_VISIBLE_DEVICES` before launching if you want to restrict
evaluation to a subset of GPUs; for example, the command below exposes only
GPUs 0, 1, 2, and 3 to the launcher:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

The checkpoint's saved `config.yaml` selects either the Qwen3-VL or Wan2.2
policy automatically.

Useful options:

```text
-s, --seed              Evaluation seed
-j, --jobs-per-gpu      Concurrent tasks per visible GPU
-p, --base-port         First policy-server port
--server-timeout        Policy-server startup timeout in seconds
```

Logs are written under the checkpoint directory by default:

```text
<ckpt_dir>/robotwin_eval_logs/<name>_<mode>_<ckpt_stem>_<timestamp>/
```

### Split PPU/H20 evaluation

RoboTwin evaluation can be split across two machines: PPU devices serve WALA
inference, while an H20 machine renders the simulator environments. Both
machines need a WALA checkout and access to the evaluated checkpoint. The H20
copy is also used to read the model configuration and normalization statistics.

On the PPU machine, launch a pool of policy servers from the WALA repository:

```bash
cd /path/to/WALA

GPU_IDS=0,1,2,3 JOBS_PER_GPU=2 \
bash examples/Robotwin/eval_files/run_robotwin_policy_servers_ppu.sh \
    results/Checkpoints/<robotwin_run_id>/checkpoints/steps_xxx_pytorch_model.pt \
    6666
```

`GPU_IDS` selects the PPU devices and `JOBS_PER_GPU` controls the number of
independent model-server processes on each device. Ports are assigned
consecutively from the second positional argument. With the settings above,
the script starts eight servers on ports `6666` through `6673`. Servers are
loaded in four-device batches to limit host-memory peaks. A Wan policy server
uses about 24 GiB of device memory with the provided checkpoint, so two servers
per 96 GiB PPU provide a conservative balance of throughput and memory headroom.
Server logs are written by default to:

```text
<ckpt_dir>/robotwin_ppu_server_logs/<timestamp>/
```

On the H20 machine, use `run_robotwin_eval_h20.sh` to establish all SSH port
forwards and start the RoboTwin clients:

```bash
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin

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

The PPU command above provides eight model servers, while the H20 command runs
16 environment jobs. The H20 launcher forwards the eight server ports and
assigns client slots to them in round-robin order. This keeps one model copy on
each server process and two copies on each PPU while allowing multiple RoboTwin
environments to run concurrently.
SSH key authentication is recommended for unattended evaluation.

Keep the PPU server command running for the duration of evaluation. Pressing
`Ctrl-C` on H20 stops the active RoboTwin clients and closes the SSH tunnel;
pressing `Ctrl-C` on PPU stops all policy servers started there. Tunnel settings
can also be provided with `ROBOTWIN_PPU_HOST`, `ROBOTWIN_PPU_SSH_PORT`,
`ROBOTWIN_PPU_USER`, and `ROBOTWIN_PPU_IDENTITY_FILE`.

The low-level scripts `run_policy_server.sh` and `eval.sh` are kept for manual
inspection. Use `start_eval.sh` for single-machine evaluation and the PPU
server launcher together with `run_robotwin_eval_h20.sh` for split PPU/H20
evaluation.

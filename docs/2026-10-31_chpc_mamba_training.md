# 2026-10-31 - Running Mamba training on Utah CHPC

Status: in-progress; instructions reviewed locally, not yet executed on CHPC.

Tags: #chpc #mamba #training #unrbl-ml-pipeline

Yes, this repository can run on CHPC through a single-GPU Slurm job. Once submitted, the job runs independently of the laptop and SSH connection. Start with a short benchmark, then submit bounded training jobs and resume from completed epochs.

This guide adapts Leonardo's 2025-10-24 CHPC notes to the current TensorFlow/Keras Mamba implementation. The recorded account, CUDA test, and Slurm experience are useful starting points; current GPU access and environment compatibility still need checking. No CHPC connection, data transfer, or job submission was performed while writing this guide.

## Choose resources

Choose by accessible GPU type, queue delay, and preemption policy, rather than idle CPU count. Logging into Kingspeak does not itself allocate a GPU. CHPC lists GPU resources across Kingspeak, Notchpeak, Granite, and other clusters. Guest access to owner nodes can be preempted. [CHPC GPU guide](https://www.chpc.utah.edu/documentation/guides/gpus-accelerators.php)

Kingspeak is a reasonable first access point because it is already familiar. Its guide lists general GPU nodes and non-preemptible access to CHPC-owned nodes. Check the GPU and software compatibility rather than assuming it will outperform the laptop. [Kingspeak guide](https://www.chpc.utah.edu/documentation/guides/kingspeak.php)

Granite lists H100NVL and L40S GPUs. Normal `granite-gpu` access requires an allocation; freecycle access can be preempted. Granite also requires a QoS selection. [Granite guide](https://www.chpc.utah.edu/documentation/guides/granite.php)

From the laptop:

```bash
ssh u1524457@kingspeak.chpc.utah.edu
```

On the chosen cluster:

```bash
mychpc batch
sinfo -o '%P %a %l %D %G'
module spider miniforge3
```

Use the account/partition/QoS combination actually available to this user. `mychpc batch` is the CHPC-recommended way to discover those combinations. Do not copy `--account=george` from an older example without confirming access. [Granite environment instructions](https://www.chpc.utah.edu/documentation/guides/granite.php)

Initial resource request for this application: **one GPU, one process, four CPU cores, and 32 GiB host RAM**. These are benchmark starting values, not measured CHPC requirements. Requesting 32 tasks would launch the wrong execution topology for this single-process trainer; it has no distributed multi-GPU training route.

The older Lonepeak MATLAB script does not request a GPU and is not a suitable template unchanged. GPU jobs need an explicit GPU request, such as `--gres=gpu:1`, plus an eligible account and partition. [CHPC GPU request options](https://www.chpc.utah.edu/documentation/guides/gpus-accelerators.php)

## Prepare and transfer the project

Use a stable project directory visible from both login and compute nodes. Keep outputs and checkpoints there across jobs. Do not use a different per-job temporary data path: the current resume fingerprint includes absolute source paths. Keep a separate durable copy of results; scratch storage is not an archive.

The Mamba implementation may include local files not yet available in a published package or remote Git revision. Transfer the actual working tree. Do not copy the laptop's `.venv`, and do not install only the published package.

First, from the laptop repository, capture the currently installed core versions:

```bash
.venv/bin/python - <<'PY' > chpc-constraints.txt
from importlib.metadata import version
for name in ('tensorflow', 'keras', 'numpy', 'scipy', 'scikit-learn', 'matplotlib'):
    print(f'{name}=={version(name)}')
PY
```

This captures the current environment, not necessarily the environment used to create an older checkpoint. For strict reproduction, retain a full environment record alongside each run.

On CHPC, create a project directory on the lab's chosen storage. Replace the example path before running:

```bash
CHPC_PROJECT=/REPLACE_WITH_PROJECT_STORAGE/unrbl_ml_pipeline
mkdir -p "$CHPC_PROJECT/data/training_data"
```

Back on the laptop, set the same absolute destination path and transfer:

```bash
CHPC_PROJECT=/REPLACE_WITH_PROJECT_STORAGE/unrbl_ml_pipeline
rsync -av --exclude='.venv/' --exclude='.git/' --exclude='.ref/' \
  --exclude='outputs/' --exclude='data/' --exclude='__pycache__/' \
  ./ "u1524457@kingspeak.chpc.utah.edu:$CHPC_PROJECT/"
rsync -av data/training_data/ \
  "u1524457@kingspeak.chpc.utah.edu:$CHPC_PROJECT/data/training_data/"
```

Use Globus for larger datasets or transfers from the lab server, as in the previous SOP. CHPC recommends Globus for managed background transfers. [CHPC transfer services](https://www.chpc.utah.edu/documentation/data_services.php)

These examples assume the data is approved for the selected CHPC environment. Use the lab's designated storage and access arrangements.

## Create the training environment

Use **Python 3.12**, matching the laptop's training environment, rather than the Python 3.7 Ripple Summit inference environment. Mamba training does not require `tflite_runtime` or a PyTorch runtime.

The following assumes a user Miniforge module has already been installed as described in the [CHPC Miniforge instructions](https://www.chpc.utah.edu/documentation/software/user-installed-python/package-managers-self-install.php). If the module is missing, complete those instructions first. `module load` selects installed software; it does not install the software.

On CHPC, in Bash:

```bash
module use "$HOME/MyModules"
module load miniforge3/latest
conda create -n unrbl-mamba python=3.12 pip -y
conda activate unrbl-mamba
CHPC_PROJECT=/REPLACE_WITH_PROJECT_STORAGE/unrbl_ml_pipeline
cd "$CHPC_PROJECT"
python -m pip install -c chpc-constraints.txt -e '.[cuda]'
python -m pip check
python -m pip freeze > chpc-environment.txt
```

Recreate the module/activation setup inside every batch job. CHPC's guide recommends module-based activation rather than `conda init`. The Miniforge package manager named `mamba` is unrelated to the Mamba neural architecture. [CHPC environment setup](https://www.chpc.utah.edu/documentation/software/user-installed-python/package-managers-self-install.php)

This installation uses the package's CUDA extra and TensorFlow's pip CUDA dependencies. Do not also load arbitrary CUDA/cuDNN modules from the old notes. The allocated node's NVIDIA driver must support the selected TensorFlow/CUDA stack. If installation or GPU detection fails, resolve the version/driver combination before training; the pinned versions are not a guarantee of compatibility with every CHPC GPU. [TensorFlow Linux installation](https://www.tensorflow.org/install/pip)

## Save a Slurm script

Save this block as `train_mamba.slurm` in the project root. Account, partition, and any required QoS are supplied at submission, so this script does not assume a lab allocation. Submit from the project root and create `logs` before `sbatch` opens the log files.

The inline Python calls the existing `run_training()` API with the shared CLI parser and `verbose=2`, which keeps batch logs readable by printing completed-epoch summaries. It saves the same run artifacts as `testfile.py`.

```bash
#!/bin/bash
#SBATCH --job-name=unrbl-mamba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module use "$HOME/MyModules"
module load miniforge3/latest
conda activate unrbl-mamba
cd "$SLURM_SUBMIT_DIR"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.mpl-cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTRAOP_THREADS="$SLURM_CPUS_PER_TASK"
export TF_NUM_INTEROP_THREADS=1
mkdir -p "$MPLCONFIGDIR"

# Slurm assigns CUDA_VISIBLE_DEVICES; do not overwrite it.
nvidia-smi
srun python -u - "$@" <<'PY'
import sys
import tensorflow as tf
from unrbl_ml_pipeline.cli import parse_training_args, training_overrides
from unrbl_ml_pipeline.training import run_training

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise SystemExit('No TensorFlow GPU detected; refusing an unintended CPU run.')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('Python:', sys.version, 'TensorFlow:', tf.__version__, 'GPUs:', gpus, flush=True)
args = parse_training_args(default_model='mamba_decoder')
if args.config is None:
    raise SystemExit('Pass --config with the package-native config path.')
_, results = run_training(
    args.model,
    config_path=args.config,
    config=training_overrides(args),
    data_dir=args.data_dir,
    output_dir=args.output_dir,
    verbose=2,
)
print('Run directory:', results['run_dir'], flush=True)
PY
```

The embedded wrapper uses model/config/data/output and training overrides; it does not implement catalog listing or the optional extra `--save-path` copy. Use the normal CLI for those operations. Slurm handles submission, job allocation, and log files. [CHPC Slurm guide](https://www.chpc.utah.edu/documentation/software/slurm.php)

## Submit a short smoke test

Use the login node for submission, not for running the training process itself. Set account and partition from `mychpc batch`, and use the chosen cluster's login host. Add `--qos=YOUR_CONFIRMED_QOS` to the following submissions if required, especially on Granite.

```bash
CHPC_ACCOUNT=REPLACE_WITH_CONFIRMED_ACCOUNT
CHPC_PARTITION=REPLACE_WITH_CONFIRMED_GPU_PARTITION
mkdir -p logs
sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" \
  train_mamba.slurm --config config_mamba.jsonc \
  --loss mse --epochs 3 --allow-missing-timestamps
```

For a specific GPU model, add the confirmed `--gres=gpu:TYPE:1` option to `sbatch`. Confirm the partition wall-time limit before increasing `--time`. CHPC publishes [GPU job templates](https://github.com/chpc-uofu/chpc-myjobs-templates) for cluster-specific examples.

The local `config_mamba.jsonc` selects `mse_pinn_jacobian` with 20 warmup epochs. The explicit `--loss mse` above is necessary for a three-epoch smoke test; a regularized run with `epochs <= warmup_epochs` is intentionally rejected. Three MSE epochs validate setup and outputs, not the speed of the regularized objective.

## Benchmark the actual recipe, then extend it

After the smoke test succeeds, start a separate regularized run. This example requests eight hours, but only use it if the selected account/partition/QoS permits that wall time:

```bash
sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" \
  --time=08:00:00 train_mamba.slurm --config config_mamba.jsonc \
  --loss mse_pinn_jacobian --epochs 25 --allow-missing-timestamps
```

Keep the normal 20-epoch warmup for this experiment. Compare epochs 21-25 with the laptop's recent median of about **635 seconds per epoch**. Record GPU model, queue delay, epoch times, calibration delay, validation RMSE, and host-memory usage. Do not infer the full objective's cost from the MSE smoke test. A GPU upgrade's speedup is unmeasured until this benchmark runs.

For context, the laptop run `mamba_decoder_20261031T002403815998Z_29e75ec9` reached epoch 58, with best validation RMSE 0.2798 at epoch 55. Its recorded epoch times total about 8.3 hours, but checkpoint timestamps span about 17.9 hours. Multi-hour gaps suggest interruptions or delays outside the epoch timer; they do not establish a hardware bottleneck.

Choose a total epoch cap that fits the measured speed and job wall time, allowing time for preprocessing, calibration, final evaluation, and plots. The configured minimum for patience-based stopping is 145 epochs, but an explicit smaller maximum is supported. Do not begin with a blind 500-epoch request.

After the CHPC regularized benchmark completes, use its printed run directory below. This resumes the same recipe to **50 total epochs**, not 50 additional epochs:

```bash
CHPC_RUN=/REPLACE_WITH_CHPC_REGULARIZED_RUN_DIRECTORY
sbatch --account="$CHPC_ACCOUNT" --partition="$CHPC_PARTITION" \
  --time=08:00:00 train_mamba.slurm --config "$CHPC_RUN/config.json" \
  --resume-from "$CHPC_RUN/checkpoints" --epochs 50
```

Each submission creates a new output run directory. Resume the newest run's checkpoint on the next submission. Do not continue the MSE smoke checkpoint with a different objective: resume rejects recipe changes. Changing streams, TBPTT length, preprocessing, or data also requires a separate experiment under the current contract.

## Monitor and recover

```bash
squeue -u u1524457
scontrol show job JOB_ID
tail -f logs/unrbl-mamba-JOB_ID.out
tail -f logs/unrbl-mamba-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Replace `JOB_ID` with the number returned by `sbatch`. `sacct` includes step rows; inspect those for memory usage. To intentionally stop your job, use `scancel JOB_ID`. These are standard CHPC-supported Slurm operations. [Slurm commands](https://www.chpc.utah.edu/documentation/software/slurm.php)

The trainer writes an atomic checkpoint after each completed epoch. If wall time expires or a job is preempted, the unfinished epoch is lost; resume starts at the next epoch after the last committed checkpoint. There is no automatic signal-triggered mid-epoch save or automatic resubmission. Avoid launching two jobs from one checkpoint unless deliberately creating separate experiments.

During an unfinished run, inspect `checkpoints/latest.json` and the referenced epoch's `metadata.json` for progress/history. `model.keras`, aggregate reports, plots, and `performance.md` are produced when training returns normally; a time-limited interruption may leave only checkpoints and logs. Keep enough job time for final reporting.

Copy completed results back through Globus or rsync. Preserve the run's config, checkpoints, model, environment record, and Slurm logs together.

## Resuming the laptop checkpoint on CHPC

**Moving the checkpoint files alone is not sufficient today.** The current fingerprint includes config values, preprocessed arrays, and absolute source paths. The laptop's configured data directory is `/home/leonardo/unrbl_ml_pipeline/data/training_data`; an ordinary CHPC home or project path will differ.

Two possible routes:

1. Start a fresh CHPC run using the workflow above. Subsequent CHPC resumes work as long as the recipe, data, and absolute paths remain stable.
2. Preserve the exact laptop-visible paths inside a suitable container/bind-mount environment, or implement and validate path-portable fingerprinting before migration. Neither route is set up or validated by this guide. A symlink is not a reliable substitute because configuration resolution canonicalizes paths.

Do not edit checkpoint fingerprint strings to bypass the check. Keep the original checkpoint intact. Reuse the saved config and matching software versions; numerical results can differ across GPU hardware even with a compatible resume.

See the repository's [Mamba guide](mamba.md) for the training and resume contract, and [fingerprint implementation](../src/unrbl_ml_pipeline/training/mamba_train.py) for the current path dependence.

## Next actions

- [ ] Confirm an accessible GPU account/partition/QoS and project storage path.
- [ ] Transfer the working tree and KDFs; recreate the Python 3.12 environment.
- [ ] Submit the three-epoch MSE smoke test and confirm an allocated TensorFlow GPU.
- [ ] Run the regularized benchmark through epoch 25 and measure post-warmup speed.
- [ ] Extend in bounded jobs using the newest CHPC checkpoint.
- [ ] Address path portability separately if continuing the laptop checkpoint is required.

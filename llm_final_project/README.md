# LLM Final Project: Does Shorter Reasoning Narrow the Reasoning Boundary?

We train small language models with **GRPO** (reinforcement learning on math problems) under different length-control settings. For each setting we measure accuracy, answer length, and **pass@k** (how many problems the model can solve at least once in k tries).

This README gets your machine set up and confirms training works on your GPU.

---

## 1. Before you start

Run this on your server:

```bash
nvidia-smi
```

Check two things in the top line:

- **Driver Version** must be **580 or newer**. Our setup uses CUDA 13.0, which older drivers can't run. If yours is older, ask your lab admin to update it.
- Note your **GPU name** and **memory** (e.g. `16376MiB` = 16 GB). We use this for the GPU plan in section 6.

You also need **conda** (Miniconda or Anaconda). Check with `conda --version`.

---

## 2. Get the code

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/henrywu983/llm_final_project.git
cd llm_final_project
```

---

## 3. Create the environment

```bash
conda create -n grpo python=3.11 -y
conda activate grpo
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The `--extra-index-url` part is needed because our PyTorch build (CUDA 13.0) is hosted on PyTorch's site, not the default pip index.

Installation can take 5–15 minutes.

---

## 4. Check the setup

**GPU check:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see something like `2.13.0+cu130 True NVIDIA GeForce RTX ...`. If it says `False`, see Troubleshooting.

**Library check:**

```bash
python -c "import vllm, unsloth, trl; print('vllm', vllm.__version__, '| trl', trl.__version__)"
```

Unsloth prints a few banner lines and some `deprecated` warnings. Those are normal. The last line should show the versions.

---

## 5. Run the smoke test

This trains a tiny model (Qwen2.5-0.5B) for 30 steps on 200 GSM8K math problems. It only checks that training runs end to end; the results don't matter.

```bash
python grpo_smoke_test.py
```

- **Time:** a few minutes; the first run also downloads the model and dataset.
- **What to look for:** one log line per step with a `reward` value. Non-zero rewards mean everything works.
- **Done when you see:** `Smoke test finished. LoRA adapter saved to outputs/smoke_test/lora`

**Post a screenshot of the last few log lines in the group chat** so we know your machine is ready.

If you get an out-of-memory error on an 8 GB card, open `grpo_smoke_test.py` and change `gpu_memory_utilization=0.5` to `0.4`, and `MAX_COMPLETION_LEN = 384` to `256`.

---

## 6. Running the experiments

First install the extra packages the experiment code needs (once):

```bash
pip install math-verify pandas matplotlib pytest
python -m pytest -q tests/        # 31 checks, CPU only, about 10 seconds
```

### The arms

| Arm | Config | What changes |
| --- | --- | --- |
| 1 | `configs/arm1_grpo.yaml` | Standard GRPO (control) |
| 2 | `configs/arm2_dr_grpo.yaml` | Dr. GRPO: constant loss normalization, no reward std scaling |
| 3 | `configs/arm3_length_penalty.yaml` | Standard GRPO + length penalty (Arora & Zanette) |
| 4 | `configs/arm4_da_gate.yaml` | Our difficulty-aware (DA) penalty: arm 3's penalty, switched off on hard questions (fewer than `tau` = 50% of the 8 answers correct) |
| 5 | `configs/arm5_ddca_scaled.yaml` | DDCA-inspired: arm 3's penalty multiplied by the share of correct answers (fades smoothly on hard questions) |
| 6 | `configs/arm6_ddca.yaml` | DDCA (Peng et al., 2026): difficulty-scaled length advantage among correct answers + RLOO accuracy advantage, built into the reward (`scale_rewards: none`, correctness × 8/7) |

All shared settings (model, data, steps, batch, lengths, LoRA, evaluation) live in `configs/base.yaml`. **Don't edit it for your own run**, since every arm must use the same values.

### Step 1: check your setup without training

```bash
python train.py --config configs/arm1_grpo.yaml --dry-run
```

This loads the training data and prints the method settings (`loss_type`, `scale_rewards`, batch layout). Nothing is trained.

### Step 2: a 5-step test run

```bash
python train.py --config configs/arm1_grpo.yaml --suffix _test \
    --set train.max_steps=5 --set train.save_steps=5
```

The `_test` suffix keeps it out of the real run's folder. Delete `outputs/arm1_grpo_seed0_test/` afterwards.

### Step 3: the real run (inside `tmux`)

```bash
tmux new -s <your arm name>
python train.py --config configs/<your arm>.yaml --seed 0
```

```bash
# reconnects to it later.
tmux attach -t <your arm name>

# List all running sessions
tmux ls

# Detach, training keeps running:
Ctrl+b, then d

# Stop the training run:
Ctrl+c (inside the session)

# Stop a finished or stuck session
tmux kill-session -t <arm_name>
```

- **Out of memory?** Add `--set machine.gpu_memory_utilization=0.35`. This only changes memory use, not results.
- **Stopped halfway?** Rerun the same command with `--resume`.
- **Live W&B charts:** add `--set logging.report_to=wandb`. Run `wandb login` once first.

Any `--set` outside `machine.` or `logging.` changes the experiment. The script warns you and records it in `run.json`.

### What gets saved (`outputs/<arm>_seed<seed>/`)

| File | Contents |
| --- | --- |
| `config.yaml`, `run.json` | Exact settings, command, git commit, GPU, start/end time, status |
| `pip_freeze.txt`, `nvidia_smi.txt`, `git_diff.patch` | Package versions, GPU state, uncommitted code changes |
| `trainer_config.json` | The exact TRL settings used |
| `train_manifest.jsonl.gz` | Every training problem, in order |
| `metrics.csv` | Every step: loss, rewards (each part), entropy, KL, clip ratios, lengths, truncation, grad norm, lr, GPU memory |
| `rollout_stats.csv` | Every step: accuracy, length stats, truncation, share of questions all-right or all-wrong |
| `rollouts.jsonl.gz` | Every answer written during training, with its length, correctness and rewards |
| `checkpoints/checkpoint-N/` | LoRA adapter every 50 steps (all kept) |
| `final_lora/`, `train.log` | Final adapter, full terminal output |

### Moving a run to the evaluation server

All evaluation runs on **one machine** so every arm is compared against the same base-model results (two servers gave different base numbers). Training can happen anywhere. After a run finishes, copy it to the evaluation server **through your own laptop**: lab servers usually can't log into each other, but your laptop can reach both.

Run these in a terminal **on your laptop** (macOS/Linux terminal, or WSL on Windows), not on a server.

- **`TRAIN_HOST` / `EVAL_HOST`:** what you type after `ssh` to log into that server, e.g. `henry@140.113.x.x`, or a short name from your `~/.ssh/config` (the names VS Code Remote-SSH shows).
- **`<run>`:** the run folder name, e.g. `arm3_length_penalty_seed0`.

**If you trained the run: download it from the training server to your laptop**

```bash
mkdir -p ~/llm_runs
rsync -av --progress --exclude=eval/ --exclude=rollouts.jsonl.gz --exclude=optimizer.pt TRAIN_HOST:projects/llm_final_project/llm_final_project/outputs/<run>/ ~/llm_runs/<run>/
```

This skips old evaluation results, the large rollout log and optimizer states, which evaluation doesn't need. It's a few hundred MB.

**If you evaluate: upload it from your laptop to the evaluation server**

```bash
rsync -av --progress ~/llm_runs/<run>/ EVAL_HOST:projects/llm_final_project/llm_final_project/outputs/<run>/
```

Then, on the evaluation server, check it arrived:

```bash
ls outputs/<run>/checkpoints     # checkpoint-50 ... checkpoint-500
ls outputs/<run>/final_lora      # adapter_config.json, adapter_model.safetensors
```

Notes:
- **Write each command on one line**, or end a line with `\` and nothing after it. A `\` in the middle of a line breaks the command (`rsync error: syntax or usage error`).
- **The trailing `/` on both paths matters:** it copies the folder's contents into `<run>/` instead of nesting `<run>/<run>/`.
- **`Permission denied`:** wrong username or password for that server, or the server only accepts SSH keys. Test with `ssh TRAIN_HOST` first.
- **No rsync (Windows without WSL):** use `scp -r` with the same paths. It can't skip files, so it copies more.

### Step 4: evaluate

```bash
# Once, for the untrained base model (AES compares against it):
python evaluate.py --base --tag quick
python evaluate.py --base --tag boundary

# Accuracy at every checkpoint (4 samples per problem, all problems):
python evaluate.py --run outputs/arm1_grpo_seed0 --tag quick --checkpoints all

# pass@k up to 256 (200 problems per benchmark), at chosen checkpoints:
python evaluate.py --run outputs/arm1_grpo_seed0 --tag boundary --checkpoints 100,300,final
```

Every sampled answer is saved in `<run>/eval/<tag>/<checkpoint>/<benchmark>.jsonl.gz`, so any metric can be recomputed later without the GPU. `boundary` is expensive (about 50k answers per benchmark per checkpoint), so run it on the evaluation GPUs. To evaluate a run trained on another server, copy it over first (see *Moving a run to the evaluation server* above).

### Step 5: results

```bash
python analyze.py --tag boundary --plots
```

This writes `results/boundary/`:
- `summary.csv`: pass@k, mean tokens, truncation, AES, per run, checkpoint and benchmark
- `summary_by_arm.csv`: mean and std over seeds
- `by_level.csv`: MATH500 results by difficulty level
- Plots: pass@k curves, metrics over training, training curves

## 7. GPU plan

Training needs the bigger GPUs; evaluation runs fine on smaller ones.

| GPU | VRAM | Role |
| --- | --- | --- |
| RTX 5090 (葉又齊) | 32 GB | Training: later arm 4 |
| RTX 4090 (葉又齊) | 24 GB | Training: arm 3 |
| RTX 4080 (吳俞亨) | 16 GB | Training: arm 1, and 2 |
| RTX 5070 (吳俞亨) | 12 GB | Evaluation, test runs |
| RTX 3060 () | 12 GB | Evaluation |
| RTX 4060 (陳冠瑄) | 8 GB | Evaluation (lower `--gpu-memory-utilization` if needed) |
| RTX 5060 (簡逸杰) | 8 GB | Evaluation, test runs |

Who runs what will be finalized in the group chat.

---

## 8. Daily workflow

**Every time you connect:**

```bash
conda activate grpo
cd ~/projects/llm_final_project
git pull                 # get teammates' latest changes
nvidia-smi               # make sure nobody else is using the GPU
```

**For long runs, use `tmux`** so training keeps going after you disconnect:

```bash
tmux new -s grpo         # start a session; activate conda and run your script inside it
# press Ctrl+b, then d   → detach (the job keeps running)
tmux attach -t grpo      # reconnect later to check on it
```

---

## 9. Team rules

- **Same settings for every training arm.** Batch size, group size, max length, number of steps and random seed all come from the shared config. Don't change them on your own, even if your GPU has room for more. Only the method being tested should differ between arms.
- **Don't commit outputs.** Model checkpoints and logs stay on your server (`outputs/` is in `.gitignore`). Share result numbers or plots instead.
- **Pull before you push**, and describe your change in the commit message.
- If you install a new package, tell the group so we can add it to `requirements.txt`.

---

## 10. Troubleshooting

| Problem | Fix |
| --- | --- |
| `torch.cuda.is_available()` prints `False` | Driver older than 580, or the environment isn't active. Run `conda activate grpo` and check `nvidia-smi`. |
| `No matching distribution found for torch==...+cu130` | You forgot `--extra-index-url https://download.pytorch.org/whl/cu130` in step 3. |
| Error mentioning a path like `@ file:///...` | Delete that line from `requirements.txt` and reinstall. |
| `CUDA out of memory` | Lower `gpu_memory_utilization` and `MAX_COMPLETION_LEN` (see section 5), and check nobody else is using the GPU. |
| `no kernel image is available` on RTX 50-series | Your PyTorch is too old for the card. Reinstall with the cu130 index from step 3. |
| `vLLM >= 0.28 moved bitsandbytes out of tree` | Run `pip install vllm-bnb-plugin`. |
| `SyntaxError` inside `cutlass/_mlir/.../_cute_nvgpu_ops_gen.py` | Corrupted download. Run `pip install --force-reinstall --no-deps --no-cache-dir nvidia-cutlass-dsl-libs-base==4.7.1`. If it persists, `pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base` (optional feature). |
| `FlashInfer ... SM 12.x requires CUDA >= 12.9` (RTX 50-series) | The server's system `nvcc` is too old. Install a newer one inside the env: `conda install -y -c nvidia cuda-nvcc=13.0`, then `conda env config vars set CUDA_HOME=$CONDA_PREFIX`, re-activate the env, run `rm -rf ~/.cache/flashinfer`, and rerun. Check with `nvcc --version` (should say 13.0). |
| Anything else | Paste the full error in the group chat. |

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

## 6. GPU plan

Training needs the bigger GPUs; evaluation (pass@k sampling) runs fine on smaller ones.

| GPU | VRAM | Role |
| --- | --- | --- |
| RTX 5090 | 32 GB | Training: arm 1 (vanilla GRPO), then arm 5 |
| RTX 4090 | 24 GB | Training: arm 3 (uniform length penalty) |
| RTX 4080 | 16 GB | Training: arm 4 (difficulty-aware penalty) |
| RTX 5070 | 12 GB | Training arm 2 (Dr. GRPO) if it fits, else evaluation |
| RTX 3060 | 12 GB | Evaluation |
| RTX 4060 | 8 GB | Evaluation, data prep |
| RTX 5060 | 8 GB | Evaluation, small debug runs |

Who runs what will be finalized in the group chat.

---

## 7. Daily workflow

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

## 8. Team rules

- **Same settings for every training arm.** Batch size, group size, max length, number of steps and random seed all come from the shared config. Don't change them on your own, even if your GPU has room for more. Only the method being tested should differ between arms.
- **Don't commit outputs.** Model checkpoints and logs stay on your server (`outputs/` is in `.gitignore`). Share result numbers or plots instead.
- **Pull before you push**, and describe your change in the commit message.
- If you install a new package, tell the group so we can add it to `requirements.txt`.

---

## 9. Troubleshooting

| Problem | Fix |
| --- | --- |
| `torch.cuda.is_available()` prints `False` | Driver older than 580, or the environment isn't active. Run `conda activate grpo` and check `nvidia-smi`. |
| `No matching distribution found for torch==...+cu130` | You forgot `--extra-index-url https://download.pytorch.org/whl/cu130` in step 3. |
| Error mentioning a path like `@ file:///...` | Delete that line from `requirements.txt` and reinstall. |
| `CUDA out of memory` | Lower `gpu_memory_utilization` and `MAX_COMPLETION_LEN` (see section 5), and check nobody else is using the GPU. |
| `no kernel image is available` on RTX 50-series | Your PyTorch is too old for the card. Reinstall with the cu130 index from step 3. |
| Anything else | Paste the full error in the group chat. |

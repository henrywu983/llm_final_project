"""
GRPO smoke test: Qwen2.5-0.5B-Instruct on a small GSM8K subset.

Goal: confirm that loading, vLLM generation, reward functions and GRPO
training all run end to end on one GPU. NOT a real experiment.

Run:
    conda activate grpo
    python grpo_smoke_test.py

Expected: ~30 training steps, a log line every step. "reward" should be
non-zero and ideally creep up; "completion_length" shows answer length.
"""

import re

from unsloth import FastLanguageModel  # import unsloth before trl/transformers
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------- settings
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 1024
LORA_RANK = 16
N_TRAIN = 200          # small subset, enough for a smoke test
MAX_STEPS = 30
NUM_GENERATIONS = 4    # group size G
MAX_PROMPT_LEN = 256
MAX_COMPLETION_LEN = 384
OUTPUT_DIR = "outputs/smoke_test"

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "Put the final numeric answer in \\boxed{}."
)

# ---------------------------------------------------------------- model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,          # same setting as the real runs
    fast_inference=True,        # use vLLM for generation
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.5, # leave room for training; lower if OOM
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ---------------------------------------------------------------- data
def gsm8k_answer(text: str) -> str:
    """GSM8K solutions end with '#### <answer>'."""
    return text.split("####")[-1].strip().replace(",", "")


def to_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": gsm8k_answer(example["answer"]),
    }


dataset = (
    load_dataset("openai/gsm8k", "main", split="train")
    .shuffle(seed=42)
    .select(range(N_TRAIN))
    .map(to_prompt)
)

# ---------------------------------------------------------------- rewards
def completion_text(completion) -> str:
    """Completions are chat messages for chat prompts, plain strings otherwise."""
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    boxed = BOXED.findall(text)
    candidate = boxed[-1] if boxed else text
    numbers = NUMBER.findall(candidate.replace(",", ""))
    return numbers[-1] if numbers else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def correctness_reward(prompts, completions, answer, **kwargs):
    """1.0 if the final answer matches GSM8K's answer, else 0.0."""
    return [
        1.0 if same_number(extract_answer(completion_text(c)), a) else 0.0
        for c, a in zip(completions, answer)
    ]


def format_reward(prompts, completions, **kwargs):
    """Small bonus for using \\boxed{} so answers are easy to grade."""
    return [0.1 if BOXED.search(completion_text(c)) else 0.0 for c in completions]


# ---------------------------------------------------------------- training
config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-6,
    per_device_train_batch_size=NUM_GENERATIONS,  # one prompt's group per step
    gradient_accumulation_steps=1,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    max_steps=MAX_STEPS,
    logging_steps=1,
    save_steps=MAX_STEPS,
    temperature=1.0,
    report_to="none",       # switch to "wandb" for real runs
    seed=42,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=config,
    train_dataset=dataset,
)

trainer.train()

model.save_lora(f"{OUTPUT_DIR}/lora")
print(f"\nSmoke test finished. LoRA adapter saved to {OUTPUT_DIR}/lora")

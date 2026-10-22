# ===========================================================================
# Shared settings for ALL training arms.
#
# Fairness rule: every value here must be identical across arms. An arm file
# (arm*.yaml) only changes the method being tested. The only settings you may
# change per machine are under `machine:` (via --set machine.xxx=...).
# ===========================================================================

experiment:
  output_root: outputs        # each run goes to outputs/<name>_seed<seed>/

model:
  name: unsloth/Qwen2.5-1.5B-Instruct
  load_in_4bit: true          # QLoRA; keep the same for every arm
  max_seq_length: 1536        # prompt + completion tokens
  lora_rank: 32
  lora_alpha: 64
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

data:
  # Training uses the official TRAIN splits only. Test sets are never trained on.
  train_sources: [gsm8k, math]
  gsm8k:
    id: openai/gsm8k
    config: main
    split: train
  math:
    id: DigitalLearningGmbH/MATH-lighteval   # 7,500 MATH train problems
    config: default
    split: train
    levels: [1, 2, 3, 4, 5]
  max_train_examples: null    # null = use everything
  shuffle_seed: 1234          # fixed, so every arm sees the same data order

prompt:
  instruction: "Please reason step by step, and put your final answer within \\boxed{}."

grpo:
  loss_type: grpo             # set explicitly: TRL 1.15's default is "dapo"
  scale_rewards: group        # group | batch | none
  beta: 0.0                   # KL coefficient (0 = no reference model)
  epsilon: 0.2
  epsilon_high: null          # null = same as epsilon (no Clip-Higher)
  num_generations: 8          # group size G (answers per question)
  prompts_per_step: 8         # questions per optimizer step -> 64 answers/step
  max_completion_length: 1024
  temperature: 1.0
  top_p: 1.0
  mask_truncated_completions: false

train:
  learning_rate: 1.0e-5
  lr_scheduler_type: constant_with_warmup
  warmup_steps: 10
  max_steps: 500
  max_grad_norm: 1.0
  weight_decay: 0.0
  optim: adamw_8bit
  per_device_train_batch_size: 8
  save_steps: 50              # a checkpoint every 50 steps, all kept
  logging_steps: 1

reward:
  correctness: 1.0            # reward for a correct final answer
  format_bonus: 0.1           # small bonus for using \boxed{}
  length_penalty:
    enabled: false
    type: arora_zanette       # arora_zanette | linear | kimi (see src/length_penalty.py)
    coef: 0.2

logging:
  report_to: none             # set to "wandb" to also log to Weights & Biases
  wandb_project: llm-final-grpo
  log_rollouts: true          # save every training answer to rollouts.jsonl.gz

machine:
  # Only affects memory use, not results. Lower it if you run out of memory
  # or share the GPU, e.g. --set machine.gpu_memory_utilization=0.35
  gpu_memory_utilization: 0.5

eval:
  model_name: unsloth/Qwen2.5-1.5B-Instruct   # 16-bit base model used for evaluation
  benchmarks:
    gsm8k:
      id: openai/gsm8k
      config: main
      split: test
    math500:
      id: HuggingFaceH4/MATH-500
      split: test
    amc23:
      id: math-ai/amc23
      split: test
  temperature: 0.6
  top_p: 0.95
  max_tokens: 2048
  seed: 0
  subset_seed: 0              # fixed, so every checkpoint/arm sees the same problems
  gpu_memory_utilization: 0.85
  # Two evaluation modes (choose with --tag):
  quick:                      # cheap accuracy check, every checkpoint
    n_samples: 4
    max_problems: null        # null = all problems
  boundary:                   # pass@k up to 256, a few checkpoints
    n_samples: 256
    max_problems: 200         # MATH500 subset is stratified by difficulty level
  boundary_full:              # like boundary, but every test problem (about 3.5x the cost)
    n_samples: 256
    max_problems: null

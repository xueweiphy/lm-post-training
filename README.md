# GRPO from scratch on GSM8K

Reinforcement learning with verifiable rewards, written function by function: a 1B base model
(OLMo-2-0425-1B) learns to solve grade-school math problems from nothing but a 0/1 grader on its
final answer. Follows the structure of Stanford's CS336 post-training assignment, done as a
self-study project.

Plan: prompting baselines → GRPO → RL variants (RFT, Dr. GRPO, MaxRL) → off-policy GRPO with
clipping → (later) instruction-tuning SFT and DPO.

Compute: UF HiPerGator (B200 / RTX PRO 6000); one GPU for the baselines, two for GRPO training.

## Results so far

Prompting baselines, OLMo-2-0425-1B, GSM8K test (1319 questions), temperature 1.0, top-p 1.0, max 512 tokens:

| prompt | correct | format ok, wrong | format wrong | accuracy |
|---|---|---|---|---|
| question_only | 3 | 214 | 1102 | 0.2 % |
| r1_zero | 1 | 819 | 499 | 0.1 % |
| r1_zero, 3-shot | 244 | 1007 | 68 | 18.5 % |

The same 3-shot prompt with greedy decoding (first 100 questions) gets 38 % accuracy with 100 % format
compliance, so much of the gap is sampling noise from a small model at temperature 1.0 rather than
missing ability. The r1_zero row (≈0 %) is the starting point GRPO has to improve.

## GRPO status

`grpo.py` holds the training components, each checked against the course's reference tests:

| function | what it does | status |
|---|---|---|
| `tokenize_prompt_and_output` | prompt + response → `input_ids`, shifted `labels`, response mask | done |
| `get_response_log_probs` | per-token log p(label) and entropy from a causal LM | done |
| `compute_rollout_rewards` | grader rewards for a batch of rollouts | done |
| `compute_group_normalized_rewards` | advantages: group mean baseline, std normalizer | done |
| `compute_policy_gradient_loss` | per-token loss `-A · log π` (on-policy) | done |
| `aggregate_loss_across_microbatch` | masked mean over tokens, then over sequences (or / constant) | done |
| `grpo_train_step` | microbatched gradient accumulation, grad clipping, logging | done |
| training loop (`grpo_train.py`) | vLLM rollouts + weight sync, validation every 10 steps, JSON-lines log | written, first GPU run next |

## Layout

| file | what |
|---|---|
| `grpo.py` | the GRPO components (table above) |
| `grpo_train.py` | GRPO training loop: policy on GPU 0, vLLM on GPU 1 → `results/grpo_seed<seed>.jsonl` |
| `prompting_baselines.py` | generate answers for the three prompts (GPU) → `results/<prompt>.jsonl` |
| `grade_baselines.py` | grade with the Dr. GRPO grader, count categories → `results/<prompt>.graded.jsonl` |
| `inspect_baselines.py` | print random category-2 / category-3 examples for reading |
| `check_greedy.py` | greedy 3-shot sanity check on 100 questions |
| `run_gpu.sh` | Slurm job: `sbatch run_gpu.sh <script.py> [args]` |
| `push.sh`, `pull.sh` | copy files Mac ↔ HiPerGator |
| `drgrpo_grader.py`, `vllm_utils.py`, `checkpoint.py`, `prompts/` | grader, vLLM server + weight sync, model loading, prompts — course starter code, unchanged |
| `data/gsm8k/` | GSM8K train / test |
| `results/*.graded.jsonl` | the graded baseline outputs behind the table above |

## Run

```
uv sync --extra gpu                          # on a GPU node, once
sbatch run_gpu.sh prompting_baselines.py     # generation (GPU)
uv run python grade_baselines.py             # grading (no GPU)
sbatch -p hpg-b200 --gres=gpu:b200:2 --time=03:00:00 run_gpu.sh grpo_train.py   # GRPO (2 GPUs)
```

## Credits

Grader, vLLM utilities, prompts and data come from
[stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment).

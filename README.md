# GRPO from scratch on GSM8K

Reinforcement learning with verifiable rewards, written function by function: a 1B base model
(OLMo-2-0425-1B) learns to solve grade-school math problems from nothing but a 0/1 grader on its
final answer. Follows the structure of Stanford's CS336 post-training assignment, done as a
self-study project.

Plan: prompting baselines → GRPO → RL variants (RFT, Dr. GRPO, MaxRL) → off-policy GRPO with
clipping → (later) instruction-tuning SFT and DPO.

Compute: UF HiPerGator (B200 / RTX PRO 6000); one GPU for the baselines, two for GRPO training.

## Results so far

### GRPO (standard, on-policy)

OLMo-2-0425-1B on GSM8K with the r1_zero prompt, handout hyperparameters (lr 1e-5, 200 steps,
32 questions × 8 rollouts per step, gradient clipping 1.0). Validation on 1024 GSM8K test questions.

| seed | final val accuracy | best (step) | final response length (tokens) |
|---|---|---|---|
| 0 | 12.5 % | 12.5 % (200) | 28 |
| 1 | 18.4 % | 18.4 % (200) | 55 |
| 2 | **42.8 %** | 44.1 % (180) | 132 |
| 3 | 6.1 % | 8.2 % (130) | 13 |
| **mean ± sd** | **19.9 ± 16.0 %** | | |

Starting point: 0.1 % (r1_zero baseline below). An earlier run with seed 0 and the same code reached 39.8 %.

![GRPO, 4 seeds](results/grpo_seeds.png)

What the curves show:

- **Two phases.** In the first ~50 steps every run learns the answer format (format reward 0.5 → 0.95)
  and its responses collapse from ~110 to 13–36 tokens: `<think> 16 </think> <answer> 24 </answer>` —
  skip the reasoning, guess a number. Accuracy plateaus near 7 %. Some runs then *escape*: response
  length and accuracy rise together as longer reasoning starts to earn reward.
- **Escape timing is the variance.** Seed 2 escaped at ~step 40, the earlier seed-0 run at ~120, seed 1
  is just starting at 200, seeds 0 and 3 never do within 200 steps. The earlier run and seed 0 saw the
  same questions in the same order (same `random.seed`) yet ended at 40 % vs 12 % — the spread comes from
  rollout sampling, amplified by the RL feedback loop.
- **Depth of the collapse predicts the outcome**: minimum length ~36 tokens → best run; 13 → stuck.
- Entropy falls from ~1.0 to 0.1–0.3 nats per token; low entropy alone is not failure (the best run has the lowest).
- Likely cause of the short-answer pull: the per-sequence 1/|o| normalization gives a short correct
  answer a larger per-token update — the length bias Dr. GRPO removes (next experiment).

The handout target (≥ 25 % averaged over seeds) is not met at 200 steps. Next: longer runs and a learning-rate sweep.

### Prompting baselines

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
| training loop (`grpo_train.py`) | vLLM rollouts + weight sync, validation every 10 steps, JSON-lines log | done — 4 seeds run |

## Layout

| file | what |
|---|---|
| `grpo.py` | the GRPO components (table above) |
| `grpo_train.py` | GRPO training loop: policy on GPU 0, vLLM on GPU 1; `python grpo_train.py <seed>` → `results/grpo_seed<seed>.jsonl` (+ `_rollouts.jsonl`) |
| `run_seeds.sh` | submit the 4-seed runs (seeds 0–3) |
| `plot_grpo.py` | train / val reward and format reward vs step from one run log |
| `plot_seeds.py` | 8 metrics (val/train reward, format, length, entropy, loss, grad norm) for several runs → `results/grpo_seeds.png` |
| `prompting_baselines.py` | generate answers for the three prompts (GPU) → `results/<prompt>.jsonl` |
| `grade_baselines.py` | grade with the Dr. GRPO grader, count categories → `results/<prompt>.graded.jsonl` |
| `inspect_baselines.py` | print random category-2 / category-3 examples for reading |
| `check_greedy.py` | greedy 3-shot sanity check on 100 questions |
| `run_gpu.sh` | Slurm job: `sbatch run_gpu.sh <script.py> [args]` |
| `push.sh`, `pull.sh` | copy files Mac ↔ HiPerGator |
| `drgrpo_grader.py`, `vllm_utils.py`, `checkpoint.py`, `prompts/` | grader, vLLM server + weight sync, model loading, prompts — course starter code, unchanged |
| `data/gsm8k/` | GSM8K train / test |
| `results/*.graded.jsonl` | the graded baseline outputs behind the baseline table |
| `results/grpo_seed*.jsonl` | GRPO run logs (one line per step) and rollout samples (`_rollouts.jsonl`, every 40 steps) |

## Run

```
uv sync --extra gpu                          # on a GPU node, once
sbatch run_gpu.sh prompting_baselines.py     # generation (GPU)
uv run python grade_baselines.py             # grading (no GPU)
./run_seeds.sh                               # GRPO, 4 seeds (2 GPUs each)
python plot_seeds.py                         # plots from results/grpo_seed0..3.jsonl
```

## Credits

Grader, vLLM utilities, prompts and data come from
[stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment).

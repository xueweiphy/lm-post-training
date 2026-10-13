# Alignment: reasoning RL on GSM8K

My work on the CS336 (Stanford, Spring 2026) Assignment 5 alignment track, built up step by step:
prompting baselines → GRPO on OLMo-2-0425-1B → RL variants (RFT, Dr. GRPO, MaxRL) → off-policy GRPO,
plus the supplement (instruction-tuning SFT and DPO on Llama 3.1 8B).

Compute: one GPU on UF HiPerGator (B200 / RTX PRO 6000).

## Results so far

Prompting baselines, OLMo-2-0425-1B, GSM8K test (1319), temperature 1.0, top-p 1.0, max 512 tokens:

| prompt | correct | format ok, wrong | format wrong | accuracy |
|---|---|---|---|---|
| question_only | 3 | 214 | 1102 | 0.2 % |
| r1_zero | 1 | 819 | 499 | 0.1 % |
| r1_zero, 3-shot | 244 | 1007 | 68 | 18.5 % |

Same 3-shot prompt with greedy decoding, first 100 questions: 38 % accuracy, 100 % format.

## Layout

| file | what |
|---|---|
| `prompting_baselines.py` | generate answers for the three prompts (GPU) → `results/<prompt>.jsonl` |
| `grade_baselines.py` | grade with the Dr. GRPO grader, count categories → `results/<prompt>.graded.jsonl` |
| `inspect_baselines.py` | print random category-2 / category-3 examples for reading |
| `check_greedy.py` | greedy 3-shot sanity check on 100 questions |
| `run_gpu.sh` | Slurm job: `sbatch run_gpu.sh <script.py> [args]` |
| `push.sh`, `pull.sh` | copy files Mac ↔ HiPerGator |
| `alignment/grpo.py` | GRPO building blocks: `tokenize_prompt_and_output`, `get_response_log_probs`, `compute_rollout_rewards`, `compute_group_normalized_rewards`, `compute_policy_gradient_loss` (on-policy) |
| `tests/adapters.py` | connects the course tests to `grpo.py`: `python -m pytest -k <test name>` |
| `alignment/` (other files), `data/`, `tests/`, `pyproject.toml`, `uv.lock` | course starter code, data and tests (unchanged) |

## Run

```
uv sync --extra gpu                          # on a GPU node, once
sbatch run_gpu.sh prompting_baselines.py     # generation
uv run python grade_baselines.py             # grading, no GPU
```

## Credits

Starter code, prompts, grader, data and tests: [stanford-cs336/assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment).

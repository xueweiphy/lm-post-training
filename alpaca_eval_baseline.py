import json, time
from vllm_utils import VLLMServer

# load AlpacaEval: 805 instructions; keep 'instruction' and 'dataset' (the GPT-4 Turbo 'output' is the reference, not ours)
with open("data/alpaca_eval/alpaca_eval_gpt4_turbo.json") as f:
    eval_set = json.load(f)
examples = [{"instruction": ex["instruction"], "dataset": ex["dataset"]} for ex in eval_set]
print(len(examples), "examples")

# prompts: the instruction itself is the task prompt, placed inside the zero-shot system prompt
path = "prompts_safety/"
system_template = open(path + "zero_shot_system_prompt.prompt").read()
alpaca_template = open(path + "alpaca_eval_zero_shot.prompt").read()
prompts = []
for ex in examples :
    task = alpaca_template.format(instruction=ex["instruction"])
    prompts.append(system_template.format(instruction=task))

# generate (greedy) on one GPU
server = VLLMServer("meta-llama/Meta-Llama-3.1-8B", gpu=0)
server.start ()

# greedy; long limit because AlpacaEval answers are free-form paragraphs
params = {"temperature": 0.0, "max_tokens" : 2048, "n": 1, "seed": 0,
            "stop": ["# Query:"]}

t0 = time.time()
out = server.generate_completions ( prompts , params )
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

# serialize as a JSON array in the AlpacaEval format: instruction, output, generator, dataset
# the system prompt opens a ``` block for the answer, so drop the closing ``` the model writes
results = []
for ex, o in zip(examples, out):
    text = o.text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    results.append({"instruction": ex["instruction"], "output": text,
                    "generator": "llama-3.1-8b-base", "dataset": ex["dataset"]})
with open("results/alpaca_eval_baseline.json", "w") as fout:
    json.dump(results, fout, indent=2)
print(len(results), "outputs written to results/alpaca_eval_baseline.json")

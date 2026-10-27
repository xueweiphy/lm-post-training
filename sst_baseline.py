import json, csv, time
from vllm_utils import VLLMServer

# load SimpleSafetyTests: 100 harmful requests (id, harm_area, category, prompts_final)
with open("data/simple_safety_tests/simple_safety_tests.csv") as f:
    examples = list(csv.DictReader(f))
print(len(examples), "examples")

# prompts: the request itself is the task prompt, placed inside the zero-shot system prompt
path = "prompts_safety/"
system_template = open(path + "zero_shot_system_prompt.prompt").read()
sst_template = open(path + "simple_safety_tests_zero_shot.prompt").read()
prompts = []
for ex in examples :
    task = sst_template.format(instruction=ex["prompts_final"])
    prompts.append(system_template.format(instruction=task))

# generate (greedy) on one GPU
server = VLLMServer("meta-llama/Meta-Llama-3.1-8B", gpu=0)
server.start ()
params = {"temperature": 0.0, "max_tokens" : 1024, "n": 1, "seed": 0,
            "stop": ["# Query:"]}
t0 = time.time()
out = server.generate_completions ( prompts , params )
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

# serialize as JSON lines for scripts/evaluate_safety.py: needs prompts_final and output
with open("results/sst_baseline.jsonl", "w") as g:
    for ex, o in zip(examples, out):
        text = o.text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        g.write(json.dumps({"id": ex["id"], "harm_area": ex["harm_area"], "category": ex["category"],
                            "prompts_final": ex["prompts_final"], "output": text}) + "\n")
print(len(examples), "outputs written to results/sst_baseline.jsonl")

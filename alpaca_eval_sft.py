import json, os, sys, time
from vllm_utils import VLLMServer

# AlpacaEval outputs from the SFT model (instruction in the Alpaca template).   usage: python alpaca_eval_sft.py [model_dir]
model_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_lr2e-05"
tag = os.path.basename(model_dir.rstrip("/"))
port = 8000 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000

with open("data/alpaca_eval/alpaca_eval_gpt4_turbo.json") as f:
    eval_set = json.load(f)
examples = [{"instruction": ex["instruction"], "dataset": ex["dataset"]} for ex in eval_set]
print(len(examples), "examples")

path = "prompts_safety/"
sft_template = open(path + "alpaca_sft.prompt").read().rstrip("\n")
alpaca_template = open(path + "alpaca_eval_zero_shot.prompt").read()
prompts = [sft_template.format(instruction=alpaca_template.format(instruction=ex["instruction"]).strip(), response="") for ex in examples]

server = VLLMServer(model_dir, gpu=0, port=port)
server.start()
params = {"temperature": 0.0, "max_tokens": 2048, "n": 1, "seed": 0,
          "stop": ["Below is an instruction", "### Instruction:"]}
t0 = time.time()
out = server.generate_completions(prompts, params)
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

# own folder: alpaca_eval writes annotations.json next to the outputs, so the baseline's is not overwritten
os.makedirs(f"results/alpaca_eval_{tag}", exist_ok=True)
outfile = f"results/alpaca_eval_{tag}/alpaca_eval_{tag}.json"
results = [{"instruction": ex["instruction"], "output": o.text.strip(), "generator": f"llama-3.1-8b-{tag}", "dataset": ex["dataset"]}
           for ex, o in zip(examples, out)]
with open(outfile, "w") as fout:
    json.dump(results, fout, indent=2)
print(len(results), "outputs written to", outfile)

import json, csv, os, sys, time
from vllm_utils import VLLMServer

# SimpleSafetyTests outputs from the SFT model (request in the Alpaca template).   usage: python sst_sft.py [model_dir]
model_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_lr2e-05"
tag = os.path.basename(model_dir.rstrip("/"))
port = 8000 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000

with open("data/simple_safety_tests/simple_safety_tests.csv") as f:
    examples = list(csv.DictReader(f))
print(len(examples), "examples")

path = "prompts_safety/"
sft_template = open(path + "alpaca_sft.prompt").read().rstrip("\n")
sst_template = open(path + "simple_safety_tests_zero_shot.prompt").read()
prompts = [sft_template.format(instruction=sst_template.format(instruction=ex["prompts_final"]).strip(), response="") for ex in examples]

server = VLLMServer(model_dir, gpu=0, port=port)
server.start()
params = {"temperature": 0.0, "max_tokens": 1024, "n": 1, "seed": 0,
          "stop": ["Below is an instruction", "### Instruction:"]}
t0 = time.time()
out = server.generate_completions(prompts, params)
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

outfile = f"results/sst_{tag}.jsonl"
with open(outfile, "w") as g:
    for ex, o in zip(examples, out):
        g.write(json.dumps({"id": ex["id"], "harm_area": ex["harm_area"], "category": ex["category"],
                            "prompts_final": ex["prompts_final"], "output": o.text.strip()}) + "\n")
print(len(examples), "outputs written to", outfile)

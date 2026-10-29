import json, os, sys, time
from gsm8k import parse_gsm8k_response
from vllm_utils import VLLMServer

# GSM8K with the SFT model: task prompt inside the Alpaca template.   usage: python gsm8k_sft.py [model_dir]
model_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_lr2e-05"
tag = os.path.basename(model_dir.rstrip("/"))
port = 8000 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000

examples = []
with open("data/gsm8k/test.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        examples.append({"question": item["question"], "answer": item["answer"].split("####")[-1].strip()})
print(len(examples), "examples")

path = "prompts_safety/"
sft_template = open(path + "alpaca_sft.prompt").read().rstrip("\n")
gsm8k_template = open(path + "gsm8k_zero_shot.prompt").read()
prompts = [sft_template.format(instruction=gsm8k_template.format(question=ex["question"]).strip(), response="") for ex in examples]

server = VLLMServer(model_dir, gpu=0, port=port)
server.start()
params = {"temperature": 0.0, "max_tokens": 512, "n": 1, "seed": 0,
          "stop": ["Below is an instruction", "### Instruction:"]}
t0 = time.time()
out = server.generate_completions(prompts, params)
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

def same(pred, gold):   # numeric compare: "18" == "18.0", "1,000" == "1000"
    return pred is not None and float(pred) == float(gold.replace(",", ""))

n_correct, n_none = 0, 0
with open(f"results/gsm8k_{tag}.jsonl", "w") as g:
    for ex, o in zip(examples, out):
        pred = parse_gsm8k_response(o.text)
        n_correct += same(pred, ex["answer"])
        n_none += pred is None
        g.write(json.dumps(ex | {"response": o.text, "pred": pred, "correct": same(pred, ex["answer"])}) + "\n")
print(f"accuracy {n_correct / len(examples):.4f}  correct {n_correct}  failed parses {n_none}  total {len(examples)}")

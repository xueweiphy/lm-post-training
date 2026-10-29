import json, csv, glob, os, sys, time
from mmlu import parse_mmlu_response
from vllm_utils import VLLMServer

# MMLU with the SFT model: task prompt inside the Alpaca template used in training.   usage: python mmlu_sft.py [model_dir]
model_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_lr2e-05"
tag = os.path.basename(model_dir.rstrip("/"))                        # sft_lr2e-05
port = 8000 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000           # own port: jobs on one node don't kill each other's vLLM

files = sorted(glob.glob("data/mmlu/test/*.csv"))
examples = []
for file in files:
    sub = " ".join(file.split('/')[-1].split('.')[0].split('_')[:-1])
    with open(file, "r") as f:
        rows = list(csv.reader(f))
    for r in rows:
        examples.append({"subject": sub, "question": r[0], "options": r[1:5], "answer": r[5]})
print(len(files), "files", len(examples), "examples")

path = "prompts_safety/"
sft_template = open(path + "alpaca_sft.prompt").read().rstrip("\n")   # ends with "### Response:\n", as in training
mmlu_template = open(path + "mmlu_zero_shot.prompt").read()
prompts = []
for ex in examples:
    task = mmlu_template.format(subject=ex["subject"], question=ex["question"], options=ex["options"])
    prompts.append(sft_template.format(instruction=task.strip(), response=""))

server = VLLMServer(model_dir, gpu=0, port=port)
server.start()
params = {"temperature": 0.0, "max_tokens": 128, "n": 1, "seed": 0,
          "stop": ["Below is an instruction", "### Instruction:"]}   # same settings as the baseline; EOS ends a normal answer
t0 = time.time()
out = server.generate_completions(prompts, params)
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

n_correct, n_none = 0, 0
with open(f"results/mmlu_{tag}.jsonl", "w") as g:
    for ex, o in zip(examples, out):
        pred = parse_mmlu_response(ex, o.text)
        n_correct += pred == ex["answer"]
        n_none += pred is None
        g.write(json.dumps(ex | {"response": o.text, "pred": pred, "correct": pred == ex["answer"]}) + "\n")
print(f"accuracy {n_correct / len(examples):.4f}  correct {n_correct}  failed parses {n_none}  total {len(examples)}")

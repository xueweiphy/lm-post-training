import json, csv, glob, time
from mmlu import parse_mmlu_response
from vllm_utils import VLLMServer

# load MMLU test: 57 subject files, rows = question, A, B, C, D, answer
files = sorted(glob.glob("data/mmlu/test/*.csv"))
examples =[]
for file in files:
    sub = " ".join(file.split('/')[-1].split('.')[0].split('_')[:-1])
    with open(file, "r") as f:
        rows = list ( csv.reader (f ) )
    for r in rows :
        ex = {"subject" : sub, "question": r[0], "options": r[1:5], "answer": r[5] }
        examples.append ( ex )
print(len(files), "files", len(examples), "examples")

# prompts: MMLU task prompt inside the zero-shot system prompt
path = "prompts_safety/"
system_template = open(path + "zero_shot_system_prompt.prompt").read()
mmlu_template = open(path + "mmlu_zero_shot.prompt").read()
prompts = []
for ex in examples :
    task = mmlu_template.format(subject=ex["subject"], question=ex["question"], options=ex["options"])
    prompts.append(system_template.format(instruction=task))

# generate (greedy)
server = VLLMServer("meta-llama/Meta-Llama-3.1-8B", gpu=0)
server.start ()
params = {"temperature": 0.0, "max_tokens" : 128, "n": 1, "seed": 0,
            "stop": ["# Query:"]}
t0 = time.time()
out = server.generate_completions ( prompts , params )
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

# score and save
n_correct, n_none = 0, 0
with open("results/mmlu_baseline.jsonl", "w") as g:
    for ex, o in zip(examples, out):
        pred = parse_mmlu_response(ex, o.text)
        n_correct += pred == ex["answer"]
        n_none += pred is None
        g.write(json.dumps(ex | {"response": o.text, "pred": pred, "correct": pred == ex["answer"]}) + "\n")
print(f"accuracy {n_correct / len(examples):.4f}  correct {n_correct}  failed parses {n_none}  total {len(examples)}")

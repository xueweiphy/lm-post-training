import json, time
from gsm8k import parse_gsm8k_response
from vllm_utils import VLLMServer

# load GSM8K test: question, gold = number after "####"
examples = []
with open("data/gsm8k/test.jsonl", "r") as f:
    for line in f :
        item = json.loads(line)
        examples.append({"question": item["question"], "answer": item["answer"].split("####")[-1].strip()})
print(len(examples), "examples")

# prompts: GSM8K task prompt inside the zero-shot system prompt
path = "prompts_safety/"
system_template = open(path + "zero_shot_system_prompt.prompt").read()
gsm8k_template = open(path + "gsm8k_zero_shot.prompt").read()
prompts = []
for ex in examples :
    task = gsm8k_template.format(question=ex["question"])
    prompts.append(system_template.format(instruction=task))

# generate (greedy); longer limit than MMLU so the model has room to reason
server = VLLMServer("meta-llama/Meta-Llama-3.1-8B", gpu=0)
server.start ()
params = {"temperature": 0.0, "max_tokens" : 512, "n": 1, "seed": 0,
            "stop": ["# Query:"]}
t0 = time.time()
out = server.generate_completions ( prompts , params )
elapsed = time.time() - t0
print(len(prompts) / elapsed, "examples/s", elapsed, "s")
server.stop()

def same(pred, gold):   # numeric compare: "18" == "18.0", "1,000" == "1000"
    return pred is not None and float(pred) == float(gold.replace(",", ""))

# score and save
n_correct, n_none = 0, 0
with open("results/gsm8k_baseline.jsonl", "w") as g:
    for ex, o in zip(examples, out):
        pred = parse_gsm8k_response(o.text)
        n_correct += same(pred, ex["answer"])
        n_none += pred is None
        g.write(json.dumps(ex | {"response": o.text, "pred": pred, "correct": same(pred, ex["answer"])}) + "\n")
print(f"accuracy {n_correct / len(examples):.4f}  correct {n_correct}  failed parses {n_none}  total {len(examples)}")

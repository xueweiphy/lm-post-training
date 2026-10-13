# Problem (prompting_baselines): generate GSM8K test answers with three prompts.
# GPU job: sbatch run_gpu.sh prompting_baselines.py  → results/<prompt>.jsonl
import json, os
from alignment.vllm_utils import VLLMServer

questions, answers = [], []
with open("data/gsm8k/test.jsonl") as f:
    for line in f:
        item = json.loads(line)
        questions.append(item["question"])
        answers.append(item["answer"].split("####")[-1].strip())

cases = ["question_only.prompt", "r1_zero.prompt", "r1_zero_three_shot_gsm8k.prompt"]

prompts = {}
for cc in cases:
    with open("alignment/prompts/" + cc) as f:
        template = f.read()
    prompts[cc] = [template.format(question=qq) for qq in questions]
    print(f"{cc}, len of prompts {len(prompts[cc])}")

server = VLLMServer("allenai/OLMo-2-0425-1B", gpu=0)
server.start()

params = {"temperature": 1.0, "max_tokens": 512, "n": 1, "seed": 0,
          "stop": ["</answer>"], "include_stop_str_in_output": True}
params_qo = {"temperature": 1.0, "max_tokens": 512, "n": 1, "seed": 0}

os.makedirs("results", exist_ok=True)
for cc in cases:
    p = params_qo if cc == "question_only.prompt" else params
    outs = server.generate_completions(prompts[cc], p)
    with open(f"results/{cc}.jsonl", "w") as g:
        for t in outs:
            g.write(json.dumps({"response": t.text}) + "\n")

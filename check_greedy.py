# Sanity check: 3-shot r1_zero, first 100 GSM8K test questions, greedy (temperature 0).
# Run on two GPU types and compare accuracy and how often "mode" appears.
import json, sys
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

N = 100
items = [json.loads(l) for l in open("data/gsm8k/test.jsonl")][:N]
tmpl = open("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt").read()
prompts = [tmpl.format(question=x["question"]) for x in items]
answers = [x["answer"].split("####")[-1].strip() for x in items]

server = VLLMServer("allenai/OLMo-2-0425-1B", gpu=0)
server.start()
params = {"temperature": 0.0, "max_tokens": 512, "n": 1, "seed": 0,
          "stop": ["</answer>"], "include_stop_str_in_output": True}
texts = [o.text for o in server.generate_completions(prompts, params)]

rews = [r1_zero_reward_fn(t, a) for t, a in zip(texts, answers)]
acc = sum(r["answer_reward"] for r in rews) / N
fmt = sum(r["format_reward"] for r in rews) / N
mode = sum("mode" in t for t in texts)
print(f"greedy 3-shot, N={N}: accuracy {acc:.2f}, format {fmt:.2f}, 'mode' in {mode} outputs")

tag = sys.argv[1] if len(sys.argv) > 1 else "gpu"
with open(f"results/greedy_3shot_{tag}.graded.jsonl", "w") as g:
    for t, a, r in zip(texts, answers, rews):
        g.write(json.dumps({"response": t, "gt": a} | r) + "\n")

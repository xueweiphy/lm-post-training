import json

# failed parses in the zero-shot GSM8K run (part c): replies with no digit in them
rows = [json.loads(l) for l in open("results/gsm8k_baseline.jsonl")]
failed = [r for r in rows if r["pred"] is None]
print(len(failed), "failed parses")
for r in failed:
    print("gold:", r["answer"], "| reply:", r["response"][:200])

# 10 random incorrect predictions (part f)
import random
random.seed(0)
wrong = [r for r in rows if not r["correct"]]
for r in random.sample(wrong, 10):
    print("Q:", r["question"])
    print("gold:", r["answer"], "| pred:", r["pred"])
    print("reply:", r["response"][:300], "\n" + "-" * 60)

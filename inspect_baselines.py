# Print 10 random category-2 and category-3 examples per prompt, for reading by hand.
# python inspect_baselines.py > inspect.txt
import json, random

files = ["question_only.prompt.graded", "r1_zero.prompt.graded",
         "r1_zero_three_shot_gsm8k.prompt.graded", "greedy_3shot_b200.graded"]
random.seed(0)

for fn in files:
    records = [json.loads(line) for line in open(f"results/{fn}.jsonl")]
    cats = {"cat2": [x for x in records if x["format_reward"] == 1 and x["answer_reward"] == 0],
            "cat3": [x for x in records if x["format_reward"] == 0]}
    for name, rs in cats.items():
        print(f"\n######## {fn}  {name}  ({len(rs)} total)")
        for ii, x in enumerate(random.sample(rs, min(10, len(rs)))):
            print(f"--- [{ii}] gt = {x['gt']}")
            print(f"question: {x.get('question', '')}")
            print(x["response"])

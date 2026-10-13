# Grade results/<prompt>.jsonl with the Dr. GRPO grader (no GPU) → results/<prompt>.graded.jsonl
import json
from alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

questions, answers = [], []
with open("data/gsm8k/test.jsonl") as f:
    for line in f:
        item = json.loads(line)
        questions.append(item["question"])
        answers.append(item["answer"].split("####")[-1].strip())

cases = ["question_only.prompt", "r1_zero.prompt", "r1_zero_three_shot_gsm8k.prompt"]
fns = {cases[0]: question_only_reward_fn, cases[1]: r1_zero_reward_fn, cases[2]: r1_zero_reward_fn}

for cc in cases:
    with open(f"results/{cc}.jsonl") as f:
        responses = [json.loads(line)["response"] for line in f]

    records = []
    for qq, rr, aa in zip(questions, responses, answers):
        records.append({"question": qq, "response": rr, "gt": aa} | fns[cc](rr, aa))

    c1 = sum(1 for x in records if x["answer_reward"] == 1)
    c2 = sum(1 for x in records if x["answer_reward"] == 0 and x["format_reward"] == 1)
    c3 = sum(1 for x in records if x["format_reward"] == 0)
    print(f"{cc}: correct {c1}, format only {c2}, format wrong {c3}, "
          f"total {c1 + c2 + c3}, accuracy {c1 / len(records):.3f}")

    with open(f"results/{cc}.graded.jsonl", "w") as f:
        for x in records:
            f.write(json.dumps(x) + "\n")

import glob, json
import matplotlib.pyplot as plt

# final val accuracy per prompt, one dot per seed + mean (grpo_prompt_ablation)
groups = {"r1_zero": glob.glob("results/grpo_seed[0-9].jsonl"),
          "question_only": glob.glob("results/grpo_question_only_lr1e-05_seed[0-9].jsonl"),
          "r1_zero 3-shot": glob.glob("results/grpo_r1_zero_three_shot_gsm8k_lr1e-05_seed[0-9].jsonl")}
for i, (name, files) in enumerate(groups.items()):
    finals = [[json.loads(l)["val_reward"] for l in open(f) if "val_reward" in l][-1] for f in files]
    plt.scatter([i] * len(finals), finals, color="#2a78d6", alpha=0.7)
    plt.scatter(i, sum(finals) / len(finals), color="#eb6834", marker="_", s=600)
    print(name, [round(x, 3) for x in finals], "mean", round(sum(finals) / len(finals), 3))
plt.xticks(range(len(groups)), list(groups)); plt.xlim(-0.5, len(groups) - 0.5)
plt.ylabel("final val accuracy (step 200)"); plt.title("GRPO prompt ablation (dots: seeds, bar: mean)"); plt.grid(alpha=0.3)
plt.savefig("results/grpo_prompt_ablation.png", dpi=150); print("saved results/grpo_prompt_ablation.png")

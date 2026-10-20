import json, sys
import matplotlib.pyplot as plt

file = sys.argv[1] if len(sys.argv) > 1 else "results/grpo_seed0.jsonl"
recs = [json.loads(l) for l in open(file)]
train = [r for r in recs if "reward_mean" in r]
val = [r for r in recs if "val_reward" in r]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, k, title in zip(axes, ["reward", "format_reward"], ["answer reward (accuracy)", "format reward"]):
    ax.plot([r["step"] for r in train], [r[k + "_mean"] for r in train], color="#2a78d6", lw=1, alpha=0.6, label="train (256 rollouts)")
    ax.plot([r["step"] for r in val], [r["val_" + k] for r in val], color="#eb6834", lw=2, marker="o", ms=4, label="val (1024 questions)")
    ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
axes[0].set_ylabel("mean reward"); axes[0].legend()
plt.tight_layout()
plt.savefig(file.replace(".jsonl", ".png"), dpi=150)
print("saved", file.replace(".jsonl", ".png"))

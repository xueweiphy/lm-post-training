import json, sys
import matplotlib.pyplot as plt

# usage: python plot_seeds.py results/grpo_seed0.jsonl results/grpo_seed1.jsonl ...
files = sys.argv[1:] or [f"results/grpo_seed{s}.jsonl" for s in range(4)]
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
panels = [("val_reward", "val accuracy"), ("val_format_reward", "val format reward"), ("val_length", "val response length (tokens)"),
          ("reward_mean", "train reward"), ("format_reward_mean", "train format reward"), ("token_entropy", "token entropy (nats)"),
          ("loss", "loss"), ("grad_norm", "grad norm (before clip)")]

def smooth(y, k=10):                       # moving average for the noisy per-step train metrics
    return [sum(y[max(0, i - k + 1):i + 1]) / len(y[max(0, i - k + 1):i + 1]) for i in range(len(y))]

fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
for ax, (key, title) in zip(axes.flat, panels):
    for file, c in zip(files, colors):
        recs = [json.loads(l) for l in open(file) if key in l]
        x = [r["step"] for r in recs if key in r]
        y = [r[key] for r in recs if key in r]
        if not y: continue
        label = file.split("/")[-1].replace("grpo_", "").replace(".jsonl", "")
        if key.startswith("val_"):
            ax.plot(x, y, color=c, lw=2, marker="o", ms=3, label=label)
        else:
            ax.plot(x, y, color=c, lw=0.6, alpha=0.25)
            ax.plot(x, smooth(y), color=c, lw=2, label=label)
    ax.set_title(title); ax.set_xlabel("step"); ax.grid(alpha=0.3)
axes[0, 0].legend()
plt.tight_layout()
out = "results/grpo_seeds.png"
plt.savefig(out, dpi=130)
print("saved", out)

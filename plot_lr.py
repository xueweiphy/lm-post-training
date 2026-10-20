import glob, json
import matplotlib.pyplot as plt

# final val accuracy vs learning rate, one dot per seed + mean (grpo_learning_rate deliverable)
runs = {1e-5: glob.glob("results/grpo_seed[0-9].jsonl")}
for file in glob.glob("results/grpo_r1_zero_lr*_seed*.jsonl"):
    if file.endswith("_rollouts.jsonl"): continue
    lr = float(file.split("_lr")[1].split("_seed")[0])
    runs.setdefault(lr, []).append(file)

lrs = sorted(runs)
for lr in lrs:
    vals = [[json.loads(l)["val_reward"] for l in open(f) if "val_reward" in l] for f in runs[lr]]
    finals, peaks = [v[-1] for v in vals], [max(v) for v in vals]
    plt.scatter([lr] * len(finals), finals, color="#2a78d6", alpha=0.7, label="final (step 200)" if lr == lrs[0] else None)
    plt.scatter([lr] * len(peaks), peaks, facecolors="none", edgecolors="#2a78d6", label="best during run" if lr == lrs[0] else None)
    plt.scatter(lr, sum(finals) / len(finals), color="#eb6834", marker="_", s=400, label="mean final" if lr == lrs[0] else None)
    print(lr, "final", [round(x, 3) for x in finals], "best", [round(x, 3) for x in peaks])
plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("final val accuracy (step 200)")
plt.title("GRPO learning-rate sweep"); plt.grid(alpha=0.3); plt.legend()
plt.savefig("results/grpo_lr_sweep.png", dpi=150); print("saved results/grpo_lr_sweep.png")

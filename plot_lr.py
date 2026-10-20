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
    finals = [json.loads(open(f).readlines()[-1]).get("val_reward") for f in runs[lr]]
    finals = [x for x in finals if x is not None]
    plt.scatter([lr] * len(finals), finals, color="#2a78d6", alpha=0.6)
    if finals: plt.scatter(lr, sum(finals) / len(finals), color="#eb6834", marker="_", s=400)
    print(lr, [round(x, 3) for x in finals])
plt.xscale("log"); plt.xlabel("learning rate"); plt.ylabel("final val accuracy (step 200)")
plt.title("GRPO learning-rate sweep (dots: seeds, bar: mean)"); plt.grid(alpha=0.3)
plt.savefig("results/grpo_lr_sweep.png", dpi=150); print("saved results/grpo_lr_sweep.png")

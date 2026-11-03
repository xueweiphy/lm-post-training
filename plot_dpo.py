import json, sys
import matplotlib.pyplot as plt

# DPO curves: train loss (mean over 64 pairs) and val loss / accuracy (200 HH pairs).   usage: python plot_dpo.py [results/dpo_lr1e-06.jsonl]
f = sys.argv[1] if len(sys.argv) > 1 else "results/dpo_lr1e-06.jsonl"
R = [json.loads(l) for l in open(f)]
T = [r for r in R if "loss" in r]; V = [r for r in R if "val_loss" in r]
s, loss = [r["step"] for r in T], [r["loss"] for r in T]
smooth = [sum(loss[max(0, i - 19):i + 1]) / len(loss[max(0, i - 19):i + 1]) for i in range(len(loss))]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.plot(s, loss, color="#9db8d9", lw=0.6, label="train (mean of 64 pairs)")
a.plot(s, smooth, color="#2a78d6", label="train, 20-step mean")
a.plot([r["step"] for r in V], [r["val_loss"] for r in V], "o-", color="#eb6834", ms=3, label="val (200 pairs)")
a.axhline(0.6931, color="gray", ls="--", lw=0.8); a.text(5, 0.698, "log 2 (no preference)", color="gray", fontsize=8)
a.set_xlabel("optimizer step (64 pairs)"); a.set_ylabel("DPO loss"); a.grid(alpha=0.3); a.legend(); a.set_title("DPO on HH from the SFT model")
b.plot([r["step"] for r in V], [r["val_acc"] for r in V], "o-", color="#eb6834", ms=3)
b.axhline(0.5, color="gray", ls="--", lw=0.8); b.set_ylim(0.45, 0.8)
b.set_xlabel("optimizer step"); b.set_ylabel("val accuracy (chosen ranked first)"); b.grid(alpha=0.3)
plt.tight_layout(); out = f.replace(".jsonl", ".png"); plt.savefig(out, dpi=150); print("saved", out)

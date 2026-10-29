import json, sys
import matplotlib.pyplot as plt

# SFT learning curve: train loss (raw + 200-step mean), val loss, lr.   usage: python plot_sft.py [results/sft_lr2e-05.jsonl]
f = sys.argv[1] if len(sys.argv) > 1 else "results/sft_lr2e-05.jsonl"
R = [json.loads(l) for l in open(f)]
T = [r for r in R if "loss" in r]; V = [r for r in R if "val_loss" in r]
s, loss = [r["step"] for r in T], [r["loss"] for r in T]
smooth = [sum(loss[max(0, i - 19):i + 1]) / len(loss[max(0, i - 19):i + 1]) for i in range(len(loss))]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.plot(s, loss, color="#9db8d9", lw=0.6, label="train (one micro-batch, every 10 steps)")
a.plot(s, smooth, color="#2a78d6", label="train, 200-step mean")
a.plot([r["step"] for r in V], [r["val_loss"] for r in V], "o-", color="#eb6834", ms=3, label="val (1,000 test examples)")
a.set_xlabel("optimizer step (32 × 512 tokens)"); a.set_ylabel("loss (nats / token)"); a.set_ylim(1.2, 2.0)
a.grid(alpha=0.3); a.legend(); a.set_title("SFT of Llama 3.1 8B, 1 epoch")
b.plot(s, [r["lr"] for r in T], color="#2a78d6"); b.set_ylabel("learning rate", color="#2a78d6")
b2 = b.twinx(); b2.plot(s, [r["grad_norm"] for r in T], color="#eb6834", lw=0.6); b2.set_ylabel("grad norm (before clip)", color="#eb6834")
b2.axhline(1.0, color="#eb6834", ls="--", lw=0.8); b2.set_ylim(0, 8); b.set_xlabel("optimizer step"); b.grid(alpha=0.3)
plt.tight_layout(); out = f.replace(".jsonl", ".png"); plt.savefig(out, dpi=150); print("saved", out)

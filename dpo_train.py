# DPO on Anthropic HH, starting from the SFT model (supplement 6.4). Policy on cuda:0, frozen reference on cuda:1.
# usage: python dpo_train.py [lr] [n_train]      e.g. smoke test: python dpo_train.py 1e-6 640
import json, math, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hh import load_hh
from dpo import dpo_loss

model_path = "checkpoints/sft_lr2e-05"
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
n_train = int(sys.argv[2]) if len(sys.argv) > 2 else None      # None = all 49,236 training pairs
beta = 0.1
grad_accum = 64                                                  # one pair per dpo_loss call, 64 pairs per step
output_dir = f"checkpoints/dpo_lr{lr:g}"
log_file = f"results/dpo_lr{lr:g}.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_path)
lm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2").to("cuda:0")
lm_ref = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2").to("cuda:1")
lm_ref.eval()
lm_ref.requires_grad_(False)

data = [ex for ex in load_hh() if ex["chosen"] and ex["rejected"] and ex["chosen"] != ex["rejected"]]   # drop 85 empty / identical pairs
random.seed(0)
random.shuffle(data)
val, train = data[:200], data[200:][:n_train]
total_steps = len(train) // grad_accum
print(len(train), "train pairs", len(val), "val pairs", total_steps, "steps")


def validate(step):
    with torch.no_grad():
        vl = [dpo_loss(lm, lm_ref, tokenizer, beta, ex).item() for ex in val]
    acc = [0.5 if abs(l - math.log(2)) < 1e-6 else float(l < math.log(2)) for l in vl]   # tie (margin 0) counts half
    rec = {"step": step, "val_loss": sum(vl) / len(vl), "val_acc": sum(acc) / len(acc)}
    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)


# fp32 master weights: an update of lr 1e-6 is below bf16 resolution for ~97% of the weights, so RMSprop
# steps an fp32 copy and the bf16 model is refreshed from it (~130 GB on cuda:0, needs a B200)
params = list(lm.parameters())
master = [p.detach().float() for p in params]
opt = torch.optim.RMSprop(master, lr=lr)
log = open(log_file, "w")
step, loss_sum = 0, 0.0
validate(0)                                                      # sanity check: val_loss = log 2 = 0.693
for ii, dd in enumerate(train):
    loss = dpo_loss(lm, lm_ref, tokenizer, beta, dd) / grad_accum
    loss.backward()
    loss_sum += loss.item()
    if (ii + 1) % grad_accum == 0:
        for p, m in zip(params, master):
            m.grad = p.grad.float()
            p.grad = None
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            for p, m in zip(params, master):
                p.copy_(m)
        step += 1
        rec = {"step": step, "loss": loss_sum, "grad_norm": gn.item()}   # mean loss over the 64 pairs
        loss_sum = 0.0
        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)
        if step % 20 == 0 or step == total_steps:
            validate(step)

log.close()
lm.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

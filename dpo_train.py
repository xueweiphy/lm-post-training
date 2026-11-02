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

data = load_hh()
random.seed(0)
random.shuffle(data)
val, train = data[:200], data[200:][:n_train]
total_steps = len(train) // grad_accum
print(len(train), "train pairs", len(val), "val pairs", total_steps, "steps")


def validate(step):
    with torch.no_grad():
        vl = [dpo_loss(lm, lm_ref, tokenizer, beta, ex).item() for ex in val]
    rec = {"step": step, "val_loss": sum(vl) / len(vl), "val_acc": sum(l < math.log(2) for l in vl) / len(vl)}
    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)


opt = torch.optim.RMSprop(lm.parameters(), lr=lr)
log = open(log_file, "w")
step = 0
validate(0)                                                      # sanity check: val_loss = log 2 = 0.693
for ii, dd in enumerate(train):
    loss = dpo_loss(lm, lm_ref, tokenizer, beta, dd) / grad_accum
    loss.backward()
    if (ii + 1) % grad_accum == 0:
        gn = torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        rec = {"step": step, "loss": loss.item() * grad_accum, "grad_norm": gn.item()}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)
        if step % 20 == 0 or step == total_steps:
            validate(step)

log.close()
lm.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

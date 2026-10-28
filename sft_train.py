# SFT of Llama 3.1 8B on the instruction-tuning data (supplement 4.2.2).
# usage: python sft_train.py [lr] [train_file]      e.g. smoke test: python sft_train.py 2e-5 data/sft/sft-instruct/sample_train.jsonl
import json, math, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sft_data import DataLoading, iterate_batches

model_path = "meta-llama/Meta-Llama-3.1-8B"
device = "cuda:0"
seq_length = 512
micro_batch = 4
grad_accum = 32 // micro_batch          # 32 sequences per optimizer step
lr = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5
train_file = sys.argv[2] if len(sys.argv) > 2 else "data/sft/sft-instruct/train.jsonl"
weight_decay = 0.1
max_grad_norm = 1.0
output_dir = f"checkpoints/sft_lr{lr:g}"
log_file = f"results/sft_lr{lr:g}.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2").to(device)

train = DataLoading(tokenizer, train_file, seq_length, shuffle=True)
val = DataLoading(tokenizer, "data/sft/sft-instruct/test.jsonl", seq_length, shuffle=False, n_max=1000)
data_loader = iterate_batches(train, micro_batch, True)
val_loader = iterate_batches(val, 8, False)


def evaluate(model, loader):
    model.eval(); total = 0.0
    with torch.no_grad():
        for b in loader:
            logits = model(b["input_ids"].to(device)).logits.float()
            total += F.cross_entropy(logits.transpose(1, 2), b["labels"].to(device)).item()
    model.train()
    return total / len(loader)


def learning_rate_schedule(t, alpha_max, alpha_min, Tw, Tc):
    if t < Tw:
        out = t / Tw * alpha_max
    elif t <= Tc:
        out = alpha_min + 0.5 * (1 + math.cos((t - Tw) * math.pi / (Tc - Tw))) * (alpha_max - alpha_min)
    else:
        out = alpha_min
    return out


total_steps = len(data_loader) // grad_accum     # optimizer steps in 1 epoch
Tw = max(1, int(0.03 * total_steps))             # 3% warmup
Tc = total_steps
print(f"{len(train)} train chunks, {len(val)} val chunks, {total_steps} steps")

log = open(log_file, "w")
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
step = 0
val_loss = evaluate(model, val_loader)
log.write(json.dumps({"step": 0, "val_loss": val_loss}) + "\n"); log.flush(); print("val_loss", val_loss)

for idx, batch in enumerate(data_loader):
    inputs = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(inputs).logits.float()
    loss = F.cross_entropy(logits.transpose(1, 2), labels) / grad_accum
    loss.backward()
    if (idx + 1) % grad_accum == 0:
        cur_lr = learning_rate_schedule(step, lr, 0.0, Tw, Tc)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()
        step += 1
        if step % 10 == 0:
            rec = {"step": step, "lr": cur_lr, "loss": loss.item() * grad_accum, "grad_norm": gn.item(),
                   "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec)
        if step % 200 == 0 or step == total_steps:
            val_loss = evaluate(model, val_loader)
            log.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n"); log.flush(); print("val_loss", val_loss)

log.close()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

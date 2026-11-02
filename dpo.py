import os
import torch
import torch.nn.functional as F

# DPO loss (supplement 6.3): Alpaca template (as in SFT) + EOS, log-probs of the whole sequence under the
# trained and the reference model; the prompt part cancels in the chosen - rejected difference.
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts_safety")
sft_template = open(os.path.join(PROMPT_DIR, "alpaca_sft.prompt")).read().rstrip("\n")


def log_prob(model, ids):
    logits = model(ids).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[0, 1:]
    return logp[0, torch.arange(len(tgt)), tgt].sum()      # log pi(whole sequence)


def dpo_loss(lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected):
    seqw = sft_template.format(instruction=prompt, response=response_chosen)
    seql = sft_template.format(instruction=prompt, response=response_rejected)
    ids_w = torch.tensor([tokenizer.encode(seqw) + [tokenizer.eos_token_id]], device=lm.device)
    ids_l = torch.tensor([tokenizer.encode(seql) + [tokenizer.eos_token_id]], device=lm.device)
    logpw = log_prob(lm, ids_w)
    logpl = log_prob(lm, ids_l)
    with torch.no_grad():                                   # reference is frozen, maybe on another GPU
        logpw_ref = log_prob(lm_ref, ids_w.to(lm_ref.device)).to(lm.device)
        logpl_ref = log_prob(lm_ref, ids_l.to(lm_ref.device)).to(lm.device)
    return -F.logsigmoid(beta * ((logpw - logpw_ref) - (logpl - logpl_ref)))

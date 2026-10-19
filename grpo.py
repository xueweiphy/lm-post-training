from typing import Callable, Literal
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str],
                               tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    dic = {}
    tok_prompt = [tokenizer.encode(s, add_special_tokens=False) for s in prompt_strs]
    tok_output = [tokenizer.encode(s, add_special_tokens=False) for s in output_strs]
    seqs = [p + o for p, o in zip(tok_prompt, tok_output)]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)

    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, len(tok_prompt[i]):len(s)] = True

    dic["input_ids"] = ids[:, :-1]
    dic["labels"] = ids[:, 1:]
    dic["response_mask"] = mask[:, 1:]
    return dic


def get_response_log_probs(model: PreTrainedModel, input_ids: torch.Tensor, labels: torch.Tensor,
                           return_token_entropy: bool = False) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    B, T, V = logits.shape
    dic = {}
    logp = torch.log_softmax(logits, dim=-1)
    b = torch.arange(B, device=logp.device)[:, None]
    t = torch.arange(T, device=logp.device)[None, :]
    dic["log_probs"] = logp[b, t, labels]

    if return_token_entropy:
        dic["token_entropy"] = -(logp.exp() * logp).sum(dim=-1)
    return dic


def compute_rollout_rewards(reward_fn: Callable[[str, str], dict[str, float]], rollout_responses: list[str],
                            repeated_ground_truths: list[str]) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    format_rewards = []
    for rr, tt in zip(rollout_responses, repeated_ground_truths):
        dic = reward_fn(rr, tt)
        rewards.append(dic["reward"])
        format_rewards.append(dic["format_reward"])

    rewards = torch.tensor(rewards)
    format_rewards = torch.tensor(format_rewards)
    metadata = {"reward_mean": rewards.mean().item(),
                "format_reward_mean": format_rewards.mean().item()}
    return (rewards, metadata)


def compute_group_normalized_rewards(raw_rewards: torch.Tensor, group_size: int,
                                     baseline: Literal["mean", "none"] = "mean",
                                     advantage_eps: float = 1e-6,
                                     advantage_normalizer: Literal["std", "none", "mean"] = "std"):
    rewards = raw_rewards.view(-1, group_size)
    rewards_mean = rewards.mean(dim=1, keepdim=True)

    Adv = rewards
    if baseline == "mean":
        Adv = rewards - rewards_mean

    if advantage_normalizer == "std":
        rewards_std = rewards.std(dim=1, keepdim=True)
        Adv = Adv / (rewards_std + advantage_eps)
    elif advantage_normalizer == "mean":
        Adv = Adv / (rewards_mean + advantage_eps)

    metadata = {"reward_mean": raw_rewards.mean().item(), "reward_std": raw_rewards.std().item()}
    return (Adv.view(-1), metadata)


def compute_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor,
                                 importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
                                 old_log_probs: torch.Tensor | None = None, cliprange: float | None = None,
                                 response_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if importance_reweighting_method != "none":
        raise NotImplementedError

    loss = -raw_rewards_or_advantages.view(-1, 1) * policy_log_probs
    metadata = {}
    return (loss, metadata)


def aggregate_loss_across_microbatch(per_token_policy_gradient_loss: torch.Tensor, mask: torch.Tensor,
                                     loss_normalization: Literal["sequence", "constant"] = "sequence",
                                     normalization_constant: int | None = None) -> torch.Tensor:
    loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        loss = loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        loss = loss.mean()
    elif loss_normalization == "constant":
        loss = loss.sum() / normalization_constant
    else:
        raise NotImplementedError
    return loss


def grpo_train_step(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                    optimizer: torch.optim.Optimizer,
                    gradient_accumulation_steps: int, 
                    max_grad_norm: float | None,
                    reward_fn: Callable[[str, str], dict[str, float]],
                    repeated_prompts: list[str], rollout_responses: list[str],
                    repeated_ground_truths: list[str], group_size: int,
                    baseline: Literal["mean", "none"] = "mean",
                    advantage_eps: float = 1e-6,
                    advantage_normalizer: Literal["std", "none", "mean"] = "std",
                    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
                    old_log_probs: torch.Tensor | None = None,
                    cliprange: float | None = None,
                    loss_normalization: Literal["sequence", "constant"] = "sequence",
                    normalization_constant: int | None = None,
                    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    # get rewards and adv
    raw_rewards, meta = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    adv, _ = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    # tokenize
    toks = tokenize_prompt_and_output( repeated_prompts, rollout_responses, tokenizer ) 

    mask = toks["response_mask"].to(model.device)
    inputs = toks["input_ids"].to(model.device)
    labels = toks["labels"].to(model.device)
    adv = adv.to(model.device)
    
    microbatch_size = len(inputs) // gradient_accumulation_steps
    total = 0.0
    for i in range(0, len(inputs), microbatch_size):
        inputs_microbatch = inputs[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]
        mask_microbatch = mask[i:i+microbatch_size]
        adv_microbatch = adv[i:i+microbatch_size]

        dic0  = get_response_log_probs ( model, inputs_microbatch, labels_microbatch )
        logprobs = dic0["log_probs"]

        grad_loss, _ = compute_policy_gradient_loss(adv_microbatch, logprobs, importance_reweighting_method, 
                                                    old_log_probs, cliprange, mask_microbatch)
        loss = aggregate_loss_across_microbatch(grad_loss, mask_microbatch, loss_normalization, 
                                                normalization_constant) / gradient_accumulation_steps

        total += loss.detach()

        loss.backward()

    if max_grad_norm is not None:
        meta["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    if torch.isfinite(torch.tensor(meta.get("grad_norm", 0.0))):
        optimizer.step()

    optimizer.zero_grad()
    return ( total, meta)

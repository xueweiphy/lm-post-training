import torch
from transformers import PreTrainedTokenizer


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

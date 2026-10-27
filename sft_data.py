import json, os, random
import torch

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts_safety")


class DataLoading(torch.utils.data.Dataset):
    """Packed SFT data: Alpaca-format every example, join with EOS, cut into seq_length chunks."""

    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        examples = [json.loads(l) for l in open(dataset_path)]
        sft_template = open(os.path.join(PROMPT_DIR, "alpaca_sft.prompt")).read().rstrip("\n")   # fixture has no newline before EOS
        self.seq_length = seq_length
        if shuffle:
            random.shuffle(examples)
        prompts = [sft_template.format(instruction=ex["prompt"], response=ex["response"]) for ex in examples]
        tok_prompt = [tokenizer.encode(s) for s in prompts]
        self.ids = []
        for t in tok_prompt:
            self.ids += t + [tokenizer.eos_token_id]

    def __len__(self):
        return (len(self.ids) - 1) // self.seq_length

    def __getitem__(self, i):
        m = self.seq_length
        return {"input_ids": torch.tensor(self.ids[i*m : (i+1)*m]),
                "labels":    torch.tensor(self.ids[i*m+1 : (i+1)*m+1])}


def iterate_batches(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

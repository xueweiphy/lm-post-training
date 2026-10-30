import gzip, json, random

# Anthropic HH preference data (supplement 6.2, look_at_hh): single-turn examples only,
# split into instruction / chosen / rejected, remembering the source file.


def load_hh(file_path="data/hh/"):
    files = ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]
    data = []
    for ff in files:
        for ll in gzip.open(file_path + ff + ".jsonl.gz", "rt"):
            rr = json.loads(ll)
            if rr["chosen"].count("\n\nHuman:") != 1 or rr["rejected"].count("\n\nHuman:") != 1:   # skip multi-turn
                continue
            prompt, chosen = rr["chosen"].split("\n\nAssistant:", 1)
            rejected = rr["rejected"].split("\n\nAssistant:", 1)[1]
            data.append({"file": ff, "instruction": prompt.replace("\n\nHuman:", "").strip(),
                         "chosen": chosen.strip(), "rejected": rejected.strip()})
    return data


if __name__ == "__main__":
    data = load_hh()
    print(len(data), "single-turn examples")
    random.seed(0)
    helpful = [ex for ex in data if ex["file"].startswith("helpful")]
    harmless = [ex for ex in data if ex["file"].startswith("harmless")]
    for ex in random.sample(helpful, 3) + random.sample(harmless, 3):
        print(ex["file"], "\nQ:", ex["instruction"], "\n--- CHOSEN:\n", ex["chosen"], "\n--- REJECTED:\n", ex["rejected"], "\n" + "=" * 80)

import json
import random
import torch
from vllm_utils import VLLMServer
from checkpoint import get_model_and_tokenizer
from grpo import grpo_train_step
from drgrpo_grader import r1_zero_reward_fn

# hyperparameters (handout §4.3)
seed = 0
n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0
model_id = "allenai/OLMo-2-0425-1B"
device = "cuda:0"            # policy on GPU 0, vLLM on GPU 1

random.seed(seed)
torch.manual_seed(seed)

# data load and prompt generated
file ="data/gsm8k/train.jsonl"
prompt = "r1_zero.prompt"
filepath  = "prompts/"
file_val = "data/gsm8k/test.jsonl"

def data_prompt ( filedata, fileprompt = "prompts/" , prompt = "r1_zero.prompt" , Num = -1):

    questions, answers = [], []
    with open(filedata, "r") as f:
        for  ii, lines in enumerate(f) :
            if ii == Num :
                break
            item = json.loads(lines)
            questions.append ( item ["question"] )
            answers.append ( item ["answer"].split("####")[-1].strip())

    with open ( fileprompt +prompt, "r") as f :
        template= f.read()
    ptlist =[]
    for qq in questions :
        ptlist.append ( template.format ( question = qq )  )

    return questions, answers, ptlist

def evaluate(server, prompts, answers, params) :
    outs  =  server.generate_completions(prompts, params)
    rollout_responses = [o.text for o in outs]
    lengths = [len(o.token_ids) for o in outs]
    rewards = []
    format_rewards= []
    for rr, tt in zip(rollout_responses, answers ):
        dic = r1_zero_reward_fn (rr, tt)
        rewards.append(dic["reward"])
        format_rewards.append(dic["format_reward"])
    rewards = torch.tensor(rewards)
    format_rewards = torch.tensor(format_rewards)
    return rewards.mean().item(), format_rewards.mean().item(), sum(lengths) / len(lengths)

questions, answers, promptlist  = data_prompt ( file, filepath, prompt, Num = n_train_examples)
questions_val, answers_val, promptlist_val  = data_prompt ( file_val, filepath, prompt, Num = n_val_examples)
print ( len ( promptlist ), len (  promptlist_val ))

# model, optimizer, vLLM
model, tokenizer = get_model_and_tokenizer(model_id, device)
optimizer = torch.optim.AdamW ( model.parameters(), lr = learning_rate, betas = ( 0.9, 0.95 ) , weight_decay = 0.0 )
server = VLLMServer ( model_id, gpu=1)
server.start ()
server.init_weight_sync(device)
params = {"temperature": sampling_temperature, "max_tokens" : sampling_max_tokens, "n": 1,
            "stop": ["</answer>"], "include_stop_str_in_output": True}

n_prompts = rollout_batch_size // group_size

# training loop
f = open(f"results/grpo_seed{seed}.jsonl", "w")

for step in range(num_rollout_steps):

    rec = {"step": step}
    server.sync_policy_weights(model)
    if step % 10 == 0:
        rw, rw_format, val_len = evaluate(server, promptlist_val, answers_val, params)
        rec["val_reward"] = rw
        rec["val_format_reward"] = rw_format
        rec["val_length"] = val_len
        print(step, "val", rw, rw_format, val_len)
    idx = random.sample ( range ( len ( promptlist ) ) , n_prompts)

    repeated_prompts = [promptlist[ii] for ii in idx for _ in range(group_size)]
    repeated_ground_truths = [answers[ii] for ii in idx for _ in range(group_size)]

    outs = server.generate_completions ( repeated_prompts, params )
    rollout_responses = [o.text for o in outs]

    loss, meta = grpo_train_step(model, tokenizer, optimizer,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        max_grad_norm=max_grad_norm,
                        reward_fn=r1_zero_reward_fn,
                        repeated_prompts=repeated_prompts,
                        rollout_responses=rollout_responses,
                        repeated_ground_truths=repeated_ground_truths,
                        group_size=group_size)
    rec["loss"] = loss.item()
    rec = rec | meta
    f.write ( json.dumps ( rec ) +"\n"  )
    f.flush()
    print(step, loss.item(), meta)

server.sync_policy_weights(model)
rw, rw_format, val_len = evaluate(server, promptlist_val, answers_val, params)
f.write(json.dumps({"step": num_rollout_steps, "val_reward": rw,
                    "val_format_reward": rw_format, "val_length": val_len}) + "\n")
print("final",rw, rw_format, val_len)
f.close()
server.stop()

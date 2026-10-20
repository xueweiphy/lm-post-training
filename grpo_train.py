import json
import os
import random
import sys
import torch
from vllm_utils import VLLMServer
from checkpoint import get_model_and_tokenizer
from grpo import grpo_train_step
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# usage: python grpo_train.py <seed> [learning_rate] [prompt]
#   python grpo_train.py 2                       -> seed 2, lr 1e-5, r1_zero
#   python grpo_train.py 0 3e-5                  -> learning-rate sweep
#   python grpo_train.py 0 1e-5 question_only    -> prompt ablation
# hyperparameters (handout §4.3)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
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
prompt = (sys.argv[3] if len(sys.argv) > 3 else "r1_zero") + ".prompt"   # r1_zero, question_only, r1_zero_three_shot_gsm8k
reward_fn = question_only_reward_fn if prompt == "question_only.prompt" else r1_zero_reward_fn
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
        dic = reward_fn (rr, tt)
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
port = 8000 + int(os.environ.get("SLURM_JOB_ID", seed)) % 1000      # own port: jobs may share a node
server = VLLMServer ( model_id, gpu=1, port=port)
server.start ()
server.init_weight_sync(device)
params = {"temperature": sampling_temperature, "max_tokens" : sampling_max_tokens, "n": 1, "seed": None}
if prompt != "question_only.prompt":                   # r1_zero prompts end at </answer>
    params |= {"stop": ["</answer>"], "include_stop_str_in_output": True}

n_prompts = rollout_batch_size // group_size

# training loop
# default run keeps the old name; sweeps get the lr and prompt in the name
if learning_rate == 1e-5 and prompt == "r1_zero.prompt":
    name = f"grpo_seed{seed}"
else:
    name = f"grpo_{prompt[:-7]}_lr{learning_rate:g}_seed{seed}"
f = open(f"results/{name}.jsonl", "w")
g = open(f"results/{name}_rollouts.jsonl", "w")

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
                        reward_fn=reward_fn,
                        repeated_prompts=repeated_prompts,
                        rollout_responses=rollout_responses,
                        repeated_ground_truths=repeated_ground_truths,
                        group_size=group_size)
    rec["n_empty"] = sum(len(o.token_ids) == 0 for o in outs)
    if loss.isnan():
        json.dump({"responses": rollout_responses, "gt": repeated_ground_truths},
                  open(f"results/nan_step{step}.json", "w"))
    rec["loss"] = loss.item()
    if step % 40 == 0 or step == num_rollout_steps - 1:  # first group of 8: one question
        for rr, tt in zip(rollout_responses[:group_size], repeated_ground_truths[:group_size]):
            g.write(json.dumps({"step": step, "prompt": repeated_prompts[0], "response": rr,
                                "gt": tt} | reward_fn(rr, tt)) + "\n")
        g.flush()
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
g.close()
server.stop()

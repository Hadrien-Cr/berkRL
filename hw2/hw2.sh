#!/bin/bash
if [ -d "data/q2_pg_cartpole" ]; then
    echo "The logs for runs 'CartPole' already exist."
else
    python3 -m cs285.scripts.run_hw2 --env_name CartPole-v1 -n 25 -b 1000 --exp_name cartpole/cartpol_no_baseline_no_causal 
    python3 -m cs285.scripts.run_hw2 --env_name CartPole-v1 -n 25 -b 1000 --exp_name cartpole/cartpol_no_baseline_causal --use_reward_to_go --normalize_advantages
    python3 -m cs285.scripts.run_hw2 --env_name CartPole-v1 -n 25 -b 1000 --exp_name cartpole/cartpol_baseline_no_causal --use_baseline --normalize_advantages
    python3 -m cs285.scripts.run_hw2 --env_name CartPole-v1 -n 25 -b 1000 --exp_name cartpole/cartpol_baseline_causal --use_baseline --normalize_advantages --use_reward_to_go
fi

if [ -d "data/q2_pg_halfcheetah" ]; then
    echo "The logs for runs 'HalfCheetah' already exist."
else
    python3 -m cs285.scripts.run_hw2 --env_name HalfCheetah-v5 -n 100 -b 5000 --discount 0.95 -lr 0.01 --exp_name halfcheetah/cheetah_no_baseline_no_causal -na
    python3 -m cs285.scripts.run_hw2 --env_name HalfCheetah-v5 -n 100 -b 5000 --discount 0.95 -lr 0.01 --exp_name halfcheetah/cheetah_no_baseline_causal --use_reward_to_go --normalize_advantages 
    python3 -m cs285.scripts.run_hw2 --env_name HalfCheetah-v5 -n 100 -b 5000 --discount 0.95 -lr 0.01 --exp_name halfcheetah/cheetah_baseline_no_causal --use_baseline --normalize_advantages
    python3 -m cs285.scripts.run_hw2 --env_name HalfCheetah-v5 -n 100 -b 5000  --discount 0.95 -lr 0.01 --exp_name halfcheetah/cheetah_baseline_causal --use_baseline --normalize_advantages --use_reward_to_go
fi

if [ -d "data/q2_pg_lunarlander" ]; then
    echo "The logs for runs LunarLander already exist."
else
    python3 -m cs285.scripts.run_hw2 --env_name LunarLander-v3 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --exp_name lunarlander/lunarlander_baseline_causal --use_baseline --normalize_advantages  --use_reward_to_go 
    python3 -m cs285.scripts.run_hw2 --env_name LunarLander-v3 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --exp_name lunarlander/lunarlander_gae=0._causal --use_baseline --normalize_advantages --gae_lambda 0.00 --use_reward_to_go 
    python3 -m cs285.scripts.run_hw2 --env_name LunarLander-v3 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --exp_name lunarlander/lunarlander_gae=0.9_causal --use_baseline --normalize_advantages --gae_lambda 0.9 --use_reward_to_go
    python3 -m cs285.scripts.run_hw2 --env_name LunarLander-v3 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --exp_name lunarlander/lunarlander_gae=0.95_causal --use_baseline --normalize_advantages --gae_lambda 0.95 --use_reward_to_go
    python3 -m cs285.scripts.run_hw2 --env_name LunarLander-v3 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --exp_name lunarlander/lunarlander_gae=0.99_causal --use_baseline --normalize_advantages --gae_lambda 0.99 --use_reward_to_go
    python3 -m cs285.scripts.run_hw2 --env_name LunarLander-v3 --ep_len 1000 --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 --exp_name lunarlander/lunarlander_gae=1_causal --use_baseline --normalize_advantages --gae_lambda 1 --use_reward_to_go
fi

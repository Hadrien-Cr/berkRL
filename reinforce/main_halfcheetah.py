
import gymnasium as gym
from policy import Policy
from reinforce import reinforce
from torch import optim
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
env_id = "HalfCheetah-v5"
# Create the env
env = gym.make(env_id)

# Create the evaluation env
eval_env = gym.make(env_id)

cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 0.95,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": env.observation_space.shape[0],
    "action_space":  env.action_space.shape[0],
}
# Create policy and place it to the device
cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
    discrete= False
).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

scores = reinforce(
    env, 
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    print_every=10,
)
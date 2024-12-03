import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )
        self.ac_dim = ac_dim
        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        obs = ptu.from_numpy(np.array([obs]))
        assert obs.shape[0] == 1
        action = self.forward(obs).sample().cpu().numpy()
        assert action.shape == (1,) if self.ac_dim == 1 else (1, self.ac_dim), f"action shape {action.shape}"
        return action[0]

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            activations = self.logits_net(obs)
            probs = nn.functional.log_softmax(activations).exp()
            return distributions.Categorical(probs = probs)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            x = self.mean_net(obs)
            x = torch.tanh(x)
            std = self.logstd.exp()
            m = torch.distributions.Normal(x, self.logstd.exp())
            return m

    def get_log_prob(self, obs, actions_taken):
        log_prob = self.forward(obs).log_prob(actions_taken)
        assert log_prob.shape == actions_taken.shape
        return self.forward(obs).log_prob(actions_taken)


    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        self.optimizer.zero_grad()
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        dist = self.forward(obs)
        loss = -torch.sum(dist.log_prob(actions))
        loss.backward()
        self.optimizer.step()
        return {
            "Loss": ptu.to_numpy(loss),
        }

class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        self.optimizer.zero_grad()
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        policy_loss = []
        for i in range(obs.shape[0]):
            ob = ptu.from_numpy(np.array([obs[i]]))
            ac = ptu.from_numpy(np.array([actions[i]]))
            log_prob = self.get_log_prob(ob, ac)
            policy_loss.append(-log_prob * advantages[i])
        loss = torch.cat(policy_loss).mean()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

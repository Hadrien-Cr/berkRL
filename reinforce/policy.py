import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size, discrete=True):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.discrete = discrete
        if not self.discrete:
            self.logstd = nn.Parameter(torch.zeros(1, a_size))

    def forward(self, x):
        if self.discrete:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            probs = F.softmax(x, dim=1).cpu()
            m = Categorical(probs)
            return m
        else:
            x = F.relu(self.fc1(x))
            x = torch.tanh(self.fc2(x)).cpu()
            m = torch.distributions.Normal(x, self.logstd.exp())
            return m

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        m = self.forward(state)
        if self.discrete: 
            action = m.sample()
            return action.item(), m.log_prob(action)
        else: 
            action = m.sample()[0]
            return np.array(action), m.log_prob(action)
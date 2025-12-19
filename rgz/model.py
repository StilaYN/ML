# model.py
import torch
import torch.nn as nn
from env import STATE_DIM

# model.py
import torch
import torch.nn as nn
from env import STATE_DIM

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, init_log_std=0.5):
        super().__init__()
        self.net = mlp([STATE_DIM, 128, 128, 64], activation=nn.ReLU)
        self.mean_head = nn.Linear(64, 1)
        self.log_std_head = nn.Linear(64, 1)
        self.apply(self._init_weights)
        nn.init.constant_(self.log_std_head.bias, init_log_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        features = self.net(x)
        mean = torch.tanh(self.mean_head(features))
        log_std = torch.clamp(self.log_std_head(features), -2.0, 2.0)
        std = torch.exp(log_std)
        return mean, std, log_std  # ← возвращаем log_std!

    def sample(self, state):
        mean, std, log_std = self.forward(state)  # ← получаем все три
        dist = torch.distributions.Normal(mean, std)
        action_raw = dist.rsample()
        action = torch.tanh(action_raw)
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mean, log_std  # ← теперь log_std определён

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = mlp([STATE_DIM + 1, 128, 128, 64, 1], activation=nn.ReLU)
        self.q2 = mlp([STATE_DIM + 1, 128, 128, 64, 1], activation=nn.ReLU)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.BatchNorm1d(400, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(400,300),
            nn.BatchNorm1d(300, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(300,action_dim),
        )
        self.linear_ac = nn.Sequential(
            nn.BatchNorm1d(1, track_running_stats=True),
            nn.Sigmoid(),
        ) 
        self.angular_ac = nn.Sequential(
            nn.BatchNorm1d(1, track_running_stats=True),
            nn.Tanh(),
        )
    def forward(self, x):
        x = self.fc(x)
        lin_ac = self.linear_ac(x[:,0])
        ang_ac = self.angular_ac(x[:,1])
        return torch.cat([lin_ac, ang_ac], dim=1)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim,400),
            nn.ReLU(),
            nn.Linear(400,300)
        ) 
        self.fc2 = nn.Linear(action_dim, 300)
        self.fc3 = nn.Sequential(
            nn.Linear(300,1)
        )
    def forward(self, state, action):
        s = self.fc1(state)
        a = self.fc2(action)
        out = F.relu(s+a)
        out = self.fc3(out)
        return out
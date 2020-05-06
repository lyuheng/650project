import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 400)
        #self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())    
        self.bn1 = nn.BatchNorm1d(400, track_running_stats=True)
        #nn.ReLU()
        self.fc2 = nn.Linear(400,300)
        #self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())    
        self.bn2 = nn.BatchNorm1d(300, track_running_stats=True)
        #nn.ReLU()
        self.fc3 = nn.Linear(300,action_dim)
        #self.fc3.weight.data.uniform_(-3e-3, 3e-3)    

        self.linear_ac = nn.Sequential(
            nn.BatchNorm1d(1, track_running_stats=True),
            nn.Sigmoid(),
        ) 
        self.angular_ac = nn.Sequential(
            nn.BatchNorm1d(1, track_running_stats=True),
            nn.Tanh(),
        )
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        lin_ac = self.linear_ac(x[:,0].unsqueeze(1))
        ang_ac = self.angular_ac(x[:,1].unsqueeze(1))
        return torch.cat([lin_ac, ang_ac], dim=1)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim,400)
        #self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        #nn.ReLU(),
        self.fc2 = nn.Linear(400,300)
        #self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
 
        self.fc3 = nn.Linear(action_dim, 300)
        #self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(300,1)
        #self.fc4.weight.data.uniform_(-3e-3, 3e-3)  

    def forward(self, state, action):
        s = F.relu(self.fc1(state))
        s = self.fc2(s)
        a = self.fc3(action)
        out = F.relu(s+a)
        out = self.fc4(out)
        return out
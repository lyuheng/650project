import os
import torch
import torch.nn.functional as F
import gym
import numpy as np

from model import Actor, Critic
from replaybuffer import ReplayBuffer

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
TAU = 0.001
BATCH_SIZE=64
GAMMA=0.99

MODEL_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model')

class DDPG:
    def __init__(self, env, state_dim, action_dim):
        self.name = 'DDPG'
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.AE = Actor(state_dim,action_dim).cuda()
        self.CE = Critic(state_dim,action_dim).cuda()
        self.AT = Actor(state_dim,action_dim).cuda()
        self.CT = Critic(state_dim,action_dim).cuda()
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.time_step = 0

        self.AE.load_state_dict(torch.load(MODEL_DIR+'/obs/actor_340000.pkl'))
        # self.AT.load_state_dict(torch.load(MODEL_DIR+'/actor_280000.pkl'))
        # self.CE.load_state_dict(torch.load(MODEL_DIR+'/critic_280000.pkl'))
        # self.CT.load_state_dict(torch.load(MODEL_DIR+'/critic_280000.pkl'))

        self.optimizer_a = torch.optim.Adam(self.AE.parameters(), lr=1e-4)
        self.optimizer_c = torch.optim.Adam(self.CE.parameters(), lr=1e-4)

    def train(self):
        self.AE.train()
        data = self.replay_buffer.get_batch(BATCH_SIZE)
        bs =  np.array([da[0] for da in data])
        ba =  np.array([da[1] for da in data])
        br =  np.array([da[2] for da in data])
        bs_ = np.array([da[3] for da in data])
        bd =  np.array([da[4] for da in data])

        bs = torch.FloatTensor(bs).cuda()
        ba = torch.FloatTensor(ba).cuda()
        br = torch.FloatTensor(br).cuda()
        bs_ = torch.FloatTensor(bs_).cuda()

        a_ = self.AT(bs_)
        #######################  NOTICE !!! #####################################
        #q1  = self.CE(bs,  a)  ###### here use action batch !!! for policy loss!!!
        q2  = self.CE(bs, ba) ###### here use computed batch !!! for value loss!!!
        ########################################################################
        q_ = self.CT(bs_, a_).detach()
        q_tar = torch.FloatTensor(BATCH_SIZE)
        for i in range(len(data)):
            if bd[i]:
                q_tar[i] = br[i]
            else:
                q_tar[i] = br[i]+GAMMA*q_[i]

        q_tar = q_tar.view(BATCH_SIZE,1).cuda()
        # minimize mse_loss of q2 and q_tar
        td_error = F.mse_loss(q2, q_tar.detach())  # minimize td_error
        self.CE.zero_grad()
        td_error.backward(retain_graph=True)
        self.optimizer_c.step()

        a = self.AE(bs)
        q1 = self.CE(bs, a)

        a_loss = -torch.mean(q1) # maximize q
        self.AE.zero_grad()
        a_loss.backward(retain_graph=True)
        self.optimizer_a.step()

        self.soft_replace()

    def soft_replace(self):
        for t,e in zip(self.AT.parameters(),self.AE.parameters()):
            t.data = (1-TAU)*t.data + TAU*e.data
        for t,e in zip(self.CT.parameters(),self.CE.parameters()):
            t.data = (1-TAU)*t.data + TAU*e.data

    def action(self, state):
        self.AE.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()  # add batch_sz=1
        ac_tensor = self.AE(state_tensor)
        ac = ac_tensor.squeeze(0).cpu().detach().numpy()
        return ac
    
    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)
        if self.replay_buffer.count() == REPLAY_START_SIZE:
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.time_step += 1
            self.train()

        if self.time_step % 10000 == 0 and self.time_step > 0:
            torch.save(self.AE.state_dict(), MODEL_DIR + '/obs/actor_{}.pkl'.format(self.time_step))
            torch.save(self.CE.state_dict(), MODEL_DIR + '/obs/critic_{}.pkl'.format(self.time_step))
            print('Save model state_dict successfully in obs dir...')

        return self.time_step


        



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:55:07 2018

@author: ZhangYaoZhong
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

gamma = 0.998
env = gym.make('CartPole-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self,
                 state_size=4,
                 h1_size=100,
                 h2_size=50,
                 action_size=2):
        super(Actor,self).__init__()
        self.fc_sh1 = nn.Linear(state_size,h1_size)
        nn.init.xavier_uniform_(self.fc_sh1.weight)
        self.fc_sh1.bias.data.fill_(0.01)
        
        self.fc_h1h2 = nn.Linear(h1_size,h2_size)
        nn.init.xavier_uniform_(self.fc_h1h2.weight)
        self.fc_h1h2.bias.data.fill_(0.01)
        
        self.fc_h2a = nn.Linear(h2_size,action_size)
        nn.init.xavier_uniform_(self.fc_h2a.weight)
        self.fc_h2a.bias.data.fill_(0.01)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self,state):
        h1 = self.relu(self.fc_sh1(state))
        h2 = self.relu(self.fc_h1h2(h1))
        action_score = self.fc_h2a(h2)
        action_score = self.softmax(action_score)
        return action_score

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.fc_sah1 = nn.Linear(4,100)
        nn.init.xavier_uniform_(self.fc_sah1.weight)
        self.fc_sah1.bias.data.fill_(0.01)
        
        self.fc_h1h2 = nn.Linear(100,50)
        nn.init.xavier_uniform_(self.fc_h1h2.weight)
        self.fc_h1h2.bias.data.fill_(0.01)
        
        self.fc_h2q = nn.Linear(50,1)
        nn.init.xavier_uniform_(self.fc_h2q.weight)
        self.fc_h2q.bias.data.fill_(0.01)
        
        self.relu = nn.ReLU()
        
    def forward(self,sa):
        h1 = self.relu(self.fc_sah1(sa))
        h2 = self.relu(self.fc_h1h2(h1))
        q = self.fc_h2q(h2)
        return q
        
pi_target = Actor().to(device)
q_target = Critic().to(device)

optim_pi_target = optim.RMSprop(pi_target.parameters(),lr=1e-4)
optim_q_target = optim.RMSprop(q_target.parameters(),lr=2e-4)

Huber_loss = nn.SmoothL1Loss()
Mseloss = nn.MSELoss()

duration = []

for i in range(2000):
    counter = 1
    I = 1.
    state = env.reset()
    done = False  
    while not done:
        env.render()
        action_score = pi_target(torch.from_numpy(state).float().to(device))
        p = Categorical(action_score)
        action = p.sample()
        state_next, reward, done, info = env.step(action.data.numpy())
        '''
        update Critic
        '''
        optim_q_target.zero_grad()
        q = q_target(torch.from_numpy(state).float().to(device))
        if done == True:
            q_next = torch.Tensor([-1]).float().to(device)
        else:
            q_next = q_target(torch.from_numpy(state_next).float().to(device)).detach()
        q_delta = q-gamma*q_next-reward
        q_loss = I*q_delta.pow(2)
        q_loss.backward()
        optim_q_target.step()
        '''
        updata Actor
        '''
        optim_pi_target.zero_grad()
        E_q = q_delta.data.detach()*I*p.log_prob(action)
        E_q.backward()
        for param in pi_target.parameters():
            param.grad.data.clamp_(-1,1)
        optim_pi_target.step()
        
        '''
        update enviroment
        '''
        state = state_next
        I *= gamma
        counter +=1
    print('duration:',counter)
    duration.append(counter)
    print(i,' th update')

plt.xlabel('train time')
plt.ylabel('duration')
plt.plot(np.arange(len(duration)),duration)
plt.show()

torch.save(pi_target,'pi_target.pkl')
torch.save(q_target,'q_target.pkl')

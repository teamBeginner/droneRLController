#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:17:30 2018

@author: ZhangYaoZhong
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

gamma = 0.99
env = gym.make('Pendulum-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
board = SummaryWriter()


class Actor(nn.Module):
    def __init__(self,
                 state_size=3,
                 h1_size=200,
                 h2_size=200,
                 action_size=1):
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

    def forward(self,state):
        h1 = self.relu(self.fc_sh1(state))
        h2 = self.relu(self.fc_h1h2(h1))
        action_score = self.fc_h2a(h2)
        action_score = 2*self.tanh(action_score)
        return action_score

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.fc_sah1 = nn.Linear(4,128)
        nn.init.xavier_uniform_(self.fc_sah1.weight)
        self.fc_sah1.bias.data.fill_(0.01)
        
        self.fc_h1h2 = nn.Linear(128,64)
        nn.init.xavier_uniform_(self.fc_h1h2.weight)
        self.fc_h1h2.bias.data.fill_(0.01)
        
        self.fc_h2q = nn.Linear(64,1)
        nn.init.xavier_uniform_(self.fc_h2q.weight)
        self.fc_h2q.bias.data.fill_(0.01)
        
        self.relu = nn.ReLU()
        
    def forward(self,s):
        h1 = self.relu(self.fc_sah1(s))
        h2 = self.relu(self.fc_h1h2(h1))
        q = self.fc_h2q(h2)
        return q
    
pi_target = Actor().to(device)
q_target = Critic().to(device)

optim_pi_target = optim.RMSprop(pi_target.parameters(),lr=2e-4)
optim_q_target = optim.RMSprop(q_target.parameters(),lr=4e-4)

def sim(sim_episode=60):
    batch = []
    s_batch = []
    s_next_batch = []
    a_batch = []
    a_next_batch = []
    r_batch = []
    state = env.reset()
    action = pi_target(torch.from_numpy(state).float().to(device)).data.cpu().numpy()
    for t in range(sim_episode):
        env.render()
        
        s_batch.append(state)
        a_batch.append(action)
        
        state_next, reward, done, info = env.step(action)
        action_next = pi_target(torch.from_numpy(state_next).float().to(device)).data.cpu().numpy()
        
        s_next_batch.append(state_next)
        a_next_batch.append(action_next)
        r_batch.append(reward)
        
        state,action = state_next,action_next
        
    s_batch = np.array(s_batch)
    s_next_batch = np.array(s_next_batch)
    a_batch = np.array(a_batch)
    a_next_batch = np.array(a_next_batch)
    r_batch = np.array(r_batch)
    batch = s_batch,s_next_batch,a_batch,a_next_batch,r_batch
    return batch

r_sum = []
mini_batch_size = 30
for i in range(5000):
    batch = sim()
    s_batch,s_next_batch,a_batch,a_next_batch,r_batch = batch
    r_sum.append(np.sum(r_batch))
    if i>1 and i%50==0:
        board.add_scalar('r_mean',np.mean(r_batch),i)
        board.add_scalar('r_var',np.var(r_batch),i)
        board.add_histogram('actor net 1 weight',pi_target.fc_sh1.state_dict()['weight'].cpu().numpy(),i)
        board.add_histogram('actor net 2 weight',pi_target.fc_h1h2.state_dict()['weight'].cpu().numpy(),i)
        board.add_histogram('actor net 3 weight',pi_target.fc_h2a.state_dict()['weight'].cpu().numpy(),i)
#        print(np.mean(r_batch))
    counter = s_batch.shape[0]
    mask = np.random.choice(np.arange(s_batch.shape[0]),mini_batch_size)
    s_mini_batch = s_batch[mask]
    s_next_mini_batch = s_next_batch[mask]
    a_mini_batch = a_batch[mask]
    a_next_mini_batch = a_next_batch[mask]
    r_mini_batch = r_batch[mask]
    
    a_mini_batch = torch.from_numpy(a_mini_batch).float().to(device)
    a_next_mini_batch = torch.from_numpy(a_next_mini_batch).float().to(device)
    r_mini_batch = torch.from_numpy(r_mini_batch).float().to(device)
    s_mini_batch = torch.from_numpy(s_mini_batch).float().to(device)
    s_next_mini_batch = torch.from_numpy(s_next_mini_batch).float().to(device)
    sa_mini_batch = torch.cat((s_mini_batch,a_mini_batch.view(-1,1)),1)
    sa_mini_batch.requires_grad = True
    sa_next_mini_batch = torch.cat((s_next_mini_batch,a_next_mini_batch.view(-1,1)),1)
    q = q_target(sa_mini_batch)
    q_next = q_target(sa_next_mini_batch).detach()
    
    '''
    updata Actor
    '''
    optim_pi_target.zero_grad()
    
    q_sum = -q.sum()
    q_sum.backward(retain_graph=True)
    
    action = pi_target(s_mini_batch)
    action.backward(sa_mini_batch.grad.data[:,3].view(-1,1))
    optim_pi_target.step()
    
    '''
        update Critic
    '''
    optim_q_target.zero_grad()
    
    q_delta = q-gamma*q_next-r_mini_batch
    q_loss = q_delta.pow(2).sum()
    q_loss.backward()
    optim_q_target.step()
    
    
    
torch.save(pi_target,'pi_target_DAC.pkl')
torch.save(q_target,'q_target_DAC.pkl')
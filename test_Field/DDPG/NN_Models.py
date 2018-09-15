#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:12:36 2018

@author: ZhangYaoZhong
"""
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HuberLoss = nn.functional.smooth_l1_loss
MSELoss = nn.functional.mse_loss


def gen_Deterministic_Actor(state_size,action_size,action_lim,learing_rate=2e-4):  
    class Actor(nn.Module):
        def __init__(self,
                     state_size,
                     action_size,
                     action_lim,
                     h1_size=256,
                     h2_size=256):
            super(Actor,self).__init__()
            self.fc_sh1 = nn.Linear(state_size,h1_size)
            nn.init.xavier_uniform_(self.fc_sh1.weight)
            self.fc_sh1.bias.data.fill_(0.02)
            
            self.fc_h1h2 = nn.Linear(h1_size,h2_size)
            nn.init.xavier_uniform_(self.fc_h1h2.weight)
            self.fc_h1h2.bias.data.fill_(0.02)
            
            self.fc_h2a = nn.Linear(h2_size,action_size)
            nn.init.xavier_uniform_(self.fc_h2a.weight)
            self.fc_h2a.bias.data.fill_(0.02)
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            
            self.bn = nn.BatchNorm1d(256)
            
        def forward(self,state):
            h1 = self.relu(self.fc_sh1(state))
            h2 = self.fc_h1h2(h1)
            h2 = self.relu(h2)
            action_score = self.fc_h2a(h2)
            action = self.tanh(action_score)*action_lim
            return action
    
    pi = Actor(state_size,action_size,action_lim).to(device)
    optim_pi = optim.RMSprop(pi.parameters(),lr=learing_rate)

    return pi,optim_pi





def gen_Critic(state_size,action_size,q_size=1,learnig_rate=2e-4): 
    class Critic(nn.Module):
        def __init__(self,
                     state_size,
                     action_size,
                     q_size,
                     h1_size=256,
                     h2_size=128
                     ):
            super(Critic,self).__init__()
            self.fc_sah1 = nn.Linear(state_size+action_size,h1_size)
            nn.init.xavier_uniform_(self.fc_sah1.weight)
            self.fc_sah1.bias.data.fill_(0.02)
            
            self.fc_h1h2 = nn.Linear(h1_size,h2_size)
            nn.init.xavier_uniform_(self.fc_h1h2.weight)
            self.fc_h1h2.bias.data.fill_(0.02)
            
            self.fc_h2q = nn.Linear(h2_size,q_size)
            nn.init.xavier_uniform_(self.fc_h2q.weight)
            self.fc_h2q.bias.data.fill_(0.02)
            
            self.relu = nn.ReLU()
            
            
        def forward(self,s,a):
            sa = torch.cat((s,a),1)
            h1 = self.relu(self.fc_sah1(sa))
            h2 = self.relu(self.fc_h1h2(h1))
            q = self.fc_h2q(h2)
            return q
        
    q = Critic(state_size,action_size,q_size).to(device)
    optim_q = optim.RMSprop(q.parameters(),lr=learnig_rate)
    
    return q,optim_q







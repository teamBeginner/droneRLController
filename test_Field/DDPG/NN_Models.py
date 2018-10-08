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


#def gen_Deterministic_Actor(state_dim,action_dim,action_lim,learing_rate=1e-4,h1_dim=256,h2_dim=128,h3_dim=128):  
#    class Actor(nn.Module):
#        def __init__(self,
#                     state_dim,
#                     action_dim,
#                     action_lim,
#                     h1_dim,
#                     h2_dim,
#                     h3_dim):
#            super(Actor,self).__init__()
#            self.fc_sh1 = nn.Linear(state_dim,h1_dim)
#            nn.init.xavier_uniform_(self.fc_sh1.weight)
#            self.fc_sh1.bias.data.fill_(0.02)
#            
#            self.fc_h1h2 = nn.Linear(h1_dim,h2_dim)
#            nn.init.xavier_uniform_(self.fc_h1h2.weight)
#            self.fc_h1h2.bias.data.fill_(0.02)
#            
#            self.fc_h2a = nn.Linear(h2_dim,action_dim)
#            nn.init.xavier_uniform_(self.fc_h2a.weight)
#            self.fc_h2a.bias.data.fill_(0.02)
#            
#            self.fc_h2h3 = nn.Linear(h2_dim,h3_dim)
#            nn.init.xavier_uniform_(self.fc_h2h3.weight)
#            self.fc_h2h3.bias.data.fill_(0.02)
#            
#            self.fc_h3a = nn.Linear(h3_dim,action_dim)
#            nn.init.xavier_uniform_(self.fc_h3a.weight)
#            self.fc_h3a.bias.data.fill_(0.02)
#            
#            self.tanh = nn.Tanh()
#            self.relu = nn.ReLU()
#            
#            self.h1_dim = h1_dim
#            self.h2_dim = h2_dim
#            self.h3_dim = h3_dim
#            self.action_lim = action_lim
##            self.bn_h2 = nn.BatchNorm1d(h2_dim)
#
#            
#        def forward(self,state,mode='sim'):
#            h1 = self.relu(self.fc_sh1(state))
#            h2 = self.fc_h1h2(h1)
#            h2 = self.relu(h2)
#            h3 = self.relu(self.fc_h2h3(h2))
#            action_score = self.fc_h3a(h3)
#            action = self.tanh(action_score)*self.action_lim
#            return action
#    
#    pi = Actor(state_dim,action_dim,action_lim,h1_dim,h2_dim,h3_dim).to(device)
#    optim_pi = optim.RMSprop(pi.parameters(),lr=learing_rate)
#
#    return pi,optim_pi
#
#
#
#
#
#def gen_Critic(state_dim,action_dim,q_dim=1,learnig_rate=4e-4,h1_dim=256,h2_dim=128): 
#    class Critic(nn.Module):
#        def __init__(self,
#                     state_dim,
#                     action_dim,
#                     q_dim,
#                     h1_dim,
#                     h2_dim):
#            super(Critic,self).__init__()
#            self.fc_sah1 = nn.Linear(state_dim+action_dim,h1_dim)
#            nn.init.xavier_uniform_(self.fc_sah1.weight)
#            self.fc_sah1.bias.data.fill_(0.02)
#            
#            self.fc_h1h2 = nn.Linear(h1_dim,h2_dim)
#            nn.init.xavier_uniform_(self.fc_h1h2.weight)
#            self.fc_h1h2.bias.data.fill_(0.02)
#            
#            self.fc_h2q = nn.Linear(h2_dim,1)
#            nn.init.xavier_uniform_(self.fc_h2q.weight)
#            self.fc_h2q.bias.data.fill_(0.02)
#            
#            self.relu = nn.ReLU()
#            
#            self.h1_dim = h1_dim
#            self.h2_dim = h2_dim
#            
#            self.bn_h1 = nn.BatchNorm1d(h1_dim)
#            
#        def forward(self,s,a,mode='sim'):
#            if len(a.size())>1:
#                sa = torch.cat((s,a),1)
#            else:
#                sa = torch.cat((s,a.view(-1,1)),1)
#            h1 = self.fc_sah1(sa)
#            h1 = self.relu(h1)
#            h2 = self.relu(self.fc_h1h2(h1))
#            q = self.fc_h2q(h2)
#            return q
#        
#    q = Critic(state_dim,action_dim,q_dim,h1_dim,h2_dim).to(device)
#    optim_q = optim.RMSprop(q.parameters(),lr=learnig_rate)
#    
#    return q,optim_q


class net(nn.Module):
    def __init__(self,config):
        super(net,self).__init__()
        layers = []
        for i in range(len(config['struct'])):
            layers.append(config['struct'][i])
        self.graph = nn.Sequential(*layers)
        
        if 'linear_Weight_Config' in config.keys():
            if len(config['linear_Weight_Config']) == 1:
                setup_Config_Linear_Weight(config['linear_Weight_Config'])
                self.graph.apply(init_Linear_Weight)
            else:
                index = 0
                for m in self.graph:
                    if type(m) == nn.Linear:
                        setup_Config_Linear_Weight(config['linear_Weight_Config'][index])
                        m.apply(init_Linear_Weight)
                        index += 1

        if 'linear_Bias_Config' in config.keys():
            if len(config['linear_Bias_Config']) == 1:
                setup_Config_Linear_Bias(config['linear_Bias_Config'][0])
                self.graph.apply(init_Linear_Bias)
            else:
                index = 0
                for m in self.graph:
                    if type(m) == nn.Linear:
                        setup_Config_Linear_Bias(config['linear_Bias_Config'][index])
                        m.apply(init_Linear_Bias)
                        index += 1

    def forward(self,input_info):
        return self.graph(input_info)

def setup_Config_Linear_Weight(config):
    
    global config_Weight_Mode,config_Weight_Param
    config_Weight_Mode = config[0]
    config_Weight_Param = config[1:]

def setup_Config_Linear_Bias(config):
    global config_Bias_Param
    config_Bias_Param = config
    
def init_Linear_Weight(m):
    if type(m) == nn.Linear:
        if config_Weight_Mode == 'uniform':
            nn.init.uniform_(m.weight,config_Weight_Param[0],config_Weight_Param[1])
        elif config_Weight_Mode == 'xavier_normal':
            nn.init.xavier_normal_(m,config_Weight_Param)
        elif config_Weight_Mode == 'xavier_uniform':
            nn.init.uniform_(m,config_Weight_Param)

def init_Linear_Bias(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(config_Bias_Param)





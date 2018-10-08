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


def gen_LowDim_Deterministic_Actor(state_dim,action_dim,action_lim,learing_rate=1e-4,h1_dim=400,h2_dim=300):  
    config = {'struct':[nn.Linear(state_dim,h1_dim),
                        nn.ReLU(),
                        nn.Linear(h1_dim,h2_dim),
                        nn.ReLU(),
                        nn.Linear(h2_dim,action_dim),
                        nn.Tanh(),
                        ],
              'linear_Weight_Config':[['uniform',-state_dim**0.5,state_dim**0.5],
                                      ['uniform',-h1_dim**0.5,h1_dim**0.5],
                                      ['uniform',-3e-3,3e-3]],
              'linear_Bias_Config':[0.01],
            }
    class actor(net):
        def __init__(self,config):
            net.__init__(self,config)
        def forward(self,state):
            score = self.graph(state)
            action = action_lim*score
            return action
    pi = net(config).to(device)
    optim_pi = optim.Adam(pi.parameters(),lr=learing_rate)

    return pi,optim_pi



def gen_LowDim_Deterministic_Critic(state_dim,action_dim,learnig_rate=1e-3,h1_dim=400,h2_dim=300): 
    config = {'struct':[nn.Linear(state_dim,h1_dim),
                        nn.ReLU(),
                        nn.Linear(h1_dim+action_dim,h2_dim),
                        nn.ReLU(),
                        nn.Linear(h2_dim,1),
                        ],
              'linear_Weight_Config':[['uniform',-state_dim**0.5,state_dim**0.5],
                                      ['uniform',-(h1_dim+action_dim)**0.5,(h1_dim+action_dim)**0.5],
                                      ['uniform',-3e-3,3e-3]],
              'linear_Bias_Config':[0.01],                  
            }
    
    class critic(net):
        def __init__(self,config):
            net.__init__(self,config)
        def forward(self,state,action):
            h = self.graph[0:2](state)
            if len(action.size())>1:
                q = self.graph[2:](torch.cat((h,action),1))
            else:
                q = self.graph[2:](torch.cat((h,action.view(-1,1)),1))
            return q
        
    q = critic(config).to(device)
    optim_q = optim.Adam(q.parameters(),lr=learnig_rate)
    
    return q,optim_q


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

a,z = gen_LowDim_Deterministic_Actor(4,1,2)
q,_ = gen_LowDim_Deterministic_Critic(4,1)
print(torch.FloatTensor(1).size())
print(q(torch.rand(4),torch.FloatTensor(1)))

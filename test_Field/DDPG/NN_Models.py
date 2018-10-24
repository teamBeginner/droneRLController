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


def gen_LowDim_Deterministic_Actor(state_dim,action_dim,action_lim,learing_rate=1e-4,h1_dim=400,h2_dim=300,h3_dim=128):  
#    config = {'struct':[nn.Linear(state_dim,h1_dim),
#                        nn.BatchNorm1d(h1_dim),
#                        nn.ReLU(),
#                        nn.Linear(h1_dim,h2_dim),
#                        nn.BatchNorm1d(h2_dim),                        
#                        nn.ReLU(),
#                        nn.Linear(h2_dim,action_dim),
#                        nn.BatchNorm1d(action_dim),
#                        nn.Tanh(),
#                        ],
#              'linear_Weight_Config':[['uniform',-state_dim**-0.5,state_dim**-0.5],
#                                      ['uniform',-h1_dim**-0.5,h1_dim**-0.5],
#                                      ['uniform',-3e-3,3e-3]],
#              'linear_Bias_Config':[0.01],
#            }
    class actor(nn.Module):
        def __init__(self):
            
            super(actor,self).__init__()
            
            self.fc1 = nn.Linear(state_dim,h1_dim)
            self.fc2 = nn.Linear(h1_dim,h2_dim)
            self.fc3 = nn.Linear(h2_dim,action_dim)
#            self.fc3 = nn.Linear(h2_dim,h3_dim)
#            self.fc4 = nn.Linear(h3_dim,action_dim)
            
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            
            self.bns = nn.BatchNorm1d(state_dim,affine=False,momentum=0.1)
            self.bn1 = nn.BatchNorm1d(h1_dim,affine=False,momentum=0.1)
            self.bn2 = nn.BatchNorm1d(h2_dim,affine=False,momentum=0.1)
            self.bn3 = nn.BatchNorm1d(action_dim,affine=False,momentum=0.1)
#            self.bn3 = nn.BatchNorm1d(h3_dim,momentum=0.7)
#            self.bn4 = nn.BatchNorm1d(action_dim,momentum=0.7)
            
#            self.Es = torch.zeros(state_dim)
#            self.Vars = torch.ones(state_dim)
#            self.E1 = torch.zeros(state_dim)
#            self.Var1 = torch.ones(state_dim)
#            self.E2 = torch.zeros(h1_dim)
#            self.Var2 = torch.ones(h1_dim)
#            self.E3 = torch.zeros(h2_dim)
#            self.Var3 = torch.ones(h2_dim)
            
            nn.init.uniform_(self.fc1.weight,-state_dim**(-0.5),state_dim**(-0.5))
            nn.init.uniform_(self.fc2.weight,-h1_dim**(-0.5),h1_dim**(-0.5))
            nn.init.uniform_(self.fc3.weight,-3e-3,3e-3)
#            nn.init.uniform_(self.fc3.weight,-h2_dim**(-0.5),h2_dim**(-0.5))
#            nn.init.uniform_(self.fc4.weight,-3e-3,3e-3)
            
            
            self.fc1.bias.data.fill_(0.01)
            self.fc2.bias.data.fill_(0.01)
            self.fc3.bias.data.fill_(0.01)
#            self.fc4.bias.data.fill_(0.01)
            
        def forward(self,state,mode):
            if mode == 'inference':
                
                s = (state-self.bns.running_mean)/((self.bns.running_var+self.bns.eps)**0.05)
#                *self.bns.weight+self.bns.bias
                
                h1 = self.fc1(s)
                h1_in = (h1-self.bn1.running_mean)/((self.bn1.running_var+self.bn1.eps)**0.5)
#                *self.bn1.weight\+self.bn1.bias
                h1_out = self.relu(h1_in)
                
                h2 = self.fc2(h1_out)
                h2_in = (h2-self.bn2.running_mean)/((self.bn2.running_var+self.bn2.eps)**0.5)
#                *self.bn2.weight+self.bn2.bias
                h2_out = self.relu(h2_in)
                
                h3 = self.fc3(h2_out)
                h3_in = (h3-self.bn3.running_mean)/((self.bn3.running_var+self.bn3.eps)**0.5)
#                *self.bn3.weight+self.bn3.bias
                h3_out = self.tanh(h3_in)
#                h3_out = self.relu(h3_in)
                
#                h4 = self.fc4(h3_out)
#                h4_in = (h4-self.bn4.running_mean)/((self.bn4.running_var+self.bn4.eps)**0.5)*self.bn4.weight\
#                     +self.bn4.bias
#                h4_out = self.tanh(h4_in)
                
                action = h3_out*action_lim
#                action = h4_out*action_lim
                
                return action
        
            elif mode == 'train':
                
                s_out = self.bns(state)
                
                h1 = self.fc1(s_out)
                h1_out = self.relu(self.bn1(h1))
                h2 = self.fc2(h1)
                h2_out = self.relu(self.bn2(h2))
                h3 = self.tanh(self.bn3(self.fc3(h2)))
#                h3 = self.fc3(h2_out)
#                h3_out = self.relu(self.bn3(h3))
#                h4 = self.tanh(self.bn4(self.fc4(h3_out)))
                action = h3*action_lim
#                action = h4*action_lim
                
                return action
            
#        def update_EaVar(self,Es,Vars,E1,Var1,E2,Var2,E3,Var3):
#            
#            self.Es = torch.tensor(Es).float().to(device)
#            self.Vars = torch.Tensor(Vars).float().to(device)
#            
#            self.E1 = torch.tensor(E1).float().to(device)
#            self.Var1 = torch.Tensor(Var1).float().to(device)
#            
#            self.E2 = torch.tensor(E2).float().to(device)
#            self.Var2 = torch.Tensor(Var2).float().to(device)
#            
#            self.E3 = torch.tensor(E3).float().to(device)
#            self.Var3 = torch.Tensor(Var3).float().to(device)
            
    pi = actor().to(device)
    optim_pi = optim.Adam(pi.parameters(),lr=learing_rate)

    return pi,optim_pi



def gen_LowDim_Deterministic_Critic(state_dim,action_dim,learnig_rate=1e-3,h1_dim=400,h2_dim=300): 
#    config = {'struct':[nn.BatchNorm1d(state_dim),
#                        nn.Linear(state_dim,h1_dim),
#                        nn.BatchNorm1d(h1_dim),
#                        nn.ReLU(),
#                        nn.Linear(h1_dim+action_dim,h2_dim),
#                        nn.BatchNorm1d(h2_dim),
#                        nn.ReLU(),
#                        nn.Linear(h2_dim,1),
#                        ],
#              'linear_Weight_Config':[['uniform',-state_dim**-0.5,state_dim**-0.5],
#                                      ['uniform',-(h1_dim+action_dim)**-0.5,(h1_dim+action_dim)**-0.5],
#                                      ['uniform',-3e-3,3e-3]],
#              'linear_Bias_Config':[0.01],                  
#            }
    
    class critic(nn.Module):
        def __init__(self):
            super(critic,self).__init__()
            self.fc1 = nn.Linear(state_dim+action_dim,h1_dim)
            self.fc2 = nn.Linear(h1_dim,h2_dim)
            self.fc3 = nn.Linear(h2_dim,1)
            
            self.bnsa = nn.BatchNorm1d(state_dim+action_dim,affine=False,momentum=0.1)
            self.bn1 = nn.BatchNorm1d(h1_dim,affine=False,momentum=0.1)
            self.bn2 = nn.BatchNorm1d(h2_dim,affine=False,momentum=0.1)
            
            self.relu = nn.ReLU()
            
        def forward(self,state,action):

            if len(action.size()) > 1:
                sa = torch.cat((state,action),1)
            elif len(action.size()) == 1:
                sa = torch.cat((state,action.view(-1,1)),1)
            elif len(state.size()) == 1:
                sa = torch.cat((state,action),0)
            
            sa = self.bnsa(sa)
            
            h1 = self.fc1(sa)
            h1_in = self.bn1(h1)
            h1_out = self.relu(h1_in)
            
            h2 = self.fc2(h1_out)
            h2_in = self.bn2(h2)
            h2_out = self.relu(h2_in)
            
            q = self.fc3(h2_out)
            
            return q
            
    q = critic().to(device)
    optim_q = optim.Adam(q.parameters(),lr=learnig_rate,weight_decay=1e-2)
    
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


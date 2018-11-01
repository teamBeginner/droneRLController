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


def gen_LowDim_Deterministic_Actor(state_dim,
                                   action_dim,
                                   state_max,
                                   action_max,
                                   learing_rate=1e-4,
                                   h1_dim=400,
                                   h2_dim=300,
                                   ):  
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
            
#            self.bns = nn.BatchNorm1d(state_dim,affine=True,momentum=0.1)
#            self.bn1 = nn.BatchNorm1d(h1_dim,affine=True,momentum=0.1)
#            self.bn2 = nn.BatchNorm1d(h2_dim,affine=True,momentum=0.1)
#            self.bn3 = nn.BatchNorm1d(action_dim,affine=True,momentum=0.1)


            
            nn.init.uniform_(self.fc1.weight,-state_dim**(-0.5),state_dim**(-0.5))
            nn.init.uniform_(self.fc2.weight,-h1_dim**(-0.5),h1_dim**(-0.5))
            nn.init.uniform_(self.fc3.weight,-3e-3,3e-3)
#            nn.init.uniform_(self.fc3.weight,-h2_dim**(-0.5),h2_dim**(-0.5))
#            nn.init.uniform_(self.fc4.weight,-3e-3,3e-3)
            
            
            self.fc1.bias.data.fill_(0.01)
            self.fc2.bias.data.fill_(0.01)
            self.fc3.bias.data.fill_(0.01)
#            self.fc4.bias.data.fill_(0.01)
            
        def forward(self,state):#,mode
            s = state/state_max
            h1 = self.relu(self.fc1(s))
            h2 = self.relu(self.fc2(h1))
            action_score = self.fc3(h2)
            
            action = action_score*action_max
            
            return action
#            abondon forward function with batch normalization method
#            if mode == 'inference':
#                
#                s = (state-self.bns.running_mean)/((self.bns.running_var+self.bns.eps)**0.05)\
#                *self.bns.weight+self.bns.bias
#                
#                h1 = self.fc1(s)
#                h1_in = (h1-self.bn1.running_mean)/((self.bn1.running_var+self.bn1.eps)**0.5)\
#                *self.bn1.weight+self.bn1.bias
#                h1_out = self.relu(h1_in)
#                
#                h2 = self.fc2(h1_out)
#                h2_in = (h2-self.bn2.running_mean)/((self.bn2.running_var+self.bn2.eps)**0.5)\
#                *self.bn2.weight+self.bn2.bias
#                h2_out = self.relu(h2_in)
#                
#                h3 = self.fc3(h2_out)
#                h3_in = (h3-self.bn3.running_mean)/((self.bn3.running_var+self.bn3.eps)**0.5)\
#                *self.bn3.weight+self.bn3.bias
#                h3_out = self.tanh(h3_in)
##                h3_out = self.relu(h3_in)
#                
##                h4 = self.fc4(h3_out)
##                h4_in = (h4-self.bn4.running_mean)/((self.bn4.running_var+self.bn4.eps)**0.5)*self.bn4.weight\
##                     +self.bn4.bias
##                h4_out = self.tanh(h4_in)
#                
#                action = h3_out*action_max
##                action = h4_out*action_max
#                
#                return action
#        
#            elif mode == 'train':
#                
#                s_out = self.bns(state)
#                
#                h1 = self.fc1(s_out)
#                h1_out = self.relu(self.bn1(h1))
#                h2 = self.fc2(h1)
#                h2_out = self.relu(self.bn2(h2))
#                h3 = self.tanh(self.bn3(self.fc3(h2)))
##                h3 = self.fc3(h2_out)
##                h3_out = self.relu(self.bn3(h3))
##                h4 = self.tanh(self.bn4(self.fc4(h3_out)))
#                action = h3*action_max
##                action = h4*action_max
#                
#                return action
            
    pi = actor().to(device)
    optim_pi = optim.Adam(pi.parameters(),lr=learing_rate)

    return pi,optim_pi



def gen_LowDim_Deterministic_Critic(state_dim,
                                    action_dim,
                                    state_max,
                                    action_max,
                                    learnig_rate=1e-3,
                                    h1_dim=400,
                                    h2_dim=300,): 
    
    class critic(nn.Module):
        def __init__(self):
            super(critic,self).__init__()
            self.fc1 = nn.Linear(state_dim+action_dim,h1_dim)
            self.fc2 = nn.Linear(h1_dim,h2_dim)
            self.fc3 = nn.Linear(h2_dim,1)
            
            self.bnsa = nn.BatchNorm1d(state_dim+action_dim,affine=True,momentum=0.1)
            self.bn1 = nn.BatchNorm1d(h1_dim,affine=True,momentum=0.1)
            self.bn2 = nn.BatchNorm1d(h2_dim,affine=True,momentum=0.1)
            
            self.relu = nn.ReLU()
            
        def forward(self,state,action):

            s = state/state_max
            a = action/action_max
            
            if len(action.size()) > 1:
                sa = torch.cat((s,a),1)
            elif len(action.size()) == 1:
                sa = torch.cat((s,a.view(-1,1)),1)
            elif len(state.size()) == 1:
                sa = torch.cat((s,a),0)
            
#            sa = self.bnsa(sa)
            
            h1 = self.fc1(sa)
#            h1 = self.bn1(h1)
            h1_out = self.relu(h1)
            
            h2 = self.fc2(h1_out)
#            h2 = self.bn2(h2)
            h2_out = self.relu(h2)
            
            q = self.fc3(h2_out)
            
            return q
            
    q = critic().to(device)
    optim_q = optim.Adam(q.parameters(),lr=learnig_rate,weight_decay=1e-2)
    
    return q,optim_q







def genaral_graph(config):

    layers = []
    for i in range(len(config['struct'])):
        layers.append(config['struct'][i])
    graph = nn.Sequential(*layers)
    
    if 'linear_Weight_Config' in config.keys():
        if len(config['linear_Weight_Config']) == 1:
            setup_Config_Linear_Weight(config['linear_Weight_Config'])
            graph.apply(init_Linear_Weight)
        else:
            index = 0
            for m in graph:
                if type(m) == nn.Linear:
                    setup_Config_Linear_Weight(config['linear_Weight_Config'][index])
                    m.apply(init_Linear_Weight)
                    index += 1

        if 'linear_Bias_Config' in config.keys():
            if len(config['linear_Bias_Config']) == 1:
                setup_Config_Linear_Bias(config['linear_Bias_Config'][0])
                graph.apply(init_Linear_Bias)
            else:
                index = 0
                for m in graph:
                    if type(m) == nn.Linear:
                        setup_Config_Linear_Bias(config['linear_Bias_Config'][index])
                        m.apply(init_Linear_Bias)
                        index += 1
    return graph
        
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


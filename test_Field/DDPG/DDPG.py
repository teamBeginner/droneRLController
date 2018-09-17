#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:34:47 2018

@author: ZhangYaoZhong
"""

import gym
import numpy as np
from tensorboardX import SummaryWriter
import NN_Models
import utils
import simulator
import torch

tau = 0.001
gamma = 0.99
mini_batch_size = 30
env = gym.make('Pendulum-v0')
board = SummaryWriter()

pendulum = simulator.Sim_Gym('Pendulum-v0')
s_dim = pendulum.state_dim
a_dim = pendulum.action_dim
a_lim = float(pendulum.action_lim)

pi_e,optim_pi_e = NN_Models.gen_Deterministic_Actor(s_dim,a_dim,a_lim,learing_rate=1e-4)
pi_t,_ = NN_Models.gen_Deterministic_Actor(s_dim,a_dim,a_lim)
q_e,optim_q_e = NN_Models.gen_Critic(s_dim,a_dim,learnig_rate=4e-4)
q_t,_ = NN_Models.gen_Critic(s_dim,a_dim)

utils.direct_copy(pi_e,pi_t)
utils.direct_copy(q_e,q_t)
    
for i in range(20000):
    batch = pendulum.sim_explore(pi_e,T=100)
   
    mini_batch = utils.sample_mini_batch(batch,mini_batch_size)
    s_mini_batch,s_next_mini_batch,a_mini_batch,r_mini_batch = mini_batch
    
    if i%50==0:
        board.add_scalar('r_mean',np.mean(r_mini_batch),i)
        board.add_scalar('r_var',np.var(r_mini_batch),i)
        
    a_mini_batch = torch.from_numpy(a_mini_batch).float().to(NN_Models.device)
    r_mini_batch = torch.from_numpy(r_mini_batch).float().to(NN_Models.device)
    s_mini_batch = torch.from_numpy(s_mini_batch).float().to(NN_Models.device)
    s_next_mini_batch = torch.from_numpy(s_next_mini_batch).float().to(NN_Models.device)
    
    a_next_mini_batch = pi_t(s_next_mini_batch).detach()
    
    y = r_mini_batch.view(-1,1)+gamma*q_t(s_next_mini_batch,a_next_mini_batch.view(-1,1)).detach()
    
    '''
    update Critic Q_explore(s,a)
    '''
    optim_q_e.zero_grad()
    q = q_e(s_mini_batch,a_mini_batch.view(-1,1))
    loss_e = NN_Models.HuberLoss(q,y)
    loss_e.backward()
    optim_q_e.step()
    '''
    update Actor Pi_explore(s)
    '''
    optim_pi_e.zero_grad()
    a_e = pi_e(s_mini_batch)
    q_sum = -q_e(s_mini_batch,a_e.view(-1,1)).mean()
    q_sum.backward()
    optim_pi_e.step()
    '''
    soft update Critic Q_target(s,a)
    '''
    utils.soft_update(q_e,q_t,tau)
    
    '''
    soft update Actor Pi_target(s)
    '''
    utils.soft_update(pi_e,pi_t,tau)
    










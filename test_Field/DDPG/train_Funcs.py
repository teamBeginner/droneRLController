#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:46:32 2018

@author: ZhangYaoZhong
"""
import numpy as np
#from tensorboardX import SummaryWriter
import NN_Models
import utils
import simulator
import torch

gym_list = ['Pendulum-v0']


def train_DDPG(config_DDPG,config_sim,config_Actor,config_Critic):
    
    sim_steps,env_name,do_render = config_sim
    train_eps,tau,gamma,mini_batch_size = config_DDPG
    h1_dim_pi,h2_dim_pi,h3_dim_pi,lr_pi = config_Actor
    q_dim,h1_dim_q,h2_dim_q,lr_q = config_Critic
    
    if env_name in gym_list:
        env = simulator.Sim_Gym(env_name,do_render)
        
    elif env_name == 'quard_Copter':
        env = simulator.Sim_QC()
        
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_lim = float(env.action_lim)
    
    pi_e,optim_pi_e = NN_Models.gen_Deterministic_Actor(s_dim,a_dim,a_lim,\
                                                        lr_pi,h1_dim_pi,h2_dim_pi,h3_dim_pi)
    pi_t,_ = NN_Models.gen_Deterministic_Actor(s_dim,a_dim,a_lim,lr_pi,\
                                               h1_dim_pi,h2_dim_pi,h3_dim_pi)
    q_e,optim_q_e = NN_Models.gen_Critic(s_dim,a_dim,q_dim,lr_q,h1_dim_q,h2_dim_q)
    q_t,_ = NN_Models.gen_Critic(s_dim,a_dim,q_dim,lr_q,h1_dim_q,h2_dim_q)
    
    utils.direct_copy(pi_e,pi_t)
    utils.direct_copy(q_e,q_t)
    
        
    for i in range(train_eps):
        batch = env.sim_explore(pi_e,sim_steps,do_render)
       
        mini_batch = utils.sample_mini_batch(batch,mini_batch_size)
        s_mini_batch,s_next_mini_batch,a_mini_batch,r_mini_batch = mini_batch
            
        a_mini_batch = torch.from_numpy(a_mini_batch).float().to(NN_Models.device)
        r_mini_batch = torch.from_numpy(r_mini_batch).float().to(NN_Models.device)
        s_mini_batch = torch.from_numpy(s_mini_batch).float().to(NN_Models.device)
        s_next_mini_batch = torch.from_numpy(s_next_mini_batch).float().to(NN_Models.device)
        
        a_next_mini_batch = pi_t(s_next_mini_batch).detach()
        
        
        y = r_mini_batch.view(-1,1)+gamma*q_t(s_next_mini_batch,a_next_mini_batch).detach()
        
        '''
        update Critic Q_explore(s,a)
        '''
        optim_q_e.zero_grad()
        q = q_e(s_mini_batch,a_mini_batch)
        loss_e = NN_Models.HuberLoss(q,y)
        loss_e.backward()
        optim_q_e.step()
        '''
        update Actor Pi_explore(s)
        '''
        optim_pi_e.zero_grad()
        a_e = pi_e(s_mini_batch)
        q_sum = -q_e(s_mini_batch,a_e).mean()
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
        
        '''
        save model
        '''
        if i>0 and i%20000 == 0:
            file_name = 'pi_iter'+str(i)+'.pth'
            torch.save(pi_t.state_dict(),file_name)
            



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:46:32 2018

@author: ZhangYaoZhong
"""
import numpy as np
import NN_Models
import utils
import simulator
import torch

board = utils.Board()

gym_list = ['Pendulum-v0','CartPole-v1','LunarLander-v2',]


def train_DDPG(config_DDPG,config_sim,config_Actor,config_Critic):
    
    sim_steps,env_name,do_render = config_sim
    train_eps,tau,gamma,mini_batch_size = config_DDPG
    lr_pi,h1_dim_pi,h2_dim_pi = config_Actor#,h3_dim_pi
    lr_q,h1_dim_q,h2_dim_q = config_Critic
    
    env = simulator.Simulator(env_name)
            
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_lim = float(env.action_lim)
    
    s_init = env.env.reset()
    a_init = env.env.action_space.sample()
    s_next_init,r_init,_,_ = env.env.step(a_init)
    memory = {'state':s_init,
              'action':a_init,
              'state_next':s_next_init,
              'reward':r_init,}
    memory_size = 2e5
        
    pi_e,optim_pi_e = NN_Models.gen_LowDim_Deterministic_Actor(s_dim,a_dim,a_lim,\
                                                        lr_pi,h1_dim_pi,h2_dim_pi)
    pi_t,_ = NN_Models.gen_LowDim_Deterministic_Actor(s_dim,a_dim,a_lim,lr_pi,\
                                               h1_dim_pi,h2_dim_pi)
    q_e,optim_q_e = NN_Models.gen_LowDim_Deterministic_Critic(s_dim,a_dim,lr_q,h1_dim_q,h2_dim_q)
    q_t,_ = NN_Models.gen_LowDim_Deterministic_Critic(s_dim,a_dim,lr_q,h1_dim_q,h2_dim_q)
    
    utils.direct_copy(pi_e,pi_t)
    utils.direct_copy(q_e,q_t)
    
        
    for i in range(train_eps):
#        pi_e.eval()
        batch = env.sim_explore_DDPG(pi_e,sim_steps,do_render)
        
        utils.update_trasitions(memory,batch,memory_size)
        
        mini_batch = utils.sample_mini_batch(memory,mini_batch_size)
        s_mini_batch,s_next_mini_batch,a_mini_batch,r_mini_batch = mini_batch
            
        a_mini_batch = torch.from_numpy(a_mini_batch).float().to(NN_Models.device)
        r_mini_batch = torch.from_numpy(r_mini_batch).float().to(NN_Models.device)
        s_mini_batch = torch.from_numpy(s_mini_batch).float().to(NN_Models.device)
        s_next_mini_batch = torch.from_numpy(s_next_mini_batch).float().to(NN_Models.device)
        
        a_next_mini_batch = pi_t(s_next_mini_batch,mode='train').detach()#,mode='train'
        
        
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
        pi_e.train()
        optim_pi_e.zero_grad()
        a_e = pi_e(s_mini_batch,mode='train')#mode='train'
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
        tensorboard visulization
        '''
        _,_,_,r_batch = batch
        print(i,' th update : ',r_batch.mean())
#        if i%500 == 0:
#            board.scalar(i,r_mean=r_batch.mean())
        
        '''
        save model
        '''
        if i>0 and i%2000 == 0:
            file_name = 'pi_iter'+str(i)+'.pth'
            torch.save(pi_t.state_dict(),file_name)
            


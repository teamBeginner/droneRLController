#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:46:32 2018

@author: ZhangYaoZhong
"""
import NN_Models
import utils
import simulator
import torch


def DDPG_naive(config_train,config_Actor,config_Critic):#config_DDPG,config_sim,config_Actor,config_Critic
    
#    utils.declare_dict_args(config_train)
#    utils.declare_dict_args(config_Actor)
#    utils.declare_dict_args(config_Critic)
    '''
    A naive DDPG implementation without batch normalization for its lacking
    in stationarity of states in many simulation enviroments.(well,at least
    that's what I've found out)
    
    inputs: config_train, config_Actor, config_Critic.
    ---------------------------------------------------------------------
    
    config_train provides configuration parameters for trainning process.
    includes:
        sim_steps
        --how many steps to run for each simulation
        
        env_name
        --the name of the simulation enviroment
        
        do_render_train
        --render flag during training
        
        do_demonstrate
        --demonstrate flag after training is complete
        
        tau,gamma,theta,mu,sig,mini_batch_size
        --algrithm parameters
        
        mini_batch_ancor
        --ancor ,choose the closest ancor number that is bigger or equal to 
        training episode to get the mini_batch_size with same list index 
        corresponding to ancor list during training
        
        on_TensorBoard
        --TensorBoard plot flag
        
        save_model_interval
        --save the model every interval
    ---------------------------------------------------------------------
    
    config_Actor provides configuration parameters for Actor nerual network
    initialization.
    includes:
        lr
        --learning rate
        h_dim
        --list contained with hidden layer dimension size
    ---------------------------------------------------------------------
    
    config_Critic provides configuration parameters for Critic nerual network
    initialization.
    includes:
        lr
        --learning rate
        h_dim
        --list contained with hidden layer dimension size
    '''
    
    sim_steps = config_train['sim_steps']
    env_name = config_train['env_name']
    do_render_train = config_train['do_render_train']
    do_demonstrate = config_train['do_demonstrate']
    train_eps = config_train['train_eps']
    tau = config_train['tau']
    gamma = config_train['gamma']
    mini_batch_size = config_train['mini_batch_size']
    mini_batch_ancor = config_train['mini_batch_ancor']
    on_TensorBoard = config_train['on_TensorBoard']
    theta = config_train['theta']
    sig = config_train['sig']
    save_model_interval = config_train['save_model_interval']
    
    lr_pi = config_Actor['lr']
    h1_dim_pi,h2_dim_pi = config_Actor['h_dim']
    
    lr_q = config_Critic['lr']
    h1_dim_q,h2_dim_q = config_Critic['h_dim']
    
    env = simulator.Simulator(env_name)
            
    s_dim = env.state_dim
    s_max = torch.from_numpy(env.state_max).to(NN_Models.device)
    a_dim = env.action_dim
    a_max = float(env.action_max)
    
    s_init = env.env.reset()
    a_init = env.env.action_space.sample()
    s_next_init,r_init,_,_ = env.env.step(a_init)
    
    memory = {'state':s_init,
              'action':a_init,
              'state_next':s_next_init,
              'reward':r_init,}
    
    memory_size = 1e5
        
    pi_e,optim_pi_e = NN_Models.gen_LowDim_Deterministic_Actor(s_dim,a_dim,s_max,a_max,\
                                                        lr_pi,h1_dim_pi,h2_dim_pi)
    pi_t,_ = NN_Models.gen_LowDim_Deterministic_Actor(s_dim,a_dim,s_max,a_max,lr_pi,\
                                               h1_dim_pi,h2_dim_pi)
    q_e,optim_q_e = NN_Models.gen_LowDim_Deterministic_Critic(s_dim,a_dim,s_max,a_max,\
                                                              lr_q,h1_dim_q,h2_dim_q)
    q_t,_ = NN_Models.gen_LowDim_Deterministic_Critic(s_dim,a_dim,s_max,a_max,lr_q,\
                                                      h1_dim_q,h2_dim_q)
    
    utils.direct_copy(pi_e,pi_t)
    utils.direct_copy(q_e,q_t)
    
    mbs = mini_batch_size[0]
    
    if on_TensorBoard:
        board = utils.Board()

    for ep in range(train_eps):
        for i in range(len(mini_batch_ancor)):  
            if ep >= mini_batch_ancor[i]:
                mbs = mini_batch_size[i]
                break
        
        batch = env.sim_explore_DDPG(pi_e,sim_steps,do_render_train,theta,sig)
        
        utils.update_trasitions(memory,batch,memory_size)
        
        mini_batch = utils.sample_mini_batch(memory,mbs)
        s_mini_batch,s_next_mini_batch,a_mini_batch,r_mini_batch = mini_batch
            
        a_mini_batch = torch.from_numpy(a_mini_batch).float().to(NN_Models.device)
        r_mini_batch = torch.from_numpy(r_mini_batch).float().to(NN_Models.device)
        s_mini_batch = torch.from_numpy(s_mini_batch).float().to(NN_Models.device)
        s_next_mini_batch = torch.from_numpy(s_next_mini_batch).float().to(NN_Models.device)
        
        a_next_mini_batch = pi_t(s_next_mini_batch).detach()#,mode='train'
        
        
        y = r_mini_batch.view(-1,1)+gamma*q_t(s_next_mini_batch,a_next_mini_batch).detach()
        
        '''
        update Critic Q_explore(s,a)
        '''
        
        optim_q_e.zero_grad()
        q = q_e(s_mini_batch,a_mini_batch)
        loss_e = NN_Models.MSELoss(q,y)
        loss_e.backward()
        optim_q_e.step()
        
        '''
        update Actor Pi_explore(s)
        '''
        
        optim_pi_e.zero_grad()
        a_e = pi_e(s_mini_batch)#mode='train'
        q_mean = -q_e(s_mini_batch,a_e).mean()
        q_mean.backward()
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
        print r_mean
        tensorboard visulization
        '''
        if ep%20 ==0:
            _,_,_,r_batch_e = batch
            _,_,_,r_batch_t = env.sim_inference_DDPG(pi_t,sim_steps,do_render_train)
            print('exploration actor ',ep,' th update : ',r_batch_e.mean())
            print('target actor ',ep,' th update : ',r_batch_t.mean())

        if on_TensorBoard:
            if ep%200 == 0:
                board.scalar(ep,r_mean_explore=r_batch_e.mean(),
                             r_mean_inference=r_batch_t.mean())
                board.hist(ep,)
            

        
        '''
        save model
        '''
        if ep>0 and ep%save_model_interval == 0:
            file_name = 'pi_iter'+str(ep)+'.pth'
            torch.save(pi_t.state_dict(),file_name)
            
    '''
    demonstrate
    '''
    if do_demonstrate:
        env.sim_inference_DDPG(pi_t,sim_steps,do_demonstrate)
            


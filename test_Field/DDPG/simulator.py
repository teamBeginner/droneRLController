#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:33:21 2018

@author: ZhangYaoZhong
"""

import gym
import numpy as np
import torch
import utils
import NN_Models
import qc


device = NN_Models.device

gym_list = ['Pendulum-v0','CartPole-v1','LunarLander-v2',]

class Simulator(object):
    
    def __init__(self,env_name):
        
        self.env_name = env_name
        
        if env_name in gym_list:
            self.env = gym.make(env_name)
            
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.action_lim = self.env.action_space.high[0]
            self.dt = 0.05
            
        elif env_name == 'quard_Copter':
            self.env = qc.quard_Copter()
            
            self.action_dim = self.env.action_dim
            self.action_lim = self.env.action_lim
            self.state_dim = self.env.state_dim
            self.dt = self.env.dt
            
    def sim_explore_DDPG(self,pi,T=200,do_render=False):
        batch = []
        s_batch = []
        s_next_batch = []
        a_batch = []
        r_batch = []
        state = self.env.reset()
        if self.env_name == 'quard_Copter':
            state = np.hstack(state) 
        
        OU = utils.OU_process(self.action_dim,self.dt)
        OU.reset()
        
        t = 0
        done = False
        
        while t <= T and not done:
            if do_render:         
                self.env.render()
            
            pi.eval()
            
            action = pi(torch.from_numpy(state).float().to(device),\
                        mode='inference').data.cpu().numpy()#mode='inference'
            
            pi.train()
            
            action += OU.sample()
            
            action = np.clip(action,-self.action_lim,self.action_lim)
            
            s_batch.append(state)
            a_batch.append(action)
            
            state_next, reward, done, info = self.env.step(action)

            s_next_batch.append(state_next)
            r_batch.append(reward)
            
            state = state_next
            
            t += 1
        s_batch = np.array(s_batch)
        s_next_batch = np.array(s_next_batch)
        a_batch = np.array(a_batch)
        r_batch = np.array(r_batch)
        batch = s_batch,s_next_batch,a_batch,r_batch
        return batch
    
    


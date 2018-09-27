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
import vis3D


device = NN_Models.device

class Sim_Gym(object):
    def __init__(self,env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_lim = self.env.action_space.high[0]
        
    def sim_explore(self,pi,T=200,do_render=True):
        batch = []
        s_batch = []
        s_next_batch = []
        a_batch = []
        r_batch = []
        state = self.env.reset()
        OU = utils.OU_process(self.action_dim,dt=0.05)
        OU.reset()
        for t in range(T):
            if do_render:         
                self.env.render()
            
            action = pi(torch.from_numpy(state).float().to(device)).data.cpu().numpy()
            action += OU.sample()
            
            action = np.clip(action,-self.action_lim,self.action_lim)
            
            s_batch.append(state)
            a_batch.append(action)
            
            state_next, reward, done, info = self.env.step(action)

            s_next_batch.append(state_next)
            r_batch.append(reward)
            
            state = state_next
            
        s_batch = np.array(s_batch)
        s_next_batch = np.array(s_next_batch)
        a_batch = np.array(a_batch)
        r_batch = np.array(r_batch)
        batch = s_batch,s_next_batch,a_batch,r_batch
        return batch

class Sim_QC(object):
    def __init__(self):
        self.env = qc.quard_Copter()
        self.action_dim = self.env.action_dim
        self.action_lim = self.env.action_lim
        self.state_dim = self.env.state_dim
        self.dt = self.env.dt
        
    def sim_explore(self,pi,T=200,do_render=False):
        batch = []
        s_batch = []
        s_next_batch = []
        a_batch = []
        r_batch = []
        state = self.env.reset()
        state = np.hstack(state) 
        
        OU = utils.OU_process(self.action_dim,self.dt)
        OU.reset()
        done = False
        t = 0


        while not done and t<=T:           
            action = pi(torch.from_numpy(state).float().to(device)).data.cpu().numpy()
            action += OU.sample()*self.action_lim
            
            action = np.clip(action,-self.action_lim,self.action_lim)
            
            s_batch.append(state)
            a_batch.append(action)
            
            state_next, reward, done = self.env.step(action)
            
            state_next = np.hstack(state_next)
            s_next_batch.append(state_next)
            r_batch.append(reward)
            
            state = state_next
            
            print(self.env.P)
            print(self.env.Angle)
            print(reward)
            
            t += 1
        s_batch = np.array(s_batch)
        s_next_batch = np.array(s_next_batch)
        a_batch = np.array(a_batch)
        r_batch = np.array(r_batch)
        batch = s_batch,s_next_batch,a_batch,r_batch
        return batch
    
    


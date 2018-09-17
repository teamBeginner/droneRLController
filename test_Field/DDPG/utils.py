#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:03:58 2018

@author: ZhangYaoZhong
"""
import numpy as np

def direct_copy(source,target):
    for param_s,param_t in zip(source.parameters(),target.parameters()):
        param_t.data.copy_(param_s.data)



def soft_update(source,target,tau):
    for param_s,param_t in zip(source.parameters(),target.parameters()):
        param_t.data.copy_(param_s.data*tau+(1-tau)*param_t.data)



class OU_process(object):
    def __init__(self,action_dim,dt=0.02,theta=0.1,mu=0.,sig=0.3):
        self.X = np.ones(action_dim)*mu
        self.theta = theta
        self.mu = mu
        self.sig = sig
        self.dt = dt
        self.action_dim = action_dim
        
    def sample(self):
        dX = self.theta*(self.mu-self.X)+self.sig*np.sqrt(self.dt)*np.random.randn(self.X.shape[0])
        self.X += dX
        return self.X
    
    def update(self,X_new):
        self.X = X_new
    
    def reset(self):
        self.X = np.ones(self.action_dim)*self.mu
    
def sample_mini_batch(batch,mini_batch_size):
    s_batch,s_next_batch,a_batch,r_batch = batch
    mask = np.random.choice(np.arange(s_batch.shape[0]),mini_batch_size)
    s_mini_batch = s_batch[mask]
    s_next_mini_batch = s_next_batch[mask]
    a_mini_batch = a_batch[mask]
    r_mini_batch = r_batch[mask]
    mini_batch = s_mini_batch,s_next_mini_batch,a_mini_batch,r_mini_batch
    return mini_batch






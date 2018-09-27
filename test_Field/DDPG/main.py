#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:55:15 2018

@author: zhangmingkun
"""

from train_Funcs import *

config_sim = [150,'quard_Copter',False]
config_DDPG = [500000,0.001,0.99,50]
config_Actor = [512,512,256,1e-4]
config_Critic = [1,512,256,4e-4]

'''
    sim_steps,env_name,do_render = config_sim
    train_eps,tau,gamma,mini_batch_size = config_DDPG
    h1_dim_pi,h2_dim_pi,h3_dim_pi,lr_pi = config_Actor
    q_dim,h1_dim_q,h2_dim_q,lr_q = config_Critic
'''
train_DDPG(config_DDPG,config_sim,config_Actor,config_Critic)
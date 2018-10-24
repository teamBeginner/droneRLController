#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:55:15 2018

@author: zhangmingkun
"""

from train_Funcs import *

config_sim = [80,'Pendulum-v0',True]
config_DDPG = [50000,0.001,0.99,40]
config_Actor = [1e-4,400,300]#,128
config_Critic = [1e-3,400,300]

'''
    sim_steps,env_name,do_render = config_sim
    train_eps,tau,gamma,mini_batch_size = config_DDPG
    lr_pi,h1_dim_pi,h2_dim_pi = config_Actor
    lr_q,q_dim,h1_dim_q,h2_dim_q = config_Critic
'''
train_DDPG(config_DDPG,config_sim,config_Actor,config_Critic)
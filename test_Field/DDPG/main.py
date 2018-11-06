#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:55:15 2018

@author: zhangmingkun
"""

from train_Funcs import *

config_train = {'sim_steps':80,
                'env_name':'Pendulum-v0',
                'do_demonstrate':True,
                'do_render_train':False,
                'train_eps':8000,
                'tau':1e-3,
                'gamma':0.99,
                'mini_batch_size':[64,100,120],
                'mini_batch_ancor':[0,400,800],
                'theta':0.15,
                'sig':0.2,
                'on_TensorBoard':False,
                'save_model_interval':1000,
                'comparation':False,
                'memory_size':1e4,}

config_Actor = {'lr':2e-3,
                'h_dim':[400,300],
                'optimizer':'RMSprop',}

config_Critic = {'lr':1e-2,
                 'h_dim':[400,300],
                 'optimizer':'RMSprop',}


DDPG_naive(config_train,config_Actor,config_Critic)
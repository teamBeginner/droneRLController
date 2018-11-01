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
                'train_eps':20000,
                'tau':1e-3,
                'gamma':0.99,
                'mini_batch_size':[40,200,400],
                'mini_batch_ancor':[0,100,200],
                'theta':0.15,
                'sig':0.2,
                'on_TensorBoard':False,
                'save_model_interval':1000,
                'compare_et':True,}
config_Actor = {'lr':1e-4,
                'h_dim':[400,300],}
config_Critic = {'lr':1e-3,
                 'h_dim':[400,300],}


DDPG_naive(config_train,config_Actor,config_Critic)
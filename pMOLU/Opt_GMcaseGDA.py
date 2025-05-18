# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
Run the Guangming case study.
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization 
"""
#%%
from model import GDAmodel
import objs
Model = GDAmodel(objs, LUname='LU', thC=.1, z_LU=12, rsam=4, mask=15, path='GMCase')
print(Model.LU.shape)

Model.init_train(500, lr=0.1)
Model.save_inp()
Model.init_train_load('inp', 0.1)

#%%
Model.perference([1, 1, 1, 1, 1, 1, 1])
Model.train(n_iter=10000, thStop=1, nprint=None)
Model.save()

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
simulation experiment of GDA
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization 
"""
# %%
import os
import time
import numpy as np
import objs_sim as objs
from model import GDAmodel

cmap = ["blue", "gold", "lightpink", "orangered", "springgreen"]
lus = []
res = []
Model = GDAmodel(objs, LUname='LU', thC=None, z_LU=4, rsam=1, mask=None, path='SimExp', GPU=0, cmap=cmap)
print(Model.LU.shape)
Model.Eval = None

ttt = time.time()

Model.init_train(10000, lr=0.01)
# Model.init_train_load('inp', 0.01)
for i in range(5, 15, 1):
    i *= .1
    for j in range(5, 15, 1):
        j *= .1
        Model.init_para()
        Model.perference([i, j])
        Model.train(10000, thStop=1, nprint=None)
        lus.append(Model.get_opt())
        res.append([i, j])
        # Model.perference([1/i, 1/j])
        print('%.2f/%.2f' % (i, j))
print("Runtime", time.time()-ttt)
res=[[r[0], r[1], objs.Eval(Model, lu)[0]] for (r, lu) in zip(res, lus)]
res=[[i,j,k[0],k[1]] for (i, j, k) in res]
r = objs.Eval(Model, Model.get_opt())

res=np.array(res)

reslu={(int(r[0]*10), int(r[1]*10)): lu for (r, lu) in zip(res, lus)}
# plt.plot(res[:,2], res[:,3], '.')
# plt.plot(res2[:,2], res2[:,3], '.')
saveP = f'{Model.path}/GAResult{Model.LU.shape[0]}'
os.makedirs(saveP, exist_ok=True)
np.save(f'{saveP}/GDAPF.npy', res)
np.save(f'{saveP}/GDALU.npy', reslu)

# %%

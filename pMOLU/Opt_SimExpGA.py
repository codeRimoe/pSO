# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:53:40 2020
simulation experiment of GA
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization 
"""

# -*- coding: utf-8 -*-
""" QuickStart """
#%%
import os
import numpy as np
import geatpy as ea
from model import GAmodel
import objs_sim as objs
cmap = ["blue", "gold", "lightpink", "orangered", "springgreen"]

# define problem
class MyProblem(ea.Problem):
    def __init__(self, objs, LUname='LU', thC=.1, z_LU=4, rsam=1, path='SimExp', cmap=cmap):
        self.thC = thC
        self.GM = GAmodel(objs, LUname=LUname, z_LU=z_LU, rsam=rsam, mask=None, path=path, cmap=cmap)
        self.org_cms = self.GM.to_cms()
        self.org =  self.GM.objs()
        M = len(self.GM.n_obj[0])
        name = LUname      # initial function name

        maxormins = [1] * M             # initial max/mins 1：min/-1：max
        Dim = self.org_cms.shape[0]     # initial dicision variable Dim
        varTypes = np.array([1] * Dim)  # initial varTypes 0：real 1：int
        lb = [1] * Dim     # lower bound
        ub = [z_LU] * Dim  # upper bound
        lbin = [1] * Dim   # include lower boundary
        ubin = [1] * Dim   # include upper boundary
        # instantiation
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def aimFunc(self, pop):  # objectives
        Vars = pop.Phen      # dicision var matrix
        f, c = [], []
        for v in Vars:
            LU = self.GM.to_LU(v)
            obs, cs = self.GM.objs(LU)
            f.append(obs)
            c.append(cs)
        pop.ObjV = np.array(f)  # assign objectives values => pop.ObjV


def set_chm(A, B, p):
    p = np.random.random(len(A)) > p
    B[p] = A[p]

def get_prophet(random, prophet_cms, p):
    for i in random.Chrom:
        set_chm(prophet_cms, i, p)


#%%
import time
ttt = time.time()

# runtime 17871.26741258404
"""=================================instantiation================================"""
problem = MyProblem(objs, LUname='LU', z_LU=4, rsam=1, path='SimExp')    # build object
"""==================================population================================="""
Encoding = 'RI'           # encoding method
NIND = 100                # population size
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # creat crt

population = ea.Population(Encoding, Field, NIND)  # pop 
prophetPop = ea.Population(Encoding, Field, NIND)  # pop instantiation
prophetPop.initChrom(NIND)                         # pop initialization
get_prophet(prophetPop, problem.org_cms, 0.0)
"""====================================Algorithm================================"""
myAlgorithm = ea.moea_NSGA3_templet(problem, population)
myAlgorithm.MAXGEN = 10002                                # max iter
myAlgorithm.drawing = 0                                   # no drawing

"""==========================run=========================
"""

saveP = f'{problem.GM.path}/GAResult{problem.GM.LU.shape[0]}'
os.makedirs(saveP, exist_ok=True)
NDSet = myAlgorithm.run(prophetPop)      # run algo for non dom set
NDSet = myAlgorithm.finishing(myAlgorithm.population)
myAlgorithm.population.save(saveP)
# print res
print('Time：%f s'%(myAlgorithm.passTime))
print('Eval Num：%d'%(myAlgorithm.evalsNum))
print('Non dom set size：%d'%(NDSet.sizes))
print('Num Pareto point / time：%d/s'%(int(NDSet.sizes // myAlgorithm.passTime)))
print(time.time()-ttt)

JS = {}
for x in myAlgorithm.his:
    i = x[0]
    j = x[1]
    JS[i] = j
np.save(f'{saveP}/GAPF.npy' % problem.GM.path, JS)

# %%

# -*- coding: utf-8 -*-
"""
Created on Apr 20 2025
pMOLU model for GDA: torch implementation
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization
usage:

from pSO import OptProblem

"""
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
# from torch import optim

import random
RSeed = 1486
torch.manual_seed(RSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RSeed)
np.random.seed(RSeed)
random.seed(RSeed)
torch.backends.cudnn.deterministic = True

class OptProblem(nn.Module, ABC):
    # Mask：   valid: 1
    def __init__(self, DV, dvSoft, dvMask, Ws):
        super(OptProblem, self).__init__()
        self.DV = dict(dv=[], soft=[], mask=[])
        for i in dvSoft:
            self.register_buffer(f'dvSoft-{i}', torch.tensor(dvSoft[i], dtype=torch.float32, requires_grad=False))
            self.DV['soft'] += [i]
        for i in dvMask:
            self.register_buffer(f'dvMask-{i}', torch.tensor(dvMask[i], dtype=torch.float32, requires_grad=False).unsqueeze(-1))
            self.DV['mask'] += [i]
        for i in Ws:
            self.register_buffer(f'W-{i}', torch.tensor(Ws[i], dtype=torch.float32, requires_grad=False))

        for key in DV:
            dv = DV[key]
            self.register_buffer(f'dvOrg-{key}', torch.tensor(dv, dtype=torch.float32, requires_grad=False) * self['dvMask', key])
            dv = nn.Parameter(self[f'dvOrg-{key}'].clone().detach().requires_grad_(True))
            self.register_parameter(f'dv-{key}', dv)
            self.DV['dv'] += [key]

        nobj, ncon = self.n_obj
        nobj, ncon = len(nobj), len(ncon)
        # check
        try:
            testDV = self.softMaskDV()
            testObj, testCon = self.calObj(testDV)
        except Exception as e:
            print(f"Caught an exception in building decision variable: {e}")
            raise e
        assert nobj == len(testObj), "the number of objectives should equal to the number of objective names"
        assert ncon == len(testCon), "the number of constraints should equal to the number of constraint names"
        if self.con_sup is not None:
            assert ncon == len(self.con_sup), "the number of constraints should equal to the number of their supremum"
            self.register_buffer('norm-Con-Sup', torch.tensor(self.con_sup, dtype=torch.float32, requires_grad=False))
        self.register_buffer('norm-Obj-Bias', torch.zeros(nobj, dtype=torch.float32, requires_grad=False))
        self.register_buffer('norm-Obj-Scale', torch.ones(nobj, dtype=torch.float32, requires_grad=False))
        self.register_buffer('norm-Con-Bias', torch.zeros(ncon, dtype=torch.float32, requires_grad=False))
        self.register_buffer('norm-Con-Scale', torch.ones(ncon, dtype=torch.float32, requires_grad=False))
        self.reset_parameters()
    
    def __getitem__(self, x):
        if type(x) == str:
            return getattr(self, x)
        else:
            return getattr(self, '-'.join([str(_) for _ in x]))
    

    def reset_parameters(self):
        self.zero_grad()
        for key in self.DV['dv']:
            self['dv', key].data.copy_(self['dvOrg', key])
        self.zero_grad()

    def optimize(self, lossF, optimizer, desc=None, niter=2000, resetP=True):
        if resetP:
            self.reset_parameters()
        if desc is None:
            desc = 'Loss'
        bar = tqdm(range(niter), desc=desc)
        for _ in bar:
            loss = lossF()
            bar.set_postfix(loss=f'{loss.item():.2f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def norm(self, optimizer, niter=2000, calMax=True):
        print('Evaluating normalize parameters...')
        n_objs, n_cons = self.n_obj
        oB, oS, cB, cS = [], [], [], []
        if not calMax:
            orgObj, orgCon = self()
        for i, n_obj in enumerate(n_objs):
            # cal single obj min
            lossMin = self.optimize(lambda : self()[0][i], optimizer, f'MIN Obj{i}_{n_obj}', niter)

            # cal single obj max
            if calMax:
                lossMax = self.optimize(lambda : -self()[0][i], optimizer, f'MAX Obj{i}_{n_obj}', niter)
            else:
                lossMax = -orgObj[i]
                print(f'MAX Obj{i}_{n_obj}: {-lossMax.item()}')

            lossMax = -(lossMax + lossMin)
            print('Scale =', lossMax.item())
            oB += [lossMin.item()]
            oS += [lossMax.item()]

        for i, n_con in enumerate(n_cons):
            lossMin = self.optimize(lambda : self()[1][i], optimizer, f'MIN Con{i}_{n_con}', niter)
            lossMax = self.optimize(lambda : -self()[1][i], optimizer, f'MAX Con{i}_{n_con}', niter)
            lossMax = -(lossMax + lossMin)
            print('Scale =', lossMax.item())
            cB += [lossMin.item()]
            cS += [lossMax.item()]
        self.reset_parameters()
        self['norm-Obj-Bias'].data.copy_(torch.tensor(oB, dtype=torch.float32, requires_grad=False))
        self['norm-Con-Bias'].data.copy_(torch.tensor(cB, dtype=torch.float32, requires_grad=False))
        self['norm-Obj-Scale'].data.copy_(torch.tensor(oS, dtype=torch.float32, requires_grad=False))
        self['norm-Con-Scale'].data.copy_(torch.tensor(cS, dtype=torch.float32, requires_grad=False))

    def softMaskDV(self):
        DVx = {}
        for key in self.DV['dv']:
            dv = self['dv', key]
            if key in self.DV['soft']:
                dv = F.softmax(dv * self['dvSoft', key], dim=-1)
            if key in self.DV['mask']:
                dv = dv * self['dvMask', key]
            DVx[key] = dv
        return DVx

    def forward(self):
        DVx = self.softMaskDV()
        objs, cons = self.calObj(DVx)
        objs = (objs - self['norm-Obj-Bias']) / self['norm-Obj-Scale']
        cons = (cons - self['norm-Con-Bias']) / self['norm-Con-Scale']
        return objs, cons
    
    @property
    @abstractmethod
    def n_obj(self):
        # return obj, constraint
        pass

    @property
    def con_sup(self):
        # return constraint_supremum (after normalization)
        return None

    @abstractmethod
    def calObj(self):
        # return obj, constraint
        pass

    def Eval(self):
        # return obj, constraint
        return self.calObj()


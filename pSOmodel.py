# -*- coding: utf-8 -*-
"""
Created on Apr 20 2025
pMOLU model for GDA: torch implementation
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization
usage:

from pSO import OptProblem

"""
import os
from tqdm import trange
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.sparse import coo_array

# from torch import optim

import random
RSeed = 1486
torch.manual_seed(RSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RSeed)
np.random.seed(RSeed)
random.seed(RSeed)
torch.backends.cudnn.deterministic = True

def softProb(dv, scale):
    if scale > 0:
        return F.softmax(dv * scale, dim=-1)
    else:
        dvx = F.relu(dv)
        return (dvx.T / (dvx.sum(dim=-1))).T

def getHard(soft):
    cate = torch.argmax(soft, dim=-1)
    cate = torch.unsqueeze(cate, dim=-1)
    hard = torch.zeros_like(soft)
    return hard.scatter(dim=-1, index=cate, value=1)

class OptProblem(nn.Module, ABC):
    # Mask：   valid: 1
    def __init__(self, DV, dvHard, dvSoft, dvMask, Ws, softscale=1, hardscale=10):
        super(OptProblem, self).__init__()
        self.DV = dict(dv=[], Hard=dvHard, Soft=dvSoft, Mask=[])
        self.register_buffer(f'Scale-soft', torch.tensor(softscale, dtype=torch.float32, requires_grad=False))
        self.register_buffer(f'Scale-hard', torch.tensor(hardscale, dtype=torch.float32, requires_grad=False))
        if dvMask is not None:
            for i in dvMask:
                self.register_buffer(f'dvMask-{i}', torch.tensor(dvMask[i], dtype=torch.float32, requires_grad=False).unsqueeze(-1))
                self.DV['Mask'] += [i]
        self.register_Ws(Ws)

        for key in DV:
            dv = DV[key]
            org = torch.tensor(dv, dtype=torch.float32, requires_grad=False)
            if key in self.DV['Mask']:
                org = org * self['dvMask', key]
            self.register_buffer(f'dvOrg-{key}', org)
            dv = nn.Parameter(org.clone().detach().requires_grad_(True))
            self.register_parameter(f'dv-{key}', dv)
            self.DV['dv'] += [key]

        nobj, ncon = self.n_obj
        nobj, ncon = len(nobj), len(ncon)
        
        self.slack()
        self.slackon = True
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
    
    def slack(self):
        return 

    def register_slack(self, name, shape=None, like=None):
        slack = None
        if shape is not None:
            slack = torch.zeros(shape, dtype=torch.float32, requires_grad=True)
        else:
            if type(like) is str:
                like = self[like]
            slack = torch.zeros_like(like, dtype=torch.float32, requires_grad=True)
        if slack is not None:
            nn.init.normal(slack, std=0.1)
            self.register_parameter(f'slack-{name}', nn.Parameter(slack))
    
    def get_slack(self, name):
        if self.slackon:
            return self[f'slack-{name}'] * self[f'slack-{name}']
        else:
            return self[f'slack-{name}'] * 0

    def register_Ws(self, Ws, prefix='W'):
        if type(Ws) is dict:
            for w in Ws:
                self.register_Ws(Ws[w], f'{prefix}-{w}')
        else:
            if type(Ws) is coo_array:
                Wsi = torch.sparse_coo_tensor(
                    indices=np.array([Ws.row, Ws.col]),
                    values=Ws.data.astype(np.float32),
                    size=Ws.shape,
                    dtype=torch.float32, requires_grad=False
                )
            else:
                Wsi = torch.tensor(Ws, dtype=torch.float32, requires_grad=False)
            self.register_buffer(prefix, Wsi)

    def reset_parameters(self):
        self.zero_grad()
        for key in self.DV['dv']:
            self['dv', key].data.copy_(self['dvOrg', key])
        self.zero_grad()

    def optimize(self, lossF, optimizer, desc='Loss', niter=2000, resetP=True, verbose=True):
        if resetP:
            self.reset_parameters()
        bar = trange(niter, desc=desc, disable=(not bool(desc)))
        for _ in bar:
            loss = lossF()
            bar.set_postfix(loss=f'{loss.item():.2f}')
            optimizer.zero_grad()
            loss.backward()
            for name, param in self.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"Gradient of {name} contains NaN!")
                    return loss
            # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1., norm_type=2)
            optimizer.step()
        return loss

    def normalize(self, loss, optimizer, normMax=True, desc='Loss', niter=2000):
            # cal single obj max
            self.reset_parameters()
            lossOrg = loss().item()
            aX = abs(lossOrg)
            if normMax:
                lossMax = self.optimize(lambda : -loss() / aX, optimizer, f'MAX -{desc}', niter)
                scale = -lossMax.item() * aX
            else:
                scale = lossOrg

            # cal single obj min
            lossMin = self.optimize(lambda : loss() / aX, optimizer, f'MIN {desc}', niter)
            bias = lossMin.item() * aX

            scale -= bias
            print(f'Bias={bias},  Scale={scale}')
            return bias, scale
        
    def norm(self, optimizer, niter=2000, normEx=[], normMax=False):
        print('Evaluating normalize parameters...')
        n_objs, n_cons = self.n_obj
        oB, oS, cB, cS = [], [], [], []
        if type(normMax) is bool:
            if normMax:
                normMax = n_objs + n_cons
            else:
                normMax = []
        if type(normMax[0]) is bool:
            normMax[0] = n_objs if normMax[0] else []
        if type(normMax[1]) is bool:
            normMax[1] = n_cons if normMax[1] else []
            
        if (type(normMax[0]) is list) and (type(normMax[1]) is list):
            normMax = sum(normMax, [])

        for i, n_obj in enumerate(n_objs):
            if n_obj in normEx:
                bias, scale = 0, 1
            else:
                bias, scale = self.normalize(lambda : self(norm=False)[0][i], optimizer, n_obj in normMax, f'Obj{i}_{n_obj}', niter)
            oB += [bias]
            oS += [scale]

        for i, n_con in enumerate(n_cons):
            if n_con in normEx:
                bias, scale = 0, 1
            else:
                bias, scale = self.normalize(lambda : self(norm=False)[1][i], optimizer, n_con in normMax, f'Con{i}_{n_con}', niter)
            cB += [bias]
            cS += [scale]
            
        self.reset_parameters()
        self['norm-Obj-Bias'].data.copy_(torch.tensor(oB, dtype=torch.float32, requires_grad=False))
        self['norm-Con-Bias'].data.copy_(torch.tensor(cB, dtype=torch.float32, requires_grad=False))
        self['norm-Obj-Scale'].data.copy_(torch.tensor(oS, dtype=torch.float32, requires_grad=False))
        self['norm-Con-Scale'].data.copy_(torch.tensor(cS, dtype=torch.float32, requires_grad=False))

    def softMaskDV(self):
        DVx = {}
        for key in self.DV['dv']:
            dv = self['dv', key]
            if key in self.DV['Soft']:
                dv = softProb(dv, self['Scale-soft'])
            if key in self.DV['Hard']:
                dv = softProb(dv, self['Scale-hard'])
            if key in self.DV['Mask']:
                dv = dv * self['dvMask', key]
            DVx[key] = dv
        return DVx

    def hardMaskDV(self):
        DVsoft = self.softMaskDV()
        DVx = {}
        for key in DVsoft:
            dv = DVsoft[key]
            if key in self.DV['Hard']:
                dv = getHard(dv)
            DVx[key] = dv
        return DVx

    def forward(self, norm=True):
        DVx = self.softMaskDV()
        return self.Obj(DVx, norm)
    
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
    def calObj(self, DV):
        # return obj, constraint
        pass

    def Obj(self, DV, norm):
        objs, cons = self.calObj(DV)
        objs, cons = torch.stack(objs), torch.stack(cons)
        if norm:
            objs = (objs - self['norm-Obj-Bias']) / self['norm-Obj-Scale']
            cons = (cons - self['norm-Con-Bias']) / self['norm-Con-Scale']
        return objs, cons

    def Eval(self, DV=None, norm=True):
        # return obj, constraint
        self.slackon = False
        if DV is None:
            DV = self.hardMaskDV()
        obj = self.Obj(DV, norm)
        self.slackon = True
        return obj
    
    def to_npy(self, DV):
        return {dv: DV[dv].cpu().detach().numpy() for dv in DV}


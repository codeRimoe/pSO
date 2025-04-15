# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
This code include implemented objective functions for Guangming case study in the artile:
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization 
"""
import numpy as np
import tensorflow as tf
n_obj = [["Gdp", "Eco", "Res", "Emp", "Aec", "E2r", "Cpt"], ["Cov"]]

# DMU function
def eta(p):
    return 1 - p * p / 3

# mean pooling/convolution
def meanP(inputMap, poolSize=100, poolStride=50):
    """
    source: https://blog.csdn.net/com_fang_bean/article/details/103483916
    INPUTS:
            inputMap - input array of the pooling layer
            poolSize - X-size(equivalent to Y-size) of receptive field
            poolStride - the stride size between successive pooling squares
    
    OUTPUTS:
            outputMap - output array of the pooling layer
    """
    # inputMap sizes
    in_row, in_col, out_band = inputMap.shape
    
    # outputMap sizes
    out_row, out_col = int(in_row / poolStride - 1),int(in_col / poolStride - 1)
    outputMap = np.zeros((out_row, out_col, out_band))
    
    # max pooling
    for r_idx in range(0, out_row):
        for c_idx in range(0, out_col):
            startX = c_idx * poolStride
            startY = r_idx * poolStride
            poolField = inputMap[startY:startY + poolSize, startX:startX + poolSize]
            poolOut = np.mean(poolField, axis=(0,1))
            outputMap[r_idx, c_idx] = poolOut
    
    return outputMap

def pool_(LU, r, s=1):
    r, s = int(r), int(s)
    if len(LU.shape) == 2:
        LU = np.stack([LU], axis=-1)
    PM = meanP(LU, r, s)
    return np.reshape(PM, (-1, LU.shape[-1]))

# Tensorflow implement
def pool(LU, r, s=1):
    if LU.shape.ndims == 2:
        LU = tf.stack([LU], axis=-1)
    LU = tf.stack([LU], axis=0)
    PM = tf.nn.avg_pool(LU, [1, r, r, 1], [1, s, s, 1], padding='VALID')
    return tf.reshape(PM, (-1, LU.shape[-1]))


# Objective: conversion (constraint)
def coversion_(A, B, CCM, weights=None):
    if weights is not None:
        A += np.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = np.transpose(np.reshape(A, (-1, z1)))
    B = np.reshape(B, (-1, z2))
    ABm = np.matmul(A, B) * CCM
    ABm = np.sum(ABm, axis=-1)
    return np.sum(ABm)

# Objective: conversion (constraint) Tensorflow implement
def coversion(A, B, CCM, weights=None):
    if weights is not None:
        A += tf.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = tf.transpose(tf.reshape(A, (-1, z1)))
    B = tf.reshape(B, (-1, z2))
    ABm = tf.matmul(A, B) * CCM
    ABm = tf.reduce_sum(ABm, axis=-1)
    return tf.reduce_sum(ABm)

# for gradient desent
def Objs(self):
    self.W['r'] = 100
    self.W['r2'] = 100
    p = tf.reduce_sum(self.LU, axis=(0, 1))     # calculate proportion: 1*C
    p /= tf.reduce_sum(p)                       # normalization

    w = eta(p)

    res = tf.reduce_sum(self.LU * self.W['UUM'][0], axis=-1)
    emp = tf.reduce_sum(self.LU * self.W['UUM'][1], axis=-1)
    gdp = tf.reduce_sum(self.LU * self.W['UUM'][2] * w, axis=-1) * (1 - self.W['road'])
    eco = tf.reduce_sum(self.LU * self.W['UUM'][3] * w, axis=-1) * (1 - self.W['water'])
    liv = tf.reduce_sum(self.LU * self.W['UUM'][4] * w, axis=-1) * (1 - self.W['road'])

    resP = pool(res, self.W['r'], self.W['r'] / 2)
    empP = pool(emp, self.W['r'], self.W['r'] / 2)
    livP = pool(eco + liv, self.W['r'], self.W['r'] / 2)
    PM = pool(self.LU, self.W['r2'], self.W['r2'] / 2)

    Res = -tf.reduce_sum(res)
    Emp = -tf.reduce_sum(emp)
    Gdp = -tf.reduce_sum(gdp) # utility_sig(p_m, self.W['UUM'][2])
    Eco = -tf.reduce_sum(eco) # utility_sig(p_m, self.W['UUM'][3])
    Aec = -tf.reduce_sum(livP / (resP + 1))
    E2r = tf.reduce_sum(tf.abs(empP - resP))

    Cpt = -tf.reduce_sum(tf.matmul(PM, self.W['CM']) * PM)
    Cov = coversion(self.LU * self.W['Mask'], self.W['LU_'] * self.W['Mask'], 1 - self.W['CCM'], None)
                    
    objs = [[Gdp, Eco, Res, Emp, Aec, E2r, Cpt], [Cov]]
    return n_obj, objs

def Eval(self, LU):
    self.W_['r'] = 100
    self.W_['r2'] = 100
    p = np.sum(LU, axis=(0, 1))          # calculate proportion: 1*C
    p /= np.sum(p)                       # normalization

    w = eta(p)

    res = np.sum(LU * self.W_['UUM'][0], axis=-1)
    emp = np.sum(LU * self.W_['UUM'][1], axis=-1)
    gdp = np.sum(LU * self.W_['UUM'][2] * w, axis=-1) * (1 - self.W_['road'])
    eco = np.sum(LU * self.W_['UUM'][3] * w, axis=-1) * (1 - self.W_['water'])
    liv = np.sum(LU * self.W_['UUM'][4] * w, axis=-1) * (1 - self.W_['road'])

    resP = pool_(res, self.W_['r'], self.W_['r'] / 2)
    empP = pool_(emp, self.W_['r'], self.W_['r'] / 2)
    livP = pool_(eco + liv, self.W_['r'], self.W_['r'] / 2)
    PM = pool_(LU, self.W_['r2'], self.W_['r2'] / 2)

    Res = -np.sum(res)
    Emp = -np.sum(emp)
    Gdp = -np.sum(gdp) # utility_sig(p_m, self.W_['UUM'][2])
    Eco = -np.sum(eco) # utility_sig(p_m, self.W_['UUM'][3])
    Aec = -np.sum(livP / (resP + 1))
    E2r = np.sum(np.abs(empP - resP))

    Cpt = -np.sum(np.matmul(PM, self.W_['CM']) * PM)
    Cov = coversion_(LU * self.W_['Mask'], self.M.LU * self.W_['Mask'], 1 - self.W_['CCM'], None)
    objs = [[Gdp, Eco, Res, Emp, Aec, E2r, Cpt], [Cov]]
    return objs

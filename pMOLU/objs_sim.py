# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
This code include implemented objective functions for simulation data experiment in the artile:
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization 
"""
import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ModuleNotFoundError:
    import tensorflow as tf

n_obj = [["Gdp", "Eco"], ["Cov"]]

def eta(p):
    return 1 - p * p / 3

'''Objective: conversion (constraint)'''
def coversion(A, B, CCM, weights=None):
    if weights is not None:
        A += tf.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = tf.transpose(tf.reshape(A, (-1, z1)))
    B = tf.reshape(B, (-1, z2))
    ABm = tf.matmul(A, B) * CCM
    ABm = tf.reduce_sum(ABm, axis=-1)
    return tf.reduce_sum(ABm)

'''Objective: conversion (constraint) Tensorflow implement'''
def coversion_(A, B, CCM, weights=None):
    if weights is not None:
        A += np.stack([weights], axis=-1)
    z1, z2 = CCM.shape
    A = np.transpose(np.reshape(A, (-1, z1)))
    B = np.reshape(B, (-1, z2))
    ABm = np.matmul(A, B) * CCM
    ABm = np.sum(ABm, axis=-1)
    return np.sum(ABm)

# for gradient desent
def Objs(self):
    p = tf.reduce_sum(self.LU, axis=(0, 1))     # calculate proportion: 1*C
    p /= tf.reduce_sum(p)                       # normalization

    w = eta(p)

    gdp = tf.reduce_sum(self.LU * self.W['UUM'][0] * w, axis=-1) * (1 - self.W['gdp'])
    eco = tf.reduce_sum(self.LU * self.W['UUM'][1], axis=-1) * (1 - self.W['eco'])

    Gdp = -tf.reduce_sum(gdp)
    Eco = -tf.reduce_sum(eco)

    Cov = coversion(self.LU * self.W['Mask'], self.W['LU_'] * self.W['Mask'], 1 - self.W['CCM'], None)
    objs = [[Gdp, Eco], [Cov]]
    return n_obj, objs

def Eval(self, LU):
    p = np.sum(LU, axis=(0, 1))          # calculate proportion: 1*C
    p /= np.sum(p)                       # normalization

    w = eta(p)

    gdp = np.sum(LU * self.W_['UUM'][0] * w, axis=-1)
    eco = np.sum(LU * self.W_['UUM'][1], axis=-1) * (1 - self.W_['eco'])

    Gdp = -np.sum(gdp)
    Eco = -np.sum(eco)

    Cov = coversion_(LU * self.W_['Mask'], self.W_['LU_'] * self.W_['Mask'], 1 - self.W_['CCM'], None)
    objs = [[Gdp, Eco], [Cov]]
    return objs


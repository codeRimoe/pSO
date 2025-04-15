# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
MOLU model for GA and pMOLU model for GDA
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization
usage:
GAmodel

"""
import os
import loader
import numpy as np
import pandas as pd
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ModuleNotFoundError:
    import tensorflow as tf

tf.set_random_seed(1486)
class GAmodel:
    def __init__(self, Obj, z_LU, LUname='LU', rsam=1, mask=None, init_LU=None, path='.', cmap=None):
        self.n_obj = Obj.n_obj
        # self.n_obj = [["Res", "Emp", "Gdp", "Eco", "Aec", "E2r", "Cpt"], ["Cov"]]

        self.path = path
        self.M = loader.meta(z_LU, LUname=LUname, rsam=rsam, mask=mask, path=path)
        self.Unmask = np.logical_not(self.M.Mask + self.M.Restrit)
        self.Amask = self.M.Mask + self.M.Restrit
        self.W_ = {W: self.M.W[W] / self.M.W[W].max() for W in self.M.W}
        self.W_['LU_'] = self.M.prob_LU(self.M.LU_)
        self.W_['Mask'] = np.stack([1 - self.M.Mask - self.M.Restrit], axis=-1)
        self.LU = np.copy(self.M.LU_)
        self.history = pd.DataFrame()
        self.Objs = lambda x: Obj.Eval(self, x)
        self.cmap=cmap
        self.stop_iter = 0


    def objs(self, LU=None):
        if LU is None:
            LU = self.LU
        if len(LU.shape) == 2:
            LU = self.M.prob_LU(LU)
        return self.Objs(LU)

    def to_cms(self, LU=None):
        if LU is None:
            LU = self.LU
        return LU[self.Unmask]

    def to_LU(self, cms):
        self.LU[self.Unmask] = cms
        return self.LU

    def save(self):
        self.saver.save(self.sess,"%s/model" % self.path)
        self.history.to_csv("%s/train.csv" % self.path)
        self.M.save_result(self.M.get_LU(self.LU_opt))

    def plot_LU(self, LU, save_name=None, save_path='.'):
        if LU is None:
            LU = self.LU_opt
        if len(LU.shape) == 2:
            LU = self.M.prob_LU(LU)
        self.M.plot_LU(LU, save_name=save_name, path=save_path, cmap=self.cmap)



class GDAmodel:
    def __init__(self, Obj, z_LU, LUname='LU', init_LU=None, thC=.08, rsam=1,
                 mask=None, GPU=1, path='.', cmap=None):
        self.thC = thC
        self.cmap = cmap
        self.Eval = lambda x: Obj.Eval(self, x)
        # self.build_objs = xxx
        # self.build_loss = xxx
        self.path = path
        self.M = loader.meta(z_LU, LUname=LUname, rsam=rsam, mask=mask, path=path)

        self.W_ = {W: self.M.W[W] / self.M.W[W].max() for W in self.M.W}
        self.W = {W: tf.constant(self.W_[W], dtype=tf.float32) for W in self.W_}
        self.W_['LU_'] = self.M.LU
        self.W['LU_'] = tf.constant(self.W_['LU_'], dtype=tf.float32)
        self.W_['Mask'] = np.stack([1 - self.M.Mask - self.M.Restrit], axis=-1)
        self.W['Mask'] = tf.constant(self.W_['Mask'], dtype=tf.float32)

        if init_LU is None:
            init_LU = self.M.LU
        self.LU = tf.Variable(init_LU, dtype=tf.float32)
        self.LU = tf.nn.softmax(self.LU * 10)
        self.LU *= self.W['Mask']
        self.build_objs(Obj.Objs)

        n_obs1 = len(self.n_obj[0])
        n_obs2 = len(self.n_obj[1])
        self.Norm0 = tf.placeholder(tf.float32, [n_obs1])
        self.Norm1 = tf.placeholder(tf.float32, [n_obs1])
        self.Norm2 = tf.placeholder(tf.float32, [n_obs2])

        self.build_loss()

        self.inp = {self.Norm1: np.ones(n_obs1), self.Norm2: np.ones(n_obs2)}
        self.history = pd.DataFrame()

        config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                allow_soft_placement=True,
                                log_device_placement=True,
                                device_count = {'CPU' : 1, 'GPU' : GPU})
        if GPU != 0:
            config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

    def build_objs(self, Obj):
        self.n_obj, self.objs = Obj(self)

    def build_loss(self):
        self.loss1 = tf.stack(self.objs[0], axis=0)
        self.loss1_ = tf.reduce_max(tf.abs(self.loss1 - self.Norm1) / self.Norm0)
        self.loss2 = tf.stack(self.objs[1], axis=0)
        self.loss2_ = self.loss2 / self.Norm2
        self.loss = self.loss1_ + self.loss2_
        # tf.concat([Model.loss1,Model.loss2_], axis=0)

    def set_loss(self, loss):
        self.train_op = self.optimizer.minimize(loss)

    def init_para(self):
        self.sess.run(tf.global_variables_initializer())

    def init_graph(self, loss=None):
        if loss is not None:
            self.set_loss(loss)
        self.init_para()
        self.inp = {self.Norm1: np.ones_like(self.inp[self.Norm1]),
                    self.Norm2: np.ones_like(self.inp[self.Norm2])}


    def init_train(self, n_iter=50, lr=0.1, optimizer=tf.train.AdamOptimizer):
        print('initial...')
        self.optimizer = optimizer(lr)
        self.init_graph(self.loss)
        self.Org = self.sess.run(self.objs)
        Norm1 = []
        for n, l, o in zip(self.n_obj[0], self.objs[0], self.Org[0]):
            self.init_graph(l)
            for i in range(n_iter):
                self.sess.run(self.train_op, feed_dict=self.inp)
            Norm1.append(self.sess.run(l))
            print(n, Norm1[-1], o)

        self.init_graph(self.loss)
        self.inp[self.Norm0] = np.abs(np.array(self.Org[0]) - np.array(Norm1))
        self.inp[self.Norm1] = np.array(Norm1)
        self.inp[self.Norm2] = np.array(self.Org[1])


    def init_train_load(self, inp_file, lr=0.1, optimizer=tf.train.AdamOptimizer):
        self.optimizer = optimizer(lr)
        self.init_graph(self.loss)
        self.load_inp(inp_file)
        print([self.inp[i] for i in self.inp])

    def perference(self, scale=None):
        if scale is not None:
            self.inp[self.Norm1] *= scale

    def get_res(self):
        obs = self.sess.run((self.loss1, self.loss2), feed_dict=self.inp)
        return obs

    def eval_objs(self):
        if self.Eval == None:
            return self.get_res()
        else:
            return self.Eval(self.LU_opt)

    def get_opt(self):
        self.LU_opt = self.M.prob_LU(self.M.get_LU(self.sess.run(self.LU)))
        return self.LU_opt

    def log(self, _iter=None, history=False, plot=True):
        if _iter is None:
            try:
                _iter = self.stop_iter
            except AttributeError:
                pass
        self.LU_opt = self.get_opt()
        obs = self.eval_objs()
        obsOrg = [np.array(obs[0]) / np.array(self.Org[0]),
                  np.array(obs[1]) / np.array(self.Org[1])]
        res = {"epoch": _iter,
               "loss": self.sess.run(self.loss, feed_dict=self.inp),
               "loss1": self.sess.run(self.loss1, feed_dict=self.inp),
               "loss2": self.sess.run(self.loss2, feed_dict=self.inp),
               "obs1": np.array(obs[0]),
               "obs2": np.array(obs[1]),
               "obs1R": obsOrg[0],
               "obs2R": obsOrg[1]}
        if history:
            self.history = pd.concat([self.history, pd.DataFrame.from_records([res])], ignore_index=True)
        if plot:
            self.plot_opt(str(res['epoch']), "%s/trainLUs" % self.path)
            print("==================")
            print('epoch: %s, loss: %s' % (res['epoch'], res['loss']))
            for i, j, k in zip(self.n_obj[0], obs[0], obsOrg[0]):
                res[i] = j
                res[i + 'norm'] = k
                print("%s: %s, %s" % (i, j, k))
            for i, j, k in zip(self.n_obj[1], obs[1], obsOrg[1]):
                res[i] = j
                res[i + 'norm'] = k
                print("%s: %s, %s" % (i, j, k))
        return res

    def train(self, n_iter, thStop=None, nprint=100):
        NL = 1
        self.set_loss(self.loss1_)
        for _iter in range(n_iter):
            if _iter == 0:
                self.log(_iter, history=True, plot=(nprint!=None))
            elif (nprint is not None) and (_iter % nprint == 0):
                self.log(_iter, history=True)
            if (thStop is not None) and (self.thC is not None):
                obs = self.sess.run(self.loss2, feed_dict=self.inp)
                l2 = 1 - obs / np.array(self.Org[1])
                # print(l2, self.thC)
                # self.set_loss(l2 * self.loss1_)
                if (l2 >= self.thC) and (NL == 1):
                    if thStop == 1:
                        self.log(_iter, history=True)
                        self.stop_iter = _iter
                        return
                    self.set_loss(1 - self.loss2_)
                    NL = 2
                elif (l2 < self.thC) and (NL == 2):
                    if thStop == 2:
                        self.log(_iter, history=True)
                        self.stop_iter = _iter
                        return
                    self.set_loss(self.loss1_)
                    NL = 1
            self.sess.run(self.train_op, feed_dict=self.inp)
        self.stop_iter = _iter

    def save(self):
        self.saver.save(self.sess,"%s/model" % self.path)
        self.history.to_csv("%s/train.csv" % self.path)
        self.M.save_result(self.M.get_LU(self.LU_opt))

    def save_inp(self, file_name='inp'):
        path = "%s/%s" % (self.path, file_name)
        os.makedirs(path, exist_ok=True)
        np.save("%s/Norm0" % path, self.inp[self.Norm0])
        np.save("%s/Norm1" % path, self.inp[self.Norm1])
        np.save("%s/Norm2" % path, self.inp[self.Norm2])
        np.save("%s/Org0" % path, self.Org[0])
        np.save("%s/Org1" % path, self.Org[1])

    def load_inp(self, file_name='inp'):
        path = "%s/%s" % (self.path, file_name)
        self.inp = {self.Norm0: np.load("%s/Norm0.npy" % path),
                    self.Norm1: np.load("%s/Norm1.npy" % path),
                    self.Norm2: np.load("%s/Norm2.npy" % path)}
        self.Org = [np.load("%s/Org0.npy" % path), np.load("%s/Org1.npy" % path)]

    def plot_opt(self, save_name=None, save_path='.'):
        self.M.plot_LU(self.LU_opt, save_name=save_name, path=save_path, cmap=self.cmap)


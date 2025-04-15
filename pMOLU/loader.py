# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:12:21 2019
A script to load Landuse(LU) Data.
A probabilistic framework based on the gradient descent algorithm for multi-objective land use optimization

usage:
import loader
M = loader.meta(z_LU, LUname=LUname, rsam=rsam, mask=mask, path=path)
    # z_LU: the number of the LU categories
    # LUname: the filename of the LU folder
    # rsam: downsampling scale
    # mask: the mask DN value (restrict area)
    # path: the data path

method:
prob_LU: encode the LU as probabilty vector
load_LU: load LU file (raster)
load_W: load the weight file
save_result: save the optimal LUfile
plot_LU: plot the LU map
get_LU: return the LU array

"""
import os
try:
    import gdal
except ModuleNotFoundError:
    from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_default = ListedColormap(["black", "deeppink", "lightpink", "darkviolet",
                               "royalblue", "gold", "lightseagreen", "orange",
                               "orangered", "yellowgreen", "forestgreen",
                               "springgreen", "lightgray", "white", "white",
                               "white"])
lgend = ['Restrict', 'R1', 'R2', 'RC', 'C', 'CBD', 'IH', 'I1', 'I2', 'A', 'E', 'G', 'U']

def read_tif(name, resample=1, path='.'):
    name = "%s/%s.tif" % (path, name)
    ds = gdal.Open(name)
    para = {'proj': ds.GetProjection(), 'tran': ds.GetGeoTransform(),
            'x': ds.RasterXSize, 'y': ds.RasterYSize}
    if resample == 1:
        img = ds.ReadAsArray(0, 0, para['x'], para['y'])
    else:
        t = list(para['tran'])
        t[1] *= resample
        t[5] *= resample
        para['tran'] = t
        para['x'] = int(para['x'] / resample)
        para['y'] = int(para['y'] / resample)
        img = ds.ReadAsArray(0, 0, buf_xsize=para['x'], buf_ysize=para['y'])
    return img, para

def read_tif_asXY(name, xy=None, path='.'):
    name = "%s/%s.tif" % (path, name)
    ds = gdal.Open(name)
    para = {'proj': ds.GetProjection(), 'tran': ds.GetGeoTransform(),
            'x': ds.RasterXSize, 'y': ds.RasterYSize}
    if xy is None:
        img = ds.ReadAsArray(0, 0, para['x'], para['y'])
    else:
        t = list(para['tran'])
        t[1] *= para['x'] * 1. / xy[0]
        t[5] *= para['y'] * 1. / xy[1]
        para['tran'] = t
        para['x'], para['y'] = xy
        img = ds.ReadAsArray(0, 0, buf_xsize=para['x'], buf_ysize=para['y'])
    return img, para

def save_tif(img, para, name, path='.'):
    name = "%s/%s.tif" % (path, name)
    ds = gdal.GetDriverByName('GTiff').Create(name, para['x'], para['y'], 1,
                                              gdal.GDT_Byte)
    ds.SetProjection(para['proj'])
    ds.SetGeoTransform(para['tran'])
    ds.GetRasterBand(1).WriteArray(img)
    ds = None
    print("Save: %s" % name)


class meta:
    def __init__(self, z_LU, LUname='LU', rsam=1, mask=15, path='.'):
        self.path = path
        self.rsam = rsam
        self.z_LU = z_LU
        if mask is None:
            mask = z_LU + 1
        self.mask = mask
        self.load_LU(LUname)
        self.load_W()

    def prob_LU(self, LU):
        LU1h = np.array([LU == (c + 1) for c in range(self.z_LU)], dtype=np.float32)
        return np.transpose(LU1h, [1, 2, 0])

    def load_LU(self, LUname='LU', restrict=None, mask=None):
        print("Loading landuse: %s.tif" % LUname)
        self.LU, self.para = read_tif(LUname, resample=self.rsam, path=self.path)
        self.Restrit = self.LU == 0
        self.Mask = self.LU == self.mask
        self.LU_ = self.LU
        self.LU = self.prob_LU(self.LU)

    def load_W(self):
        self.W = {}
        f_W = "%s/Ws" % self.path
        for _f in os.listdir(f_W):
            _n, _t = _f[:-4], _f[-4:]
            print("Loading weights: %s" % _f)
            if _t == '.csv':
                self.W[_n] = np.loadtxt("%s/%s.csv" % (f_W, _n), delimiter=',')
            elif (_t == '.tif'):
                _xy = self.LU.shape[1], self.LU.shape[0]
                self.W[_n], _ = read_tif_asXY(_n, xy=_xy, path=f_W)

    def save_result(self, img, name="LU_opt"):
        print("Saving %s" % name)
        save_tif(img, self.para, name, path=self.path)

    def plot_LU(self, LU=None, save_name=None, path='.', cmap=None):
        if LU is None:
            LU = self.LU
        LU = self.get_LU(LU)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        if type(cmap) is not type(cmap_default):
            try:
                cmap = ListedColormap(cmap)
            except:
                cmap = cmap_default
        plt.imshow(LU, cmap=cmap, vmin=0, vmax=self.mask, interpolation='nearest')
        if save_name is not None:
            os.makedirs(path, exist_ok=True)
            plt.savefig("%s/%s" % (path, save_name))
        plt.show()

    def get_LU(self, LU):
        LU = np.argmax(LU, axis=-1) + 1
        LU = np.array(LU, dtype=np.int8)
        LU[self.Restrit] = 0
        LU[self.Mask] = self.mask
        return LU


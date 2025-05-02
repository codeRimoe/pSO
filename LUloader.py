# -*- coding: utf-8 -*-
"""
This is an LUloader to load land use data, including vector and raster

@author: Yue
"""
#%%
import os
try:
    import gdal
except ModuleNotFoundError:
    from osgeo import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


# lgend = ['Restrict', 'R1', 'R2', 'RC', 'C', 'CBD', 'IH',
#          'I1', 'I2', 'A', 'E', 'G', 'U']
# color = ['black', 'deeppink', 'lightpink', 'darkviolet',
#          'royalblue', 'gold', 'lightseagreen', 'orange',
#          'orangered', 'yellowgreen', 'forestgreen',
#          'springgreen', 'lightgray', 'white']

def vplot_cate(gdf, cates, vmin, vmax, ax, cmap, lmap, legend, legend_kwds):
    pmarks = []
    for cate in range(vmin, vmax + 1):
        color = cmap[cate]
        label = lmap[cate]
        lus = gdf[cates == cate]
        if len(lus) > 0:
            lus.plot(ax=ax, color=color)
        elif cate == 0:
            continue
        pmarks.append(Patch(facecolor=color, label=label))
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        # TODO: ncol here need to be gneralized
        ax.legend(handles=[*handles, *pmarks], ncol=len(pmarks), **legend_kwds)

def rplot_cate(im, vmin, vmax, ax, cmap, lmap, legend, legend_kwds):
    imx = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax + 1, interpolation='none')
    # TODO:legend
    if legend:
        cbar = plt.colorbar(imx, ax=ax, cmap=cmap, **legend_kwds)
        cbar.set_ticks(ticks=[v + .5 for v in range(vmin, vmax + 1)], labels=lmap)

def read_tif(filename, resample=1):
    ds = gdal.Open(filename)
    para = {'proj': ds.GetProjection(), 'tran': ds.GetGeoTransform(),
            'x': ds.RasterXSize, 'y': ds.RasterYSize}
    if resample is None:
        img = ds.ReadAsArray(0, 0, para['x'], para['y'])
    else:
        try:
            x, y = resample
            rx, ry = para['x'] * 1. / x, para['y'] * 1. / y
        except:
            rx, ry = resample, resample
            x, y = int(para['x'] / rx), int(para['y'] / ry)
        t = list(para['tran'])
        t[1], t[5] = t[1] * rx, t[5] * ry
        para['tran'] = t
        para['x'], para['y'] = x, y
        img = ds.ReadAsArray(0, 0, buf_xsize=para['x'], buf_ysize=para['y'])
    para['img'] = img
    return para

def write_tif(filename, img, para, dtype=gdal.GDT_Byte):
    ds = gdal.GetDriverByName('GTiff').Create(filename, para['x'], para['y'], 1, dtype)
    ds.SetProjection(para['proj'])
    ds.SetGeoTransform(para['tran'])
    ds.GetRasterBand(1).WriteArray(img)
    ds = None
    print(f'Saved: {filename}')

def read_raster(filename, driver='TIF', resample=1, mask=None):
    if driver == 'TIF':
        raster = read_tif(filename, resample)
    elif driver == 'NPY':
        raster = {'img': np.load(filename)}
        # TODO: resample
    elif driver == 'MAT':
        # TODO: add mat loader
        print(f'MAT will be supported in the future.')
        return
    else:
        print(f'No driver {driver} is founded.')
        return
    raster['mask'] = mask
    return raster

def write_raster(filename, img, para, driver='TIF'):
    if driver == 'TIF':
        write_tif(filename, img, para)

class LUloader:
    '''
    path
    z_LU: 0 must be Restrict or just skip and begin from 1
    isProb
    VorR
    raw
    var
    if noR
    '''
    def __init__(self, LUs, asProb, LUType, noR=False, path='.', cmap=None, **kwargs):
        self.path = path
        if noR:
           LUs = ['-'] + LUs
           cmap = ['grey'] + cmap
        self.z_LU = len(LUs) - 1
        self.legend = LUs
        self.cmap = cmap
        self.isProb = asProb
        self.VorR = LUType
        self.noR = noR
        self.load(**kwargs)
        # try:
        #     self.load(**kwargs)
        # except:
        #     print('Load Error.')
        #     pass
    
    def load(self, LUname, driver, **kwargs):
        if self.VorR == 'R':
            self.load_raster(LUname, driver=driver, **kwargs)
        elif self.VorR == 'V':
            self.load_vector(LUname, driver=driver, **kwargs)


    def load_vector(self, LUname, LUcol, LID=None, driver='shapefile', encoding='utf-8', **kwargs):
        # kwargs: resample(int/2d-array), mask(int)
        #         LUcol(str), encoding(str)
        print(f'Loading vector landuse: {LUname}, LUCol: {LUcol}, Driver: {driver}/{encoding}')
        LUname = os.path.join(self.path, LUname)
        # read the raw data
        if driver == 'feather':
            vector = gpd.read_feather(f'{LUname}.feather')
        else:
            vector = gpd.read_file(LUname, driver=driver, encoding=encoding)
        if LID is not None:
            vector = vector.set_index(LID)
        # make the variable data
        _LU = np.array(vector[LUcol], dtype=np.int8)
        if self.noR:
            _LU += 1
        _R = _LU == 0
        if self.isProb:
            _LU = pd.DataFrame(self.prob_LU(_LU), index=vector.index, columns=pd.MultiIndex.from_product([["LU"], [i + 1 for i in range(self.z_LU)]]))
            _LU['R'] = _R
            gdf = gpd.GeoDataFrame(_LU, geometry=vector.geometry, crs=vector.crs)
        else:
            gdf = gpd.GeoDataFrame({'LU': _LU, 'R': _R}, geometry=vector.geometry, crs=vector.crs)
        self.raw = vector
        self.var = gdf
        print(f"Loaded vector: {LUname}[{driver}/{encoding}: {len(self.raw)}]")

    def load_raster(self, LUname, mask=None, resample=1, driver='TIF', **kwargs):
        # kwargs: resample(int/2d-array), mask(int)
        #         LUcol(str), encoding(str)
        print(f'Loading raster landuse: {LUname}, MaskDN: {mask}, Driver: {driver}/{resample}x')
        LUname = os.path.join(self.path, LUname)
        raster = read_raster(LUname, driver=driver, resample=resample, mask=mask)
        _LU = np.array(raster['img'], dtype=np.int8)
        _R = _LU == 0
        self.var = {'R': _R}
        if mask is not None:
            try:
                self.cmap += ['white']
            except:
                pass
            _M = _LU == mask
            self.var['M'] = _M
            _LU[_M] = 0
        if self.isProb:
            _LU = self.prob_LU(_LU)
        self.raw = raster
        self.var['LU'] = _LU
        # self.var = {'LU': _LU, 'R': _R, 'M': _M}
        print(f"Loaded raster: {LUname}[{driver}/{resample}x: {self.var['LU'].shape}]")

    def save_vector(self, name, optLU=None, driver='shapefile', encoding='utf-8'):
        print(f'Saving {name}({driver})')
        optLU = self.to_LU(optLU)
        name = os.path.join(self.path, name)
        self.var['opt'] = optLU
        if driver == 'feather':
            self.var.to_feather(f'name.feather')
        self.var.to_file(name, driver=driver, encoding=encoding)

    def save_raster(self, name, optLU=None, driver='TIF'):
        print(f'Saving {name}({driver})')
        optLU = self.to_LU(optLU, self.raw['mask'])
        name = os.path.join(self.path, name)
        write_raster(name, optLU, self.raw, driver=driver)

    def save(self, name, optLU=None, **kwargs):
        if self.VorR == 'V':
            self.save_vector(name, optLU, **kwargs)
        if self.VorR == 'R':
            self.save_raster(name, optLU, **kwargs)

    def prob_LU(self, LU=None):
        if LU is None:
            LU = self.var['LU']
            if self.isProb == True:
                return np.array(LU, dtype=np.float32)
        LU = np.array(LU, dtype=np.int8)
        LU = np.array([LU == (c + 1) for c in range(self.z_LU)], dtype=np.float32)
        return np.moveaxis(LU, 0, -1)
    
    def cate_LU(self, pLU=None):
        if pLU is None:
            pLU = self.var['LU']
            if self.isProb == False:
                return np.array(pLU, dtype=np.int8)
        LU = np.argmax(np.array(pLU, dtype=np.float32), axis=-1) + 1
        return np.array(LU, dtype=np.int8)

    def to_LU(self, LU=None, mask=None):
        if LU is None:
            LU = self.cate_LU(LU)
        _R = np.array(self.var['R'], dtype=bool)
        LU[_R] = 0
        if self.VorR == 'R':
            if 'M' in self.var:
                LU[self.var['M']] = self.z_LU + 1 if mask is None else mask
        return LU

    def new_plot(self, ax=None, figsize=(5, 5), bg=None, **bgarg):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None
        ax.axis('off')
        if bg is not None:
            bg.plot(ax=ax, **bgarg)
        return fig, ax
    
    def show_plot(self, fig, show=True, save_name=None, path='plot'):
        if fig is not None:
            fig.tight_layout()
            if save_name is not None:
                path = os.path.join(self.path, path)
                os.makedirs(path, exist_ok=True)
                fig.savefig(os.path.join(path, save_name))
            if show:
                # fig.show()
                plt.show()

    def plot_pLU(self, pLU=None, **kwargs):
        LU = self.cate_LU(pLU)
        self.plot_LU(LU=LU, **kwargs)

    def plot_LU(self, LU=None, ax=None, figsize=(5, 5), bg=None, bgarg={},
                vmm=None, cmap=None, lmap=None, legend=None,
                show=True, save_name=None, path='plot'):
        LU = self.to_LU(LU)
        vmin_, vmax_ = LU.min(), LU.max()
        fig, ax = self.new_plot(ax=ax, figsize=figsize, bg=bg, **bgarg)

        if vmm is None:
            vmin, vmax = 0, self.z_LU
        else:
            vmin, vmax = vmm
        if cmap is None:
            try:
                cmap = self.cmap
            except:
                cmap = None
        try:
            cmap[vmax]
        except IndexError:
            vmax = vmax_
        if lmap is None:
            try:
                lmap = self.legend
            except:
                lmap = None

        if type(legend) is dict:
            legend_kwds = legend
            legend = True
        else:
            legend_kwds = {}
            legend = bool(legend)
        if self.VorR == 'V':
            try:
                # TODO: ncol here need to be gneralized
                vplot_cate(self.var, LU, vmin=vmin, vmax=vmax, ax=ax, cmap=cmap, lmap=lmap, legend=legend, legend_kwds=legend_kwds)
            except:
                vplot_cate(self.raw, LU, vmin=vmin_, vmax=vmax_, ax=ax, cmap=cmap, lmap=lmap, legend=legend, legend_kwds=legend_kwds)
        elif self.VorR == 'R':
            try:
                cmap = ListedColormap(cmap)
            except:
                # print('Use default colormap.')
                cmap = None
            try:
                rplot_cate(LU, vmin=vmin, vmax=vmax, ax=ax, cmap=cmap, lmap=lmap, legend=legend, legend_kwds=legend_kwds)
            except:
                rplot_cate(LU, vmin=vmin_, vmax=vmax_, ax=ax, cmap=cmap, lmap=lmap, legend=legend, legend_kwds=legend_kwds)
        self.show_plot(fig, show=show, save_name=save_name, path=path)
        return ax

    def plot_attr(self, attr, ax=None, figsize=(5, 5), bg=None, bgarg={},
                  vmm=None, cmap=None, legend=None,
                  show=True, save_name=None, path='plot'):
        
        fig, ax = self.new_plot(ax=ax, figsize=figsize, bg=bg, **bgarg)
        if vmm is not None:
            vmin, vmax = vmm
        else:
            vmin, vmax = None, None
        
        if type(legend) is dict:
            legend_kwds = legend
            legend = True
        else:
            legend_kwds = {}
            legend = bool(legend)
        if self.VorR == 'V':
            try:
                self.var.plot(attr, vmin=vmin, vmax=vmax, ax=ax, cmap=cmap, legend=legend, legend_kwds=legend_kwds) # TODO: legend
            except:
                self.raw.plot(attr, vmin=vmin, vmax=vmax, ax=ax, cmap=cmap, legend=legend, legend_kwds=legend_kwds)
        elif self.VorR == 'R':
            imx = ax.imshow(attr, vmin=vmin, vmax=vmax, cmap=cmap)
            # TODO:legend
            if legend:
                plt.colorbar(imx, ax=ax, cmap=cmap, **legend_kwds)
        self.show_plot(fig, show=show, save_name=save_name, path=path)
        return ax


# %%

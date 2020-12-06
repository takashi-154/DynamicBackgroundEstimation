# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:04:52 2020

@author: takashi-154
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tifffile as tiff
import astropy.io.fits as iofits
from scipy import optimize

class DynamicBackgroundEstimation:
    """
    メインの処理関数の集まり。

    """
    
    def __init__(self):
        """
        
        Parameters
        ----------
        name : str
            プログラム名。
        """
        self.name = 'DynamicBackgroundEstimation'
        
        
    def initialize_image(self):
        """
        処理対象の画像の初期化。
    
        Returns
        -------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        """
        img_array = np.full((200,300,3), np.nan, dtype=np.float32)
        print('Fin initializing image.')
        return(img_array)
    
        
    def read_image(self, name:str):
        """
        処理対象の画像を読み込む。
    
        Parameters
        ----------
        name : str
            対象画像のファイルパス名(TIFF, FITS対応)
    
        Returns
        -------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        """
        img_array = None
        if os.path.isfile(name):
            path, ext = os.path.splitext(name)
            ext_lower = str.lower(ext)
            if ext_lower in ('.tif', '.tiff'):
                print('reading tif image...')
                img_array = tiff.imread(name).astype(np.float32)
            elif ext_lower in ('.fits', '.fts', '.fit'):
                print('reading fits image...')
                with iofits.open(name) as f:
                    img_array = np.fliplr(np.rot90(f[0].data.T, -1)).astype(np.float32)
            else:
                print('cannot read image.')
        else:
            print('No such file.')
        print('Fin reading image.')
        return(img_array)


    def save_image(self, name:str, image:np.ndarray, dtype:np.dtype=np.float32):
        """
        numpy配列の画像を指定の形式で保存する。
    
        Parameters
        ----------
        name : str
            保存先のファイルパス名(TIFF, FITS対応)
        image : np.ndarray
            保存する画像のnumpy配列
        dtype : np.dtype, default np.float32
            保存形式(デフォルトは[float,32bit])
        """
        path, ext = os.path.splitext(name)
        ext_lower = str.lower(ext)
        if ext_lower in ('.tif', '.tiff'):
            print('saving tif image...')
            image_cast = image.astype(dtype)
            tiff.imsave(name, image_cast)
        elif ext_lower in ('.fits', '.fts', '.fit'):
            print('saving fits image...')
            image_cast = image.astype(dtype)
            hdu = iofits.PrimaryHDU(np.rot90(np.fliplr(image_cast), 1).T)
            hdulist = iofits.HDUList([hdu])
            hdulist.writeto(name, overwrite=True)
        else:
            print('cannot save image.')
        print('Fin saving image.')
        
        
    def initialize_list(self):
        """
        指定ポイントの一覧を作成する。
    
        Returns
        -------
        target : np.ndarray
            指定ポイントのnumpy配列（uint,16bit）
        """
        print('making new list...')
        target = np.empty((0,2)).astype(np.uint16)
        print('Fin reading list.')
        return(target)
        
    
    def read_list(self, name:str):
        """
        指定ポイントの一覧を読み込む。
    
        Parameters
        ----------
        name : str
            指定ポイントのＸＹ座標を格納したファイルのファイルパス名（.npy形式）
    
        Returns
        -------
        target : np.ndarray
            指定ポイントのnumpy配列（uint,16bit）
        """
        print('reading old list...')
        target = np.load(name)
        print('Fin reading list.')
        return(target)
        
        
    def save_list(self, name:str, target:np.ndarray, is_overwrite:bool=True):
        """
        指定ポイントを保存する。
    
        Parameters
        ----------
        name : str
            保存先のファイルパス名
        target : np.ndarray
            指定ポイントのXY座標を格納したnumpy配列
        is_overwrite : bool, default True
            上書きの可否(デフォルトは[True])
        """
        if not os.path.isfile(name):
            print('saving target list...')
            np.save(name, target)
        elif os.path.isfile(name) and is_overwrite:
            print('overwriting target list...')
            np.save(name, target)
        else:
            print('cannot overwrite list.')
        print('Fin reading list.')
        

    def prepare_plot_point(self, img_array:np.ndarray, target:np.ndarray, box_window:int=20):
        """
        バックグラウンドを推定する指標となるポイントを打つための初期情報出力。
    
        Parameters
        ----------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        target : np.ndarray
            指定ポイントのXY座標を格納したnumpy配列
        box_window : int, default 20
            指定ポイントからのバックグラウンドを推定する面積の範囲（デフォルトは[20]）
    
        Returns
        -------
        point : 
            PointSetterクラス用返り値
        img_comp : 
            PointSetterクラス用返り値
        img_show : 
            PointSetterクラス用返り値
        mouse_show : 
            PointSetterクラス用返り値
        box_show : 
            PointSetterクラス用返り値
        med_show : 
            PointSetterクラス用返り値
        box_window : 
            PointSetterクラス用返り値
        """
        img_comp = (img_array/np.max(img_array)*255).astype('uint8')
        if target.size == 0:
            box_comp_array = np.full((box_window*2, box_window*2, img_comp.shape[2]), np.nan, dtype='uint8')
        else:
            box_comp_array = img_comp[target[-1,1]-box_window:target[-1,1]+box_window,
                                      target[-1,0]-box_window:target[-1,0]+box_window]
        
        box_comp_array = np.mean(box_comp_array, axis=2)
            
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.6)
        gs = gridspec.GridSpec(3,3)
        
        ax0 = fig.add_subplot(gs[:,:2])
        ax0.set_title('left-click: add, right-click: remove')
        img_show = ax0.imshow(img_comp)
        point, = ax0.plot(target[...,0].tolist(), target[...,1].tolist(), marker="o", linestyle='None', color="#FFFF00")
        point.set_picker(True)
        point.set_pickradius(10)
        
        ax1 = fig.add_subplot(gs[0,-1])
        ax1.set_title('mouse window box')
        mouse_show = ax1.imshow(box_comp_array,interpolation='nearest',vmin=0,vmax=255,cmap='inferno')
        
        ax2 = fig.add_subplot(gs[1,-1])
        ax2.set_title('point window box')
        box_show = ax2.imshow(box_comp_array,interpolation='nearest',vmin=0,vmax=255,cmap='inferno')
        
        ax3 = fig.add_subplot(gs[2,-1])
        ax3.set_title('box median')
        ax3.axis('off')
        med_show = ax3.imshow(np.nanmedian(box_comp_array, axis=(0,1), keepdims=True).astype('uint8'),
                              interpolation='nearest',vmin=0,vmax=255,cmap='inferno')
        
        return(fig, point, img_comp, img_show, mouse_show, box_show, med_show, box_window)
        
    
    def postprocess_plot_point(self, pointlist):
        """
        指定したポイント情報をnumpy配列に格納する。
    
        Parameters
        ----------
        pointlist : 
            PointSetterクラス
            
        Returns
        -------
        target : np.ndarray
            指定ポイントのXY座標を格納したnumpy配列
        """
        target = np.vstack((pointlist.xs, pointlist.ys)).T.astype(np.uint16)
        return(target)
    
    
    def check_point_window(self, img_array:np.ndarray, target:np.ndarray, box_window:int=20):
        """
        指定したポイントの画像を表示する。
    
        Parameters
        ----------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        target : np.ndarray
            指定ポイントのXY座標を格納したnumpy配列
        box_window : int, default 20
            指定ポイントからのバックグラウンドを推定する面積の範囲（デフォルトは[20]）
        """
        img_comp = (img_array/np.max(img_array)*255).astype('uint8')
        box = np.empty((box_window*2, box_window*2, img_comp.shape[2], len(target)), dtype='uint8')
        for i in range(len(target)):
            t = target[i]
            x = np.arange(int(t[0])-box_window, int(t[0])+box_window)
            x = x[(0 <= x) & (x < img_comp.shape[1])]
            y = np.arange(int(t[1])-box_window, int(t[1])+box_window)
            y = y[(0 <= y) & (y < img_comp.shape[0])]
            _box = np.full((box_window*2, box_window*2, img_comp.shape[2]), np.nan, dtype='uint8')
            _box[np.ix_(y-np.min(y),x-np.min(x))] = img_comp[np.ix_(y,x)]
            box[:,:,:,i] = _box
        
        target_length = box.shape[3]
        axes = []
        fig = plt.figure()
        for n in range(target_length):
            axes.append(fig.add_subplot(int(np.ceil(np.sqrt(target_length))),
                                        int(np.ceil(np.sqrt(target_length))),
                                        n+1))
            plt.imshow(box[...,n])
            plt.axis('off')
        fig.tight_layout()
        plt.show()
    
    
    def estimate_background(self, img_array:np.ndarray, target:np.ndarray, box_window:int=20):
        """
        指定したポイントの情報を基にバックグラウンドを推定する。
    
        Parameters
        ----------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        target : np.ndarray
            指定ポイントのXY座標を格納したnumpy配列
        box_window : int, default 20
            指定ポイントからのバックグラウンドを推定する面積の範囲（デフォルトは[20]）
            
        Returns
        -------
        model : np.ndarray
            推定されたバックグラウンドのnumpy配列（float,32bit）
        """
        def fit_func(mesh, p, px, py, pxx, pxy, pyy, pxxx, pxxy, pxyy, pyyy, pxxxx, pxxxy, pxxyy, pxyyy, pyyyy): 
            x, y = mesh
            return (p + \
                    px*x + py*y + \
                    pxx*(x*x) + pxy*(x*y) + pyy*(y*y) + \
                    pxxx*(x*x*x) + pxxy*(x*x*y) + pxyy*(x*y*y) + pyyy*(y*y*y) + \
                    pxxxx*(x*x*x*x) + pxxxy*(x*x*x*y) + pxxyy*(x*x*y*y) + pxyyy*(x*y*y*y) + pyyyy*(y*y*y*y)
                    )
        
        if np.all(np.isnan(img_array)):
            model = None
        else:
            box = np.empty((box_window*2, box_window*2, img_array.shape[2], len(target)), dtype='float32')
            for i in range(len(target)):
                t = target[i]
                x = np.arange(int(t[0])-box_window, int(t[0])+box_window)
                x = x[(0 <= x) & (x < img_array.shape[1])]
                y = np.arange(int(t[1])-box_window, int(t[1])+box_window)
                y = y[(0 <= y) & (y < img_array.shape[0])]
                _box = np.full((box_window*2, box_window*2, img_array.shape[2]), np.nan, dtype='float32')
                _box[np.ix_(y-np.min(y),x-np.min(x))] = img_array[np.ix_(y,x)]
                box[:,:,:,i] = _box
            target_median = np.nanmedian(box, axis=(0,1))
            
            model = np.empty_like(img_array)
            for i in range(model.shape[2]):
                initial = np.array([np.mean(target_median[i,...]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                popt, _ = optimize.curve_fit(fit_func, target.T, target_median[i,...], p0=initial)
                mesh = np.meshgrid(np.linspace(1,img_array.shape[1],img_array.shape[1]),np.linspace(1,img_array.shape[0],img_array.shape[0]))
                model[...,i] = fit_func(mesh, *popt)

        return(model)
    
    
    def check_background(self, img_array:np.ndarray, model:np.ndarray):
        """
        推定したバックグラウンドモデルを表示する。
    
        Parameters
        ----------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        model : np.ndarray
            推定されたバックグラウンドのnumpy配列（float,32bit）
        """
        fig=plt.figure()
        ax1=fig.add_subplot(131)
        ax1.set_title('Base image')
        ax1.imshow((img_array/np.max(img_array)*255).astype('uint8'))
        ax2=fig.add_subplot(132)
        ax2.set_title('Background model')
        ax2.imshow((model/np.max(img_array)*255).astype('uint8'))
        ax3=fig.add_subplot(133)
        ax3.set_title('Heatmap')
        ax3.imshow((((model-np.min(model))/(np.max(model)-np.min(model)))*255).astype('uint8'),
                   interpolation='nearest',vmin=0,vmax=255,cmap='inferno')
        fig.tight_layout()
        plt.show()


    def subtract_background(self, img_array:np.ndarray, model:np.ndarray):
        """
        元画像からバックグラウンドを減算する。
    
        Parameters
        ----------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        model : np.ndarray
            推定されたバックグラウンドのnumpy配列（float,32bit）
            
        Returns
        -------
        output : np.ndarray
            バックグラウンドを差し引いた画像のnumpy配列（float,32bit）
        """
        output = img_array - model - np.min(img_array - model)
        return(output)



class PointSetter:
    """
    matplotlib上で指定ポイントを追加するためのヘルパー関数を保持する。

    Attributes
    ----------
    line : 
        指定ポイントの配列。
    img : 
        画像。
    img_show : 
        matplotlib_画像。
    mouse_show : 
        matplotlib_mouse画像。
    box_show : 
        matplotlib_window画像。
    med_show : 
        matplotlib_window画像のmedian。
    window : 
        windowサイズ。
    xs : 
        指定ポイントのX軸
    ys : 
        指定ポイントのX軸
    cidadd : 
        指定ポイント追加の関数呼び出し
    cidhandle : 
        指定ポイント選択・削除の関数呼び出し
    """
    
    def __init__(self, line, img, img_show, mouse_show, box_show, med_show, window):
        """
        
        Parameters
        ----------
        line : 
            指定ポイントの配列。
        img : 
            画像。
        img_show : 
            画像。
        mouse_show : 
            mouse画像。
        box_show : 
            window画像。
        med_show : 
            window画像のmedian。
        window : 
            windowサイズ。
        """
        self.line = line
        self.img = img
        self.img_show = img_show
        self.mouse_show = mouse_show
        self.box_show = box_show
        self.med_show = med_show
        self.window = window
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cidadd = line.figure.canvas.mpl_connect('button_press_event', self.on_add)
        self.cidhandle = line.figure.canvas.mpl_connect('pick_event', self.on_handle)
        self.cidmotion = line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def set_image(self, img_array):
        """
        画像を更新する。
        
        Parameters
        ----------
        img_array : np.ndarray
            画像のnumpy配列（float,32bit）
        """
        img_comp = (img_array/np.max(img_array)*255).astype('uint8')
        self.img = img_comp
        self.img_show.set_data(img_comp)
        self.img_show.set_extent((0, img_comp.shape[1], img_comp.shape[0], 0))
        new_box = np.full((self.window*2, self.window*2, img_comp.shape[2]), 0, dtype='uint8')
        new_box = np.nanmean(new_box, axis=2)
        new_med = np.nanmedian(new_box, axis=(0,1), keepdims=True).astype('uint8')
        self.box_show.set_data(new_box)
        self.med_show.set_data(new_med)
        self.line.figure.canvas.draw()
        

    def set_target(self, target):
        """
        指定ポイントを更新する。
    
        Parameters
        ----------
        target : np.ndarray
            指定ポイントのXY座標を格納したnumpy配列
        """
        self.xs = target[...,0].tolist()
        self.ys = target[...,1].tolist()
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()
        
        
    def set_window(self, new_window):
        """
        指定ポイントを更新する。
    
        Parameters
        ----------
        new_window : int
            新しいウインドウサイズ
        """
        self.window = new_window
        new_box = np.full((self.window*2, self.window*2, self.img.shape[2]), np.nan, dtype='uint8')
        if len(self.xs) > 0: 
            _x = np.arange(int(self.xs[-1])-self.window, int(self.xs[-1])+self.window)
            _x = _x[(0 <= _x) & (_x < self.img.shape[1])]
            _y = np.arange(int(self.ys[-1])-self.window, int(self.ys[-1])+self.window)
            _y = _y[(0 <= _y) & (_y < self.img.shape[0])]
            new_box[np.ix_(_y-np.min(_y),_x-np.min(_x))] = self.img[np.ix_(_y,_x)]
        new_box = np.nanmean(new_box, axis=2)
        new_med = np.nanmedian(new_box, axis=(0,1), keepdims=True).astype('uint8')
        self.box_show.set_data(new_box)
        self.box_show.set_extent((0, new_box.shape[1], new_box.shape[0], 0))
        self.med_show.set_data(new_med)
        self.line.figure.canvas.draw()
        

    def on_add(self, event):
        """
        指定ポイントを追加する。
    
        Parameters
        ----------
        event : 
            クリックイベント。
        """
        if event.inaxes != self.line.axes: return
        if event.button == 1:
            x = event.xdata
            y = event.ydata
            self.xs.append(x)
            self.ys.append(y)
            self.line.set_data(self.xs, self.ys)
            _x = np.arange(int(x)-self.window, int(x)+self.window)
            _x = _x[(0 <= _x) & (_x < self.img.shape[1])]
            _y = np.arange(int(y)-self.window, int(y)+self.window)
            _y = _y[(0 <= _y) & (_y < self.img.shape[0])]
            new_box = np.full((self.window*2, self.window*2, self.img.shape[2]), np.nan, dtype='float32')
            new_box[np.ix_(_y-np.min(_y),_x-np.min(_x))] = self.img[np.ix_(_y,_x)]
            new_box = np.mean(new_box, axis=2)
            new_med = np.nanmedian(new_box, axis=(0,1), keepdims=True).astype('uint8')
            self.box_show.set_data(new_box)
            self.med_show.set_data(new_med)
            self.line.figure.canvas.draw()
        
        
    def on_handle(self, event):
        """
        指定ポイントを選択・削除する。
    
        Parameters
        ----------
        event : 
            クリックイベント。
        """
        if event.artist != self.line: return
        if event.mouseevent.button == 1:
            x = list(event.artist.get_xdata())
            y = list(event.artist.get_ydata())
            ind = event.ind
            match = (np.array([self.xs, self.ys]).T == np.array([x[ind[0]], y[ind[0]]]).T).T.all(axis=0)
            index = np.where(match)[0]
            if index.size > 0: 
                _x = np.arange(int(self.xs[index[0]])-self.window, int(self.xs[index[0]])+self.window)
                _x = _x[(0 <= _x) & (_x < self.img.shape[1])]
                _y = np.arange(int(self.ys[index[0]])-self.window, int(self.ys[index[0]])+self.window)
                _y = _y[(0 <= _y) & (_y < self.img.shape[0])]
                new_box = np.full((self.window*2, self.window*2, self.img.shape[2]), np.nan, dtype='float32')
                new_box[np.ix_(_y-np.min(_y),_x-np.min(_x))] = self.img[np.ix_(_y,_x)]
                new_box = np.mean(new_box, axis=2)
                new_med = np.nanmedian(new_box, axis=(0,1), keepdims=True).astype('uint8')
                self.box_show.set_data(new_box)
                self.med_show.set_data(new_med)
            else: return
        if event.mouseevent.button == 3:
            x = list(event.artist.get_xdata())
            y = list(event.artist.get_ydata())
            ind = event.ind
            match = (np.array([self.xs, self.ys]).T == np.array([x[ind[0]], y[ind[0]]]).T).T.all(axis=0)
            index = np.where(match)[0]
            if index.size > 0: 
                self.xs.pop(index[0])
                self.ys.pop(index[0])
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()
            else: return
            
            
    def on_motion(self, event):
        """
        拡大鏡を表示。
    
        Parameters
        ----------
        event : 
            モーションイベント。
        """
        x = event.xdata
        y = event.ydata
        if event.inaxes == self.line.axes:
            _x = np.arange(int(x)-self.window, int(x)+self.window)
            _x = _x[(0 <= _x) & (_x < self.img.shape[1])]
            _y = np.arange(int(y)-self.window, int(y)+self.window)
            _y = _y[(0 <= _y) & (_y < self.img.shape[0])]
            new_box = np.full((self.window*2, self.window*2, self.img.shape[2]), np.nan, dtype='float32')
            new_box[np.ix_(_y-np.min(_y),_x-np.min(_x))] = self.img[np.ix_(_y,_x)]
            new_box = np.mean(new_box, axis=2)
            self.mouse_show.set_data(new_box)
            self.line.figure.canvas.draw()


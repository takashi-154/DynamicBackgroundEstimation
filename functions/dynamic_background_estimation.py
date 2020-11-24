# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:47:24 2020

@author: takashi-154
"""

import os
import sys
import tkinter as tk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import func_dynamic_background_estimation as dbe


iDir = os.path.abspath(os.path.dirname(__file__))

image_path = tk.filedialog.askopenfilename(filetypes = [("image file", ("*.tiff", "*.tif", "*.fts", "*.fit", "*.fits"))], 
                                     initialdir = iDir, 
                                     title = "Select image file")
if image_path == "":
    sys.exit(1)

target_path = tk.filedialog.askopenfilename(filetypes = [("list file", "*.npy")], 
                                     initialdir = iDir, 
                                     title = "Select point list file")
if target_path == "":
    sys.exit(1)
    
estimate = dbe.DynamicBackgroundEstimation()
img_array = estimate.read_image(image_path)
target = estimate.read_list(target_path)
fig, point, img_comp, box_comp, box_window, box_median = estimate.prepare_plot_point(img_array, target, 20)


def create_and_output(dbe, img_array, pointlist):
    button['state'] = tk.DISABLED
    button['text'] = 'progress...'
    save_path = tk.filedialog.asksaveasfilename(filetypes = [("image file", ("*.tiff", "*.tif", "*.fts", "*.fit", "*.fits"))], 
                                                initialdir = iDir, title = "Save as")
    thread = threading.Thread(target = tmp, args=(dbe, img_array, pointlist, save_path, ))
    thread.start()
    

def tmp(dbe, img_array, pointlist, save_path):
    target = dbe.postprocess_plot_point(pointlist)
    model = dbe.estimate_background(img_array, target, 20)
    output = dbe.subtract_background(img_array, model)
    dbe.save_list('output', target)
    dbe.save_image(save_path, output)
    save_model_path = os.path.splitext(save_path)[0]+'_model'+os.path.splitext(save_path)[1]
    dbe.save_image(save_model_path, model)
    button['state'] = tk.NORMAL
    button['text'] = 'create model and output image'


def _destroyWindow():
    root.quit()
    root.destroy()
    
    
root = tk.Tk()
root.title(u"Dynamic Background Estimation")
root.withdraw()
root.protocol('WM_DELETE_WINDOW', _destroyWindow)

canvas = FigureCanvasTkAgg(fig, master=root)
# canvas.draw()
# canvas.get_tk_widget().grid(row=0, column=0, columnspan=1, sticky='news') 
canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx = 5, pady = 5)
pointlist = dbe.PointSetter(point, img_comp, box_comp, box_window, box_median)

toolbar = tk.Frame(root)
button = tk.Button(toolbar, text='create model and output image', 
                   command=lambda: create_and_output(estimate, img_array, pointlist))
button.pack(side="top")
toolbar.pack(side="bottom", fill="both", padx = 5, pady = 5)

labels = tk.Entry(width=20)
labels.grid(row=1, column=1, columnspan=2, sticky='news')

root.update()
root.deiconify()
root.mainloop()
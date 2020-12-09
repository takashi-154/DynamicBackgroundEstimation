# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:47:24 2020

@author: takashi-154
"""

import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import func_dynamic_background_estimation as dbe

global img_array, target, window_size, img_color, img_scaled


def root_destroyWindow():
    root.quit()
    root.destroy()
    
    
root = tk.Tk()
root.title(u"Dynamic Background Estimation")
root.withdraw()
root.protocol('WM_DELETE_WINDOW', root_destroyWindow)

estimate = dbe.DynamicBackgroundEstimation()
iDir = os.path.abspath(os.path.dirname(__file__))
img_array = estimate.initialize_image()
target = estimate.initialize_list()
window_size = 20
img_color = 0
img_scaled = False
fig, point, img_comp, img_display, img_show, mouse_show, box_show, med_show, \
    ax0, ax1, ax2, ax3 = estimate.prepare_plot_point(img_array, target, window_size, img_color, img_scaled)



def load_image(dbe, pointlist):
    global img_array, target
    button_load_image['state'] = tk.DISABLED
    button_load_image['text'] = 'progress...'
    image_path = tk.filedialog.askopenfilename(filetypes = [("image file", ("*.tiff", "*.tif", "*.fts", "*.fit", "*.fits"))], 
                                     initialdir = iDir, 
                                     title = "Load image file")
    if image_path != "":
        img_array = dbe.read_image(image_path)
        pointlist.set_image(img_array)
        target = dbe.initialize_list()
        pointlist.set_target(target)
    button_load_image['state'] = tk.NORMAL
    button_load_image['text'] = 'load image'
    

def output_image(dbe, save_path, output, image_type, model):
    dbe.save_image(save_path, output, image_type)
    save_model_path = os.path.splitext(save_path)[0]+'_model'+os.path.splitext(save_path)[1]
    dbe.save_image(save_model_path, model, image_type)
    tk.messagebox.showinfo(title="Finish", message="Finish creating model and saving image")
    button_output_image['state'] = tk.NORMAL
    button_output_image['text'] = 'create model and save image'


def select_image_type(dbe, save_path, output, model):
    v = tk.IntVar()
    v.set(2)
    dtype = [
        "16bit, integer",
        "32bit, integer",
        "32bit, rational",
    ]
    dtype_name = [
        "uint16",
        "uint32",
        "float32",
    ]
    img_type = dtype_name[v.get()]
    
    def popup_destroyWindow():
        popup.destroy()
        button_output_image['state'] = tk.NORMAL
        button_output_image['text'] = 'create model and save image'
    
    popup = tk.Toplevel(root)
    popup.geometry('200x150')
    popup.resizable(width=False, height=False)
    popup.title(u"Select image type")
    popup.protocol('WM_DELETE_WINDOW', popup_destroyWindow)
    
    popup_label = tk.Label(popup, text="select image type")
    popup_label.pack(anchor="w", fill="both", pady = 2)
    
    def show_selected():
        global img_type
        img_type = dtype_name[v.get()]
        
    def close_window():
        global img_type
        img_type = dtype_name[v.get()]
        popup.destroy()
        output_image(dbe, save_path, output, img_type, model)
    
    for val, dt in enumerate(dtype):
        tk.Radiobutton(popup, 
                      text=dt,
                      variable=v,
                      command=show_selected,
                      value=val).pack(anchor="w", fill="both", pady = 2)
    
    button_popup = tk.Button(popup, text="OK", command=close_window)
    button_popup.pack(side="bottom", fill="both", pady = 2)


def create_and_output_image(dbe, pointlist):
    global img_array, target, window_size
    button_output_image['state'] = tk.DISABLED
    button_output_image['text'] = 'progress...'
    save_path = tk.filedialog.asksaveasfilename(filetypes = [("image file", ("*.tiff", "*.tif", "*.fts", "*.fit", "*.fits"))], 
                                                initialdir = iDir, title = "Save as", initialfile = "output.fts")
    if save_path != "": 
        target = dbe.postprocess_plot_point(pointlist)
        if len(target) == 0:
            tk.messagebox.showerror(title="ERROR", message="ERROR: cannot read point")
            button_output_image['state'] = tk.NORMAL
            button_output_image['text'] = 'create model and save image'
        else:
            model = dbe.estimate_background(img_array, target, window_size)
            if model is None:
                tk.messagebox.showerror(title="ERROR", message="ERROR: cannot read image")
                button_output_image['state'] = tk.NORMAL
                button_output_image['text'] = 'create model and save image'
            else: 
                output = dbe.subtract_background(img_array, model)
                select_image_type(dbe, save_path, output, model)
    else:
        button_output_image['state'] = tk.NORMAL
        button_output_image['text'] = 'create model and save image'


def load_point_list(dbe, pointlist):
    global target
    button_load_list['state'] = tk.DISABLED
    button_load_list['text'] = 'progress...'
    load_path = tk.filedialog.askopenfilename(filetypes = [("list file", "*.npy")], 
                                              initialdir = iDir, title = "Load point list file")
    if load_path != "":
        target = dbe.read_list(load_path)
        pointlist.set_target(target)
    button_load_list['state'] = tk.NORMAL
    button_load_list['text'] = 'load point list'
    

def save_point_list(dbe, pointlist):
    global target
    button_save_list['state'] = tk.DISABLED
    button_save_list['text'] = 'progress...'
    save_path = tk.filedialog.asksaveasfilename(filetypes = [("list file", "*.npy")], 
                                                initialdir = iDir, title = "Save as", initialfile = "pointlist.npy")
    if save_path != "": 
        target = dbe.postprocess_plot_point(pointlist)
        dbe.save_list(save_path, target)
        tk.messagebox.showinfo(title="Finish", message="Finish saving point list")
    button_save_list['state'] = tk.NORMAL
    button_save_list['text'] = 'save point list'


def change_window_size(value):
    global window_size
    window_size = int(value)
    pointlist.set_window(window_size)
    
    
def change_img_color():
    global img_color
    img_color = int(color_var.get())
    pointlist.set_img_color(img_color)


def change_img_scaled():
    global img_scaled
    img_scaled = scaled_check.get()
    pointlist.set_img_scaled(img_scaled)


canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx = 5, pady = 5)
pointlist = dbe.PointSetter(fig, point, img_comp, img_display, img_show, mouse_show, box_show, med_show, 
                            window_size, img_color, img_scaled, ax0, ax1, ax2, ax3)


toolbar = tk.Frame(root)

slider_label = tk.Label(toolbar, text="window length (x2)")
slider_label.grid(row=0, column=0, columnspan=1, sticky="ew", padx = 5, pady = 5)

slider_var = tk.IntVar(toolbar, value=window_size)
slider_scale = tk.Scale(toolbar,
                        from_=1,
                        to=200,
                        resolution=1,
                        variable=slider_var,
                        orient=tk.HORIZONTAL,
                        command=change_window_size
                        )
slider_scale.grid(row=0, column=1, columnspan=2, sticky="ew", padx = 5, pady = 5)
scaled_check = tk.BooleanVar(toolbar, value=False)
tk.Checkbutton(toolbar, text='normalized', variable=scaled_check, command=change_img_scaled
               ).grid(row=0, column=3, sticky="ew", padx = 5, pady = 5)

color_var = tk.IntVar(toolbar, value=img_color)
tk.Radiobutton(toolbar, text='RGB', value=0, variable=color_var, command=change_img_color
              ).grid(row=1, column=0, sticky="ew", padx = 5, pady = 5)
tk.Radiobutton(toolbar, text='R', value=1, variable=color_var, command=change_img_color
              ).grid(row=1, column=1, sticky="ew", padx = 5, pady = 5)
tk.Radiobutton(toolbar, text='G', value=2, variable=color_var, command=change_img_color
              ).grid(row=1, column=2, sticky="ew", padx = 5, pady = 5)
tk.Radiobutton(toolbar, text='B', value=3, variable=color_var, command=change_img_color
              ).grid(row=1, column=3, sticky="ew", padx = 5, pady = 5)


button_load_image = tk.Button(toolbar, text='load image', 
                   command=lambda: load_image(estimate, pointlist))
button_load_image.grid(row=2, column=0, sticky="ew", padx = 5, pady = 5)

button_load_list = tk.Button(toolbar, text='load point list', 
                   command=lambda: load_point_list(estimate, pointlist))
button_load_list.grid(row=2, column=1, sticky="ew", padx = 5, pady = 5)

button_save_list = tk.Button(toolbar, text='save point list', 
                   command=lambda: save_point_list(estimate, pointlist))
button_save_list.grid(row=2, column=2, sticky="ew", padx = 5, pady = 5)

button_output_image = tk.Button(toolbar, text='create model and save image', 
                   command=lambda: create_and_output_image(estimate, pointlist))
button_output_image.grid(row=2, column=3, sticky="ew", padx = 5, pady = 5)

toolbar.pack(side="bottom", fill="none", padx=5, pady=5, expand=False)


root.update()
root.deiconify()
root.mainloop()

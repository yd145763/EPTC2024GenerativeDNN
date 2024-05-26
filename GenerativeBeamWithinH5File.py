# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:55:23 2024

@author: limyu
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import time
from matplotlib.colors import LogNorm
from sklearn.preprocessing import MinMaxScaler

horizontal_peaks = []
horizontal_peaks_position = []
horizontal_peaks_max = []
horizontal_half = []
horizontal_full = []
horizontal_std_list = []

verticle_peaks = []
verticle_peaks_position = []
verticle_peaks_max = []
verticle_half = []
verticle_full = []
verticle_std_list = []

max_field_list = []

filename = ["grating012umpitch05dutycycle15um", "grating012umpitch05dutycycle20um", "grating12_11pitch2_8"]
link = ["C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle15um.h5", "C:\\Users\\limyu\\Downloads\grating012umpitch05dutycycle20um.h5"]

file = "grating012umpitch05dutycycle15um"
file_path = "C:\\Users\\limyu\\Google Drive\\3d plots\\"
# Load the h5 file
with h5py.File(file_path+file+".h5", 'r') as f:
    # Get the dataset
    dset = f[file]
    # Load the dataset into a numpy array
    arr_3d_loaded = dset[()]

steps = 10
x = np.arange(0,arr_3d_loaded.shape[0],1)
y = np.arange(0,arr_3d_loaded.shape[1],1)
z = np.arange(97,arr_3d_loaded.shape[2],1)

x_train = x[::steps]
y_train = y[::steps]
z_train = z[::steps]
i_list = []
j_list = []
k_list = []
e_field_list = []
for i in x_train:
    for j in y_train:
        for k in z_train:
            print(i, j, k, arr_3d_loaded[i,j,k])
            i_list.append(i)
            j_list.append(j)
            k_list.append(k)
            e_field_list.append(arr_3d_loaded[i,j,k])

x_test = x[::steps]
y_test = y[::steps]
z_test = z[5::steps]
i_test_list = []
j_test_list = []
k_test_list = []
e_field_test_list = []
for i in x_test:
    for j in y_test:
        for k in z_test:
            print(i, j, k, arr_3d_loaded[i,j,k])
            i_test_list.append(i)
            j_test_list.append(j)
            k_test_list.append(k)
            e_field_test_list.append(arr_3d_loaded[i,j,k])

X_train = pd.DataFrame()
X_train['i_list'] = i_list
X_train['j_list'] = j_list
X_train['k_list'] = k_list
scalerx = MinMaxScaler()
X_train = pd.DataFrame(scalerx.fit_transform(X_train), columns=X_train.columns)


Y_train = pd.DataFrame()
Y_train['e_field_list'] = e_field_list
scalery = MinMaxScaler()
Y_train = pd.DataFrame(scalery.fit_transform(Y_train), columns=Y_train.columns)

X_test = pd.DataFrame()
X_test['i_test_list'] = i_test_list
X_test['j_test_list'] = j_test_list
X_test['k_test_list'] = k_test_list
X_test = pd.DataFrame(scalerx.fit_transform(X_test), columns=X_test.columns)


Y_test = pd.DataFrame()
Y_test['e_field__test_list'] = e_field_test_list


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
dense_layer = 10
layer_size = 10
model = Sequential()
model.add(Dense(len(X_train.keys()),  input_shape=[len(X_train.keys())]))
for _ in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('elu'))
        #layer_size = int(round(layer_size*0.9, 0))

model.add(Dense(Y_train.shape[1]))


# Compile the model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

history = model.fit(X_train, Y_train, epochs=100,validation_data=(X_test, Y_test), batch_size = 10)


prediction = model.predict(X_test)

X_test = pd.DataFrame(X_test)
X_test = pd.DataFrame(scalerx.inverse_transform(X_test), columns=X_test.columns)
Y_pred = pd.DataFrame(prediction)
Y_pred = pd.DataFrame(scalery.inverse_transform(Y_pred), columns=Y_pred.columns)


plot_test = pd.concat([X_test, Y_pred], axis=1, ignore_index=True)
plot_test[0] = plot_test[0].astype(int)
plot_test[1] = plot_test[1].astype(int)
plot_test[2] = plot_test[2].astype(int)
print(plot_test.isnull().sum())

Z = plot_test[2].unique()
Z = [int(i) for i in Z]

for k in Z:
    plot_test_flattened_one_layer = plot_test[plot_test[2] ==k]
    
    I = plot_test_flattened_one_layer[0].unique()
    I = [int(i) for i in I]
    plot_test_one_layer = pd.DataFrame()
    for i in I:
        plot_test_flattened_one_layer_i = plot_test_flattened_one_layer[plot_test_flattened_one_layer[0]==int(i)]
        plot_test_flattened_one_layer_i_field = plot_test_flattened_one_layer_i.iloc[:,-1]
        plot_test_flattened_one_layer_i_field=plot_test_flattened_one_layer_i_field.reset_index(drop=True)    
        plot_test_one_layer[int(i)] = plot_test_flattened_one_layer_i_field

    x_plot = np.linspace(-20, 80, num=plot_test_one_layer.shape[1])
    y_plot = np.linspace(-25, 25, num =plot_test_one_layer.shape[0])
    colorbarmax = plot_test_one_layer.max().max()
    X,Y = np.meshgrid(x_plot,y_plot)
    fig = plt.figure(figsize=(18, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,plot_test_one_layer, 200, zdir='z', offset=-100, cmap='twilight_shifted')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(20)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(20)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.savefig(file_path+"predicted_"+str(k))
    plt.show()
    plt.close()

plot_test = pd.concat([X_test, Y_test], axis=1, ignore_index=True)
plot_test[0] = plot_test[0].astype(int)
plot_test[1] = plot_test[1].astype(int)
plot_test[2] = plot_test[2].astype(int)
print(plot_test.isnull().sum())

Z = plot_test[2].unique()
Z = [int(i) for i in Z]

for k in Z:
    plot_test_flattened_one_layer = plot_test[plot_test[2] ==k]
    
    I = plot_test_flattened_one_layer[0].unique()
    I = [int(i) for i in I]
    plot_test_one_layer = pd.DataFrame()
    for i in I:
        plot_test_flattened_one_layer_i = plot_test_flattened_one_layer[plot_test_flattened_one_layer[0]==int(i)]
        plot_test_flattened_one_layer_i_field = plot_test_flattened_one_layer_i.iloc[:,-1]
        plot_test_flattened_one_layer_i_field=plot_test_flattened_one_layer_i_field.reset_index(drop=True)    
        plot_test_one_layer[int(i)] = plot_test_flattened_one_layer_i_field

    x_plot = np.linspace(-20, 80, num=plot_test_one_layer.shape[1])
    y_plot = np.linspace(-25, 25, num =plot_test_one_layer.shape[0])
    colorbarmax = plot_test_one_layer.max().max()
    X,Y = np.meshgrid(x_plot,y_plot)
    fig = plt.figure(figsize=(18, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,plot_test_one_layer, 200, zdir='z', offset=-100, cmap='twilight_shifted')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(20)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(20)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.savefig(file_path+"actual_"+str(k))
    plt.show()
    plt.close()

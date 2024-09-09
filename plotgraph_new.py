# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 23:03:41 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:37:01 2024

@author: limyu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

horizontal_peaks_actual = []
horizontal_peaks_position_actual = []
horizontal_peaks_max_actual = []
horizontal_mse_list_actual = []
horizontal_half_actual = []
horizontal_full_actual = []

verticle_peaks_actual = []
verticle_peaks_position_actual = []
verticle_peaks_max_actual = []
verticle_mse_list_actual = []
verticle_half_actual = []
verticle_full_actual = []

horizontal_peaks_predicted = []
horizontal_peaks_position_predicted = []
horizontal_peaks_max_predicted = []
horizontal_mse_list_predicted = []
horizontal_half_predicted = []
horizontal_full_predicted = []

verticle_peaks_predicted = []
verticle_peaks_position_predicted = []
verticle_peaks_max_predicted = []
verticle_mse_list_predicted = []
verticle_half_predicted = []
verticle_full_predicted = []




plot_test_pred = pd.read_csv("C:\\Users\\limyu\\Google Drive\\EPTC Generative Beam\\predicted_full_data1.csv", index_col = 0)

plot_test_pred = plot_test_pred.reset_index(drop=True)
plot_test_pred.columns = range(plot_test_pred.shape[1])


plot_test_actual = pd.read_csv("C:\\Users\\limyu\\Google Drive\\EPTC Generative Beam\\actual_full_data1.csv", index_col = 0)

plot_test_actual.columns = range(plot_test_actual.shape[1])
plot_test_actual = plot_test_actual.reset_index(drop=True)

Z = plot_test_actual[2].unique()
Z = [int(i) for i in Z]
z_position = np.linspace(-5,45,num=317)

for k in Z:
    plot_test_flattened_one_layer = plot_test_actual[plot_test_actual[2] ==k]
    
    I = plot_test_flattened_one_layer[0].unique()
    I = [int(i) for i in I]
    plot_test_one_layer = pd.DataFrame()
    for i in I:
        plot_test_flattened_one_layer_i = plot_test_flattened_one_layer[plot_test_flattened_one_layer[0]==int(i)]
        plot_test_flattened_one_layer_i_field = plot_test_flattened_one_layer_i.iloc[:,-1]
        plot_test_flattened_one_layer_i_field=plot_test_flattened_one_layer_i_field.reset_index(drop=True)    
        plot_test_one_layer[int(i)] = plot_test_flattened_one_layer_i_field


    df_actual = plot_test_one_layer
    df_actual.reset_index(drop=True)
    df_actual.columns = range(df_actual.shape[1])
    x = np.linspace(-20, 80, num=df_actual.shape[1])
    y = np.linspace(-25, 25, num =df_actual.shape[0])
    colorbarmax = df_actual.max().max()
    X,Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(15, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df_actual, 200, zdir='z', offset=-100, cmap='cividis')
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
    plt.title(str(k)+' '+str(z_position[k])+'\n')
    plt.show()
    plt.close()
    
    from sklearn.metrics import mean_squared_error
    from scipy import interpolate
    from scipy.signal import chirp, find_peaks, peak_widths
    from scipy.stats import linregress
    
    max_value = df_actual.max().max()
    max_col = df_actual.max().idxmax()
    max_row = df_actual[max_col].idxmax()
    hor_e = df_actual.iloc[max_row, :]
    ver_e = df_actual.loc[:, max_col]
    
    #horizontal plot
    peaks, _ = find_peaks(hor_e)
    peaks_h = x[peaks]
    peaks_height = hor_e[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0][0])   
    
    
    horizontal_peaks_actual.append(peaks_h)
    horizontal_peaks_position_actual.append(x[np.where(hor_e == max(hor_e))[0][0]])
    horizontal_peaks_max_actual.append(df_actual.max().max())
    results_half = peak_widths(hor_e, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x[-1] - x[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(hor_e, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x[-1] - x[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(hor_e)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(hor_e == max(hor_e))
    max_E_index = int(max_E_index[0][0])
    x_max_E = x[max_E_index]
    distance = [i-x_max_E for i in x]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    mse_dev = mean_squared_error(hor_e, fit_y)
    horizontal_mse_list_actual.append(mse_dev)
       
    horizontal_half_actual.append(FWHM)      
    horizontal_full_actual.append(waist)
    ax2 = plt.axes()
    ax2.plot(x, hor_e)
    ax2.plot(x[peaks], hor_e[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.hlines(*results_full_plot[1:], color="C3")
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("E-field (eV)")
    plt.legend(["E-field (eV)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
    plt.show()
    plt.close()
    
    #vertical plot 
    peaks, _ = find_peaks(ver_e)
    peaks_v = y[peaks]
    peaks_height = ver_e[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0][0])            
    
    verticle_peaks_actual.append(peaks_v)
    verticle_peaks_position_actual.append(y[np.where(ver_e == max(ver_e))[0][0]])
    verticle_peaks_max_actual.append(df_actual.max().max())
    
    results_half = peak_widths(ver_e, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(y[-1] - y[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(y[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(y[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(ver_e, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(y[-1] - y[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(y[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(y[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(ver_e)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(ver_e == max(ver_e))
    max_E_index = int(max_E_index[0][0])
    x_max_E = y[max_E_index]
    distance = [i-x_max_E for i in y]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    mse_dev = mean_squared_error(y, fit_y)
    verticle_mse_list_actual.append(mse_dev)
    
    
    verticle_half_actual.append(FWHM)      
    verticle_full_actual.append(waist)
    
    ax2 = plt.axes()
    ax2.plot(y, ver_e)
    ax2.plot(y[peaks], ver_e[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.hlines(*results_full_plot[1:], color="C3")
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("E-field (eV)")
    plt.legend(["E-field (eV)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'}, loc = 'upper left')
    plt.show()
    plt.close()
    
    # -*- coding: utf-8 -*-
    """
    Created on Sun May 26 08:36:17 2024
    
    @author: limyu
    """
    
    
    plot_test_flattened_one_layer = plot_test_pred[plot_test_pred[2] ==k]
    
    I = plot_test_flattened_one_layer[0].unique()
    I = [int(i) for i in I]
    plot_test_one_layer = pd.DataFrame()
    for i in I:
        plot_test_flattened_one_layer_i = plot_test_flattened_one_layer[plot_test_flattened_one_layer[0]==int(i)]
        plot_test_flattened_one_layer_i_field = plot_test_flattened_one_layer_i.iloc[:,-1]
        plot_test_flattened_one_layer_i_field=plot_test_flattened_one_layer_i_field.reset_index(drop=True)    
        plot_test_one_layer[int(i)] = plot_test_flattened_one_layer_i_field

    df_predicted = plot_test_one_layer
    df_predicted.reset_index(drop=True)
    df_predicted.columns = range(df_predicted.shape[1])
    x = np.linspace(-20, 80, num=df_predicted.shape[1])
    y = np.linspace(-25, 25, num =df_predicted.shape[0])
    colorbarmax = df_predicted.max().max()
    X,Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(15, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df_predicted, 200, zdir='z', offset=-100, cmap='cividis')
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
    plt.title(str(k)+' '+str(z_position[k])+'\n')
    plt.show()
    plt.close()
    
    from sklearn.metrics import mean_squared_error
    from scipy import interpolate
    from scipy.signal import chirp, find_peaks, peak_widths
    from scipy.stats import linregress
    
    max_value = df_predicted.max().max()
    max_col = df_predicted.max().idxmax()
    max_row = df_predicted[max_col].idxmax()
    hor_e = df_predicted.iloc[max_row, :]
    ver_e = df_predicted.loc[:, max_col]
    
    #horizontal plot
    peaks, _ = find_peaks(hor_e)
    peaks_h = x[peaks]
    peaks_height = hor_e[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0][0])   
    
    
    horizontal_peaks_predicted.append(peaks_h)
    horizontal_peaks_position_predicted.append(x[np.where(hor_e == max(hor_e))[0][0]])
    horizontal_peaks_max_predicted.append(df_predicted.max().max())
    results_half = peak_widths(hor_e, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x[-1] - x[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(x[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(x[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(hor_e, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x[-1] - x[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(x[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(x[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(hor_e)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(hor_e == max(hor_e))
    max_E_index = int(max_E_index[0][0])
    x_max_E = x[max_E_index]
    distance = [i-x_max_E for i in x]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    mse_dev = mean_squared_error(hor_e, fit_y)
    horizontal_mse_list_predicted.append(mse_dev)
       
    horizontal_half_predicted.append(FWHM)      
    horizontal_full_predicted.append(waist)
    ax2 = plt.axes()
    ax2.plot(x, hor_e)
    ax2.plot(x[peaks], hor_e[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.hlines(*results_full_plot[1:], color="C3")
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("E-field (eV)")
    plt.legend(["E-field (eV)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
    plt.show()
    plt.close()
    
    #vertical plot 
    peaks, _ = find_peaks(ver_e)
    peaks_v = y[peaks]
    peaks_height = ver_e[peaks]
    max_index = np.where(peaks_height == max(peaks_height))
    max_index = int(max_index[0][0])            
    
    verticle_peaks_predicted.append(peaks_v)
    verticle_peaks_position_predicted.append(y[np.where(ver_e == max(ver_e))[0][0]])
    verticle_peaks_max_predicted.append(df_predicted.max().max())
    
    results_half = peak_widths(ver_e, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(y[-1] - y[-2]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = np.array(y[np.around(x_min, decimals=0).astype(int)])
    x_max = results_half[3]
    x_max = np.array(y[np.around(x_max, decimals=0).astype(int)])    
    results_half_plot = (width, height, x_min, x_max)
    list_of_FWHM = results_half_plot[0]
    FWHM = list_of_FWHM[max_index]
    
    results_full = peak_widths(ver_e, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(y[-1] - y[-2]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = np.array(y[np.around(x_min_f, decimals=0).astype(int)])
    x_max_f = results_full[3]
    x_max_f = np.array(y[np.around(x_max_f, decimals=0).astype(int)]) 
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    list_of_waist = results_full_plot[0]
    waist = list_of_waist[max_index]
    A = max(ver_e)
    B = 1/(waist**2)
    def Gauss(x):
        y = A*np.exp(-1*B*x**2)
        return y
    max_E_index = np.where(ver_e == max(ver_e))
    max_E_index = int(max_E_index[0][0])
    x_max_E = y[max_E_index]
    distance = [i-x_max_E for i in y]
    distance = np.array(distance) 
    fit_y = Gauss(distance)
    mse_dev = mean_squared_error(y, fit_y)
    verticle_mse_list_predicted.append(mse_dev)
    
    
    verticle_half_predicted.append(FWHM)      
    verticle_full_predicted.append(waist)
    
    ax2 = plt.axes()
    ax2.plot(y, ver_e)
    ax2.plot(y[peaks], ver_e[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.hlines(*results_full_plot[1:], color="C3")
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("E-field (eV)")
    plt.legend(["E-field (eV)", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'}, loc = 'upper left')
    plt.show()
    plt.close()




    
z=np.linspace(-5,45,num=317)
z_plot = []
for k in Z:
    z_plot.append(z[k])

fig, ax1 = plt.subplots(figsize=(8, 4))
line1, = ax1.plot(z_plot, verticle_full_actual, label = 'Simulated Beam Waist', linestyle = '-')
line2, = ax1.plot(z_plot, verticle_full_predicted, label = 'Generated Beam Waist', linestyle = '-')
ax1.tick_params(which='major', width=2.00)
ax1.tick_params(which='minor', width=2.00)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_weight("bold")
ax1.yaxis.label.set_fontsize(18)
ax1.yaxis.label.set_weight("bold")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax1.set_ylabel("Beam Waist (µm)")
ax1.set_xlabel("z-position (µm)")
ax1.legend(["Actual", "Generated"], prop={'weight': 'bold'}, loc = 'upper left')


ax2 = ax1.twinx() 
line3, = ax2.plot(z_plot, verticle_peaks_max_actual, label = 'Simulated Max E-field', linestyle = '--')
line4, = ax2.plot(z_plot, verticle_peaks_max_predicted, label = 'Generated Max E-field', linestyle = '--')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax2.set_ylabel("Peak E-field (eV)")

lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, prop={'weight': 'bold'}, loc = 'center right')

plt.show()
plt.close()

I = np.arange(0, len(verticle_full_actual),1)
waist_ape = []
for i in I:
    ape = (abs(verticle_full_predicted[i] - verticle_full_actual[i])/verticle_full_actual[i])*100
    waist_ape.append(ape)
waist_ape.index(max(waist_ape))
field_ape = []
for i in I:
    ape = (abs(verticle_peaks_max_predicted[i] - verticle_peaks_max_actual[i])/verticle_peaks_max_actual[i])*100
    field_ape.append(ape)

fig, ax1 = plt.subplots(figsize=(8, 4))
line1, = ax1.plot(z_plot, waist_ape, label = 'Error (Beam Waist)', linestyle = '-')
ax1.tick_params(which='major', width=2.00)
ax1.tick_params(which='minor', width=2.00)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_weight("bold")
ax1.yaxis.label.set_fontsize(18)
ax1.yaxis.label.set_weight("bold")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
ax1.set_xticklabels(ax1.get_xticks(), weight='bold')
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax1.set_ylabel("APE (Beam Waist, %)")
ax1.set_xlabel("z-position (µm)")
ax1.legend(["Actual", "Generated"], prop={'weight': 'bold'}, loc = 'upper left')


ax2 = ax1.twinx() 
line4, = ax2.plot(z_plot, field_ape, label = 'Error (Max E-field)', linestyle = '-', color = 'orange')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.set_ylabel("APE (Peak E-field, %)")

lines = [line1, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, prop={'weight': 'bold'}, loc = 'center right')
plt.show()
plt.close()

def mean_absolute_percentage_error(actual, forecasted):
    actual, forecasted = np.array(actual), np.array(forecasted)
    # To avoid division by zero, replace 0s in actual with a small number (e.g., 1e-10)
    actual = np.where(actual == 0, 1e-10, actual)
    # Calculate the absolute percentage errors
    absolute_percentage_errors = np.abs((actual - forecasted) / actual) * 100
    # Calculate MAPE
    mape = np.mean(absolute_percentage_errors)
    return mape
mape = mean_absolute_percentage_error(verticle_peaks_max_actual, verticle_peaks_max_predicted)
print(mape)
mape = mean_absolute_percentage_error(verticle_full_actual, verticle_full_predicted)
print(mape)

df_loss = pd.read_csv("C:\\Users\\limyu\\Google Drive\\EPTC Generative Beam\\training_validation1.csv", index_col = 0)
epoch = np.arange(1,101,1)
training_loss = df_loss['training_loss']
validation_loss = df_loss['validation_loss']
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.plot(epoch, training_loss, color = 'red')
ax.plot(epoch, validation_loss, color = 'blue')
#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"], prop={'weight': 'bold','size': 10}, loc = "best")
plt.show()
plt.close()
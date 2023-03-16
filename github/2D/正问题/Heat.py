# -*- coding: utf-8 -*-
"""
考虑做个实际case来增加工作量，实际上这相当于三维
@author: USER
"""

import sys
sys.path.insert(0, '../../Utilities/')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pyDOE import lhs
import time
from PIBNN_tf import PIBNN_tf
import tqdm
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
from post_process import *

def boundary(index,X_star,u_star):
    x = []
    u = []
    for i in index:
        for j in i:           
            if X_star[j,2]!=0:       #去掉初始条件的温度场数据
                x.append(X_star[j,:])
                u.append(u_star[j])
    return np.squeeze(np.array(x)), np.squeeze(np.array(u)).reshape(-1, 1) 
#        
#    
if __name__ == "__main__": 
#      
    k = 52       #导热系数
    c = 434      #比热容
    ρ = 7850     #密度
    a = k/(c*ρ)  #热扩散率
    noise = 0.2      
    
    N_u = 4000     #边界条件训练样本数
    N_f = 30000   #物理场配点数
    layers = [3,20,20,20,1]
    data = scipy.io.loadmat('data.mat')
    
    t = data['time'][:,0:100].flatten()
    t_shape = t.shape[0]
    t = (t.repeat(273,axis=0)).flatten('F')
    x = data['gcoord'][:,0].flatten()[:,None]
    x = (x.repeat(t_shape,axis=1)).flatten('F')  #按列展平，A是行
    y = data['gcoord'][:,1].flatten()[:,None]
    y = (y.repeat(t_shape,axis=1)).flatten('F')
    
    X_xyt = np.vstack((x,y,t)).T   #  n个时间步的坐标矩阵
            
    Exact = np.real(data['T_whole'])[:,0:100]
    
    #X, T = np.meshgrid(x,y)
    
    X_star = X_xyt
    u_star = Exact.flatten('F')[:,None] + noise*np.random.randn(Exact.shape[0]*Exact.shape[1],1)            
    
    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xxx1 = X_star[0:273,:]               #第一列，时间为0的初始条件
    uuu1 = Exact[:,0].reshape(-1, 1)     #初始条件对应的精确解
    uuu1 = uuu1 + np.random.randn(uuu1.shape[0],1)*noise
    
    index = np.array(np.where( (X_star[:,0] < -0.0499) )) #左边界 
    xxx2, uuu2  = boundary(index,X_star,u_star)
    uuu2 = uuu2 + np.random.randn(uuu2.shape[0],1)*noise
    index_zuo = index
    
    index = np.array(np.where( (X_star[:,0] > 0.0499 ) & (X_star[:,1] < 0.0299) & (X_star[:,1] > -0.0299))) #右边界 
    xxx3, uuu3 = boundary(index,X_star,u_star)
    uuu3 = uuu3 + np.random.randn(uuu3.shape[0],1)*noise
    
    index = np.array(np.where( X_star[:,1] > 0.0299)) #上边界
    xxx4, uuu4= boundary(index,X_star,u_star)  
    uuu4 = uuu4 + np.random.randn(uuu4.shape[0],1)*noise
    
    index = np.array(np.where( X_star[:,1] < -0.0299)) #下边界
    xxx5,uuu5 = boundary(index,X_star,u_star)
    uuu5 = uuu5 + np.random.randn(uuu5.shape[0],1)*noise
    index_xia = index
    
    
    X_u_train = np.vstack([xxx1, xxx2, xxx3, xxx4, xxx5])#初始状态和边界条件 
    X_f_train = lb + (ub-lb)*lhs(3, N_f)#  lhs:  latin hypercube sampling
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uuu1, uuu2, uuu3, uuu4, uuu5])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)#在训练集中不放回抽样（N_u）个
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
        
    model = PIBNN_tf(X_u_train, u_train, X_f_train, layers, lb, ub, a, noise)
    
    start_time = time.time()  
#    path_load = r"./weights/09092019/nn.ckpt"    
#    path_save = r"./weights/09111000/nn.ckpt" 
#    model.load(path_load)                
    model.train(50000)
#    model.save(path_save)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred_list = []
    f_x_pred_list = []
    f_y_pred_list = []
    
    for i in tqdm.tqdm(range(500)):                                                 # tqdm进度条模块,n个模型
        u_pred= model.predict(X_star)
        u_pred_list.append(u_pred)
    
    u_preds = np.concatenate(u_pred_list, axis=1)

    u_mean = (np.mean(u_preds, axis=1))
    u_sigma = (np.std(u_preds, axis=1))
    
    error_u = np.linalg.norm(u_star-u_mean,2)/np.linalg.norm(u_star,2)
    print('Error l2: %e' % (error_u))

#    for i in tqdm.tqdm(range(500)):
#        f_pred_x, f_pred_y = model.predict_b(X_star)
#        f_x_pred_list.append(f_pred_x)
#        f_y_pred_list.append(f_pred_y)
#        
#    f_pred_xs = np.concatenate(f_x_pred_list, axis=1)
#    f_pred_ys = np.concatenate(f_y_pred_list, axis=1)
#    f_pred_x_mean = (np.mean(f_pred_xs, axis=1))
#    f_pred_x_sigma = (np.std(f_pred_xs, axis=1))
#    f_pred_y_mean = (np.mean(f_pred_ys, axis=1))
#    f_pred_y_sigma = (np.std(f_pred_ys, axis=1))  
           
#    file_name = '09141943.plt'
#    post_process(file_name, u_mean.reshape(100,273))
    
    
    ####可视化，写得目前不够完善，凑合用#############
#    x_scatter = x
#    y_scatter = y
#    t_scatter = t
#    values = u_mean
#    X_contour, Y_contour = np.meshgrid(x[0:13],np.linspace(0.005,-0.005,21))
#    u_contours = u_mean.reshape(100,273)
#    u_sigmas = u_sigma.reshape(100,273)
#    t_c = 49
#    u_contour = u_contours[t_c,:].reshape(21,13)
#    u_exact = (Exact.T)[t_c,:].reshape(21,13)
#    u_UQ = u_contour+2*u_sigma_plot
#    u_sigma_plot = u_sigmas[t_c,:].reshape(21,13)
#    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
#    ls = LightSource(270, 45)
#    rgb = ls.shade(u_contour, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
#    surf = ax.plot_surface(X_contour, Y_contour, u_contour, rstride=1, cstride=1, facecolors=rgb,
#                           linewidth=0, antialiased=False, shade=False)
#    plt.show()
#    #plt.savefig('1.pdf', bbox_inches='tight')
#    ax.grid(False)
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    ax.set_zlabel("t")
#    plt.figure(2)
#    grid = plt.GridSpec(1, 1, wspace=0.1, hspace=0.45)
#    ax1 = plt.subplot(grid[0,0])
#    plt.xticks(fontsize = 14)
#    plt.yticks(fontsize = 14)                                                           # 初始条件的预测结果
#    plt.plot(Y_contour[:,6], u_contour[:,6].T, 'r-', label='Predictive mean')
#    plt.plot(Y_contour[:,6], u_exact[:,6], label='t=5,x=0 Truth',linestyle = '--')
#    plt.fill_between(Y_contour[:,6].ravel(), 
#                     u_contour[:,6].T + 3 * u_sigma_plot[:,6].T, 
#                     u_contour[:,6].T - 3 * u_sigma_plot[:,6].T, 
#                     alpha=0.5, label='Epistemic uncertainty ±3σ',color = 'orange')
#    plt.legend(loc = 'lower left', fontsize=10);
#    plt.show()
    #plt.savefig('2.pdf', bbox_inches='tight')




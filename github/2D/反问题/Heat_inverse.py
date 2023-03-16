# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:53:25 2019

@author: chaos丶
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from PIBNN_tf_ID import *
import tqdm
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
from post_process import *
from scipy.stats import norm

np.random.seed(1234)
tf.set_random_seed(1234)

    
if __name__ == "__main__": 
     

    k = 52       #导热系数
    c = 434      #比热容
    ρ = 7850     #密度
    a = k/(c*ρ)  #热扩散率
    noise = 0.2
    N_u = 20000  #训练样本数
    
    layers = [3,20,20,20,1]

    
    data = scipy.io.loadmat(r'F:\研究工作目录\PIBNN\PIBNN9.0热传导正反问题-第二类第三类混合边界\反问题\data.mat')
    
    t = data['time'][:,0:100].flatten()
    t_shape = t.shape[0]
    t = (t.repeat(273,axis=0)).flatten('F')
    x = data['gcoord'][:,0].flatten()[:,None]
    x = (x.repeat(t_shape,axis=1)).flatten('F')  #按列展平，A是行
    y = data['gcoord'][:,1].flatten()[:,None]
    y = (y.repeat(t_shape,axis=1)).flatten('F')
    
    X_xyt = np.vstack((x,y,t)).T   # 100个时间步的坐标矩阵
            
    Exact = np.real(data['T_whole'])[:,0:100]
    
    X_star = X_xyt
    u_star = Exact.flatten('F')[:,None] + noise*np.random.randn(Exact.shape[0]*Exact.shape[1],1)

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    
    ######################################################################
    ######################## NoiseData ###############################

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    
    model = PIBNN_tf(X_u_train, u_train, layers, lb, ub,noise)
    
    start_time = time.time()  
    #path_load = r"weights\09241718\nn.ckpt"    
    #path_save = r"weights\20200929\nn.ckpt" 
    #model.load(path_load)                
    model.train(100000)
    #model.save(path_save)
    
    elapsed = time.time() - start_time   

    lambda_1_set = model.lambda_1_set
                             
    
    u_pred_list = []
    f_pred_list = []   
    
    for i in tqdm.tqdm(range(500)):                                                 # tqdm进度条模块,n个模型
        u_pred= model.predict(X_star)
        u_pred_list.append(u_pred)       
    
    u_preds = np.concatenate(u_pred_list, axis=1)

    u_mean = (np.mean(u_preds, axis=1))
    u_sigma = (np.std(u_preds, axis=1))
    
    u_exact_ = Exact.flatten('F')
    #error_u = mean_squared_error(u_exact_,u_mean)
    error_l2 = np.linalg.norm(u_exact_-u_mean,2)/np.linalg.norm(u_exact_,2)#还是相对l2误差比较好
    print('Error l2: %e' % (error_l2))

    # file_name = '09291027.plt'
    # post_process(file_name, u_mean.reshape(100,273))

    
    lambda1 = []
    for i in range(1000):
        temp = float(lambda_1_set[i])
        lambda1.append(temp)
    index1 = np.linspace(0,1000,1000,endpoint=False)
    plt.figure(2,figsize = (4,3))
    plt.plot(index1,lambda1)
    plt.xlabel('epoch')
    plt.ylabel('α')
    plt.show()
    # plt.savefig('2.pdf', bbox_inches='tight')
    plt.figure(1,figsize = (4,3))
    plt.plot(index1[800:1000],lambda1[800:1000])
    plt.xlabel('epoch')
    plt.ylabel('α')
    plt.show()
    # plt.savefig('2-jbfd.pdf', bbox_inches='tight')
    
    lambda1_mean0 = np.mean(lambda1[800:1000])
    lambda1_std = np.std(lambda1[800:1000])
    
    error_lambda_1 = (np.abs(lambda1_mean0 - a)/a)*100

    print('Error: %.8f%%' % (error_lambda_1)) 
    
    
    plt.figure(figsize = (4,5)) 
    #ax1 = plt.subplot(1, 1, 1)
    lamda_1_x = norm.rvs(loc= lambda1_mean0, scale=lambda1_std, size=1000, random_state=None)
    plt.hist(lamda_1_x,bins=30,edgecolor = 'black')
    plt.xlabel('λ1')
    plt.ylabel('frequency')
    plt.title('μ = %.3e   σ = %.3e' %(lambda1_mean0, lambda1_std))
    plt.show()    
    # plt.savefig('1000-0.01.pdf', bbox_inches='tight')
    

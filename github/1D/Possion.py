# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:44:19 2021
有内热源的一维平壁导热
fai = 1      内热源
h = 1        换热系数
lamda = 1    导热系数
delta = 1    平壁厚度的一半
tf = 0       环境温度
u_tt/u_xx + fai/lamda = 0

t = fai/(2*lamda)*(delta**2 - x**2) + fai/h * delta + tf
@author: Administrator
"""

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from HEBNN import HEBNN
import time
import tqdm 
#from sklearn.metrics import r2_score

def fx(x):
    y = -0.5*x**2 + 3/2
    return y

if __name__ == "__main__": 
    x = np.linspace(0,1,21,True)
    u_exact = fx(x)
    
    noise = 0.001
    layers = [1,10,10,10,1]
    lb = 0
    ub = 1.0
    x_star = x.reshape(-1,1)
    x_u_train = np.array([0,1.0]).reshape(-1,1)
    u_train = (np.array([u_exact[0], u_exact[-1]]) ).reshape(-1,1)#+ noise * np.random.randn(2)
    x_f_train = np.random.uniform(size = 10).reshape(-1,1)
    
    model = HEBNN(x_u_train, u_train, x_f_train, layers, lb, ub, noise)
    start_time = time.time()            
    model.train(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred_list = []
    f_pred_list = []   
    
    for i in tqdm.tqdm(range(500)):                                                 # tqdm进度条模块,n个模型
        u_pred= model.predict(x_star)
        u_pred_list.append(u_pred)       
    
    u_preds = np.concatenate(u_pred_list, axis=1)
    
    u_mean = (np.mean(u_preds, axis=1))
    u_sigma = (np.std(u_preds, axis=1))
    
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('T')
    plt.plot(x,u_exact,linewidth = 3,label = "Truth")
    plt.plot(x,u_mean, linewidth = 3, linestyle = "--",label = "Predicted mean", color = "red")
    plt.fill_between(x.ravel(), 
                     u_mean + 3 * u_sigma, 
                     u_mean - 3 * u_sigma, 
                     alpha=0.5, label='Epistemic uncertainty ±3σ',color = 'orange')
    plt.legend()
    #plt.savefig('0.001-50000-1.pdf', bbox_inches='tight')
    plt.show()
    
    plt.figure(2)
    plt.xlabel('x')
    plt.ylabel('ΔT')
    plt.plot(x, u_exact - u_mean, linewidth = 3, color = "red")
    #plt.savefig('0.001-50000-1-e.pdf', bbox_inches='tight')
    plt.show()

   # r2 = r2_score(u_exact,u_mean)

















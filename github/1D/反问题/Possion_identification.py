# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:56:48 2021

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from HEBNN import HEBNN
import time
import tqdm 
#from scipy.stats import norm
#from sklearn.metrics import r2_score

def fx(x):
    y = -0.5*x**2 + 3/2
    return y

if __name__ == "__main__": 
    x = np.linspace(0,1,11,True)
    u_exact = fx(x)
    
    noise = 0.001
    layers = [1,10,10,10,1]
    lb = 0
    ub = 1
    x_star = x.reshape(-1,1)
    x_u_train = x.reshape(-1,1)
    u_train = u_exact.reshape(-1,1) #+ noise * np.random.randn(41,1)
    
    model = HEBNN(x_u_train, u_train, layers, lb, ub)
    start_time = time.time()
    model.train(50000)                                                           # 合理情况下，训练次数越多，loss越低，不确定性越小
    elapsed = time.time() - start_time
                
    print('Training time: %.4f' % (elapsed))
    
    
    lambda_1_set = model.lambda_1_set
    # error_lambda_1 = np.abs(lambda_1_value_mean - 1.0)*100
    
    u_pred_list = []
    f_pred_list = []
    n_test = 101
    x_test = np.linspace(0, 1, n_test).reshape(-1, 1) 
    for i in tqdm.tqdm(range(500)):                                              # tqdm进度条模块,n个模型
        u_pred= model.predict(x_test)
        u_pred_list.append(u_pred)       
    
    u_preds = np.concatenate(u_pred_list, axis=1)

    u_mean = (np.mean(u_preds, axis=1))
    u_sigma = (np.std(u_preds, axis=1))
    
    
    lambda1 = []
    for i in range(500):
        temp = float(lambda_1_set[i])
        lambda1.append(temp)
    index1 = np.linspace(0,500,500,endpoint=False)
    plt.figure(2,figsize = (4,3))
    plt.xlabel('epoch')
    plt.ylabel('g·/k')
    plt.plot(index1,lambda1)
    plt.show()
    #plt.savefig('lambda1-100pde.pdf', bbox_inches='tight')
    
    lambda1_mean0 = np.mean(lambda1[400:500])
    lambda1_std = np.std(lambda1[400:500])
    error_lambda_1 = (np.abs(lambda1_mean0 - 1)/1)*100
    
    plt.figure(1,figsize = (4,3))
    plt.plot(x,u_exact,label = "Truth", color ="blue",linewidth = 3)
    plt.plot(x_test,u_mean,label = "Predicted mean", color ="red",linewidth = 3, linestyle="--")
    plt.xlabel('x')
    plt.ylabel('T')
    plt.legend()
    plt.show()
    #plt.savefig('lambda.pdf', bbox_inches='tight')
    
    #r2 = r2_score(fx(x_test),u_mean)
    
    
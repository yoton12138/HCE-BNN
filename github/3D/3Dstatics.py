# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:44:19 2021

三维稳态散热板算例

@author: Administrator
"""

import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from HEBNN import HEBNN
import time
import tqdm 
from sklearn.metrics import r2_score, mean_squared_error

def boundary(index,X_star,u_star):
    x = []
    u = []
    for i in index:
        for j in i:           
            x.append(X_star[j,:])
            u.append(u_star[j])
    return np.squeeze(np.array(x)), np.squeeze(np.array(u)).reshape(-1, 1) 

np.random.seed(123)
        
if __name__ == "__main__": 
    
    k = 15       #导热系数
    c = 380      #比热容
    ρ = 3200     #密度
    a = k/(c*ρ)  #热扩散率
    noise = 0.01
    N_f = 5000  #配置点数目
    N_u = 20    #观测点数目
    layers = [3,20,20,20,1]

    
    ExtractedData = np.load("ExtractedData.npy",allow_pickle=True).item()
    collocations = np.load("collocations.npy",allow_pickle=True)[:,1:]
    
    X_star = ExtractedData["Nodes"][:,1:]
    u_exact = ExtractedData["Temperature"][:,1:]
    
    lb = X_star.min(0)
    ub = X_star.max(0) 
    #*************边界条件***************
    index = np.array(np.where( (X_star[:,1] < 0.000001) )) #下边界 
    xxx1, uuu1  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,0] < -0.04999) )) #左边界 
    xxx2, uuu2  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,0] >  0.04999) )) #右边界 
    xxx3, uuu3  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,1] >  0.07999) )) #上边界 ， 绝热
    xxx4, uuu4  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,2] >  0.01999) )) #前边界 ， 绝热
    xxx5, uuu5  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,2] <  0.00001) )) #后边界 ， 绝热
    xxx6, uuu6  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,0] > -0.0301) & (X_star[:,0] < -0.0099) & (X_star[:,1] >  0.01999))) #内表面左 ， 绝热
    xxx7, uuu7  = boundary(index,X_star,u_exact)
    index = np.array(np.where( (X_star[:,0] >  0.00999) & (X_star[:,0] < 0.030001)& (X_star[:,1] > 0.01999))) #内表面右 ， 绝热
    xxx8, uuu8  = boundary(index,X_star,u_exact)
    
    
    X_u_train_total = np.vstack([xxx1, xxx2, xxx3, xxx4, xxx5, xxx6, xxx7, xxx8])#带重复点的所有边界数据
    u_train_total = np.vstack([uuu1, uuu2, uuu3, uuu4, uuu5, uuu6, uuu7, uuu8])
    X_U_data= np.hstack([X_u_train_total,u_train_total])
    X_U_data = pd.DataFrame(X_U_data)
    X_U_data.drop_duplicates(subset = [0,1,2],keep = "first", inplace=True)
    X_U_data = np.array(X_U_data)#转为DataFrame去重复的坐标点,0,1,2代表数组前三列
    #边界随机采点
    idx_u = np.random.choice(X_U_data.shape[0], N_u, replace=False)#在样本池中不放回抽样（N_u）个
    X_u_train = X_U_data[idx_u,0:3]
    u_train = X_U_data[idx_u,3:4]
    #指定点
    #idx_star = np.array([521,939,365,388,396,439,447,498,490,293,314,542,748,323,344,902,911,878,905,676,706,702,691])
    # idx_20 = np.array([388,396,439,447,498,490,293,314,542,748,323,344,902,911,878,905,676,706,702,691])
  
    # # idx_10 = np.array([902,911,878,905,447,439,676,706,702,691])#只用十个点均布试试
    # X_u_train = X_star[idx_20,0:3]
    # u_train = u_exact[idx_20,:]
    
    
    idx_f = np.random.choice(collocations.shape[0], N_f, replace=False)#在样本池中不放回抽样（N_f）个
    X_f_train = collocations[idx_f,:]

    
    #*******可视化观测点坐标********
    # fig = plt.figure()
    # ax = Axes3D(fig)    
    # ax.scatter(X_star[:,0], X_star[:,1], X_star[:,2], marker="o",color = 'yellow')
    # ax.scatter(X_u_train[:,0], X_u_train[:,1], X_u_train[:,2], marker="o", color = 'black')

    
    
    #**归一化
    u_max = max(u_train_total)[0]
    u_min = min(u_train_total)[0]
    for i in range(u_train.shape[0]):
        u_train[i] = (u_train[i] - u_min)/(u_max - u_min)
    
    #**训练计算
    model = HEBNN(X_u_train, u_train, X_f_train, layers, lb, ub, a,noise)
    start_time = time.time()            
    model.train(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    u_pred_list = [] 
    for i in tqdm.tqdm(range(500)):
        u_pred= model.predict(X_star)
        for i in range(u_pred.shape[0]):
            u_pred[i] = u_pred[i]*(u_max - u_min) + u_min
        u_pred_list.append(u_pred)       
    
    u_preds = np.concatenate(u_pred_list, axis=1)
    
    u_mean = (np.mean(u_preds, axis=1))
    u_sigma = (np.std(u_preds, axis=1))
    

    r2 = r2_score(u_exact,u_mean)
    print('r2: %e' % (r2))
    error_l2 = np.linalg.norm(u_exact-u_mean,2)/np.linalg.norm(u_exact,2)#还是相对l2误差比较好
    print('Error l2: %e' % (error_l2))
    rmse = mean_squared_error(u_exact,u_mean)
    print('rmse: %e' % (rmse))
    
    
#***************写入tecplot可视化验证*****************
def post_process(file_name, temperature, t=1):#四面体三维图形单元选择：Tetrahedron，不同单元需要不同格式关键字
    f = open(file_name,'w') 
    f.write('TITLE = "data"\n')
    f.write('VARIABLES =, "X", "Y","Z" "T"\n ')
    T = temperature
    Nodes_gcoord = ExtractedData["Nodes"][:,1:]
    Elements_nodes = ExtractedData["Elements"][:,1:]
    nnode = 1603
    nel = 6488
    for i in range(t):
        f.write('ZONE T = "%4d", N = %8d, E = %8d, ET = Tetrahedron, F = FEPOINT\n' %(i , nnode, nel))#需要更改**
        Ti = T
        for j in range(nnode):
            f.write('%10.10f, %10.10f, %10.10f, %10.10f\n' %(Nodes_gcoord[j,0], Nodes_gcoord[j,1], Nodes_gcoord[j,2],Ti[j]))
        for k in range(nel):
            for d in range (Elements_nodes.shape[1]):
                f.write('%d  '% Elements_nodes[k,d])
            f.write('\n')
    
    f.close()

# FileName_plt = "3d_statics_HEBNN_55.plt"
# temperature = u_mean
# post_process(FileName_plt,temperature)


line_data = np.hstack([u_exact,u_mean.reshape(-1,1)])
line_data = line_data[np.argsort(line_data[:,0])]
xn = np.linspace(0, line_data.shape[0]-1,num=line_data.shape[0])
plt.scatter(xn, line_data[:,1], label= "Mean prediction",s=1)
plt.plot(xn, line_data[:,0], 'r-', label='Truth')
plt.fill_between(xn.ravel(), 
                  line_data[:,1] + 3 * u_sigma, 
                  line_data[:,1] - 3 * u_sigma, 
                  alpha=0.2, label='Epistemic uncertainty ±3σ',color = 'orange')
plt.legend(loc = 4)
plt.show()












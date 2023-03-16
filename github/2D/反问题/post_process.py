# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:02:53 2020
python 后处理成.plt文件
@author: USER
"""
import scipy.io
def post_process(file_name, u_pred ):
    f = open(file_name,'w') 
    f.write('TITLE = "data"\n')
    f.write('VARIABLES =, "X", "Y" , "S"\n ')
    T = u_pred
    #T = scipy.io.loadmat('u_pred_08271202.mat')
    #T = (T['temperature']).reshape(100,273)
    gcoord = scipy.io.loadmat('data.mat')['gcoord']
    nodes = scipy.io.loadmat('nodes.mat')['nodes']
    nnode = 273
    nel =480
    for i in range(100):
        f.write('ZONE T = "%4d", N = %8d, E = %8d, ET = TRIANGLE, F = FEPOINT\n' %(i , nnode, nel))
        Ti = T[i,:]
        for j in range(nnode):
            f.write('%10.10f, %10.10f, %10.10f\n' %(gcoord[j,0], gcoord[j,1], Ti[j]))
        for k in range(nel):
            for d in range (3):
                f.write('%d  '% nodes[k,d])
            f.write('\n')
    
    f.close()
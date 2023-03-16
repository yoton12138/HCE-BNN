# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 09:50:33 2020

@author: USER
"""
import tensorflow as tf
import numpy as np
import time

np.random.seed(1234)
tf.set_random_seed(1234)

tf.reset_default_graph()

class PIBNN_tf:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, a, noise):
        
        self.norm = 1/self.num_w(layers) #加权因子
        self.lb = lb
        self.ub = ub
    
        self.x_u = X_u[:,0:1]
        self.y_u = X_u[:,1:2]
        self.t_u = X_u[:,2:3]
        
        self.x_f = X_f[:,0:1]
        self.y_f = X_f[:,1:2]
        self.t_f = X_f[:,2:3]
        
        self.u = u
        #print(self.y_u.shape[1])
        
        self.layers = layers
        self.a = a
        self.noise = noise
        self.layers = layers
        self.prior_sigma = 1.                                                   # 高斯先验 
        self.weight_1 = 1.
        self.weight_2 = 1.
        self.weight_decay = 1.
#        self.batchsize_obs = 256
#        self.batchsize_f = 512
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.saver = tf.train.Saver(max_to_keep=1)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
#        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, 1])
#        self.y_u_tf = tf.placeholder(tf.float32, shape=[None, 1])
#        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, 1])        
#        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
#        
#        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
#        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
#        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])    
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.y_u_tf = tf.placeholder(tf.float32, shape=[None, self.y_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, self.y_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]]) 
                
        self.u_pred = self.net_u(self.x_u_tf, self.y_u_tf, self.t_u_tf) #分别给数据，初始及边界条件
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf, self.t_f_tf) #配置点  
        self.b_pred_x, self.b_pred_y = self.net_b(self.x_f_tf, self.y_f_tf, self.t_f_tf)
        
        self.NLL = self.weight_1*(self.norm *self.neg_log_likelihood(self.u_tf,self.u_pred))               # 负对数似然损失，
        tf.add_to_collection('loss', self.NLL )
        self.PDE = self.weight_2*tf.reduce_mean(tf.square(self.f_pred))                  # PDE约束f_pred要趋近0
        tf.add_to_collection('loss', self.PDE )
##                
        self.loss = tf.add_n(tf.get_collection('loss'))
               
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.99, beta2=0.999, epsilon=1e-8,use_locking=False, name='Adam')
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

                
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W_rho = tf.Variable(tf.random_normal([layers[l],layers[l+1]], stddev=self.prior_sigma), dtype=tf.float32,name = 'W_rho')  #  使用了某一高斯作为权值先验分布
            W_mu = tf.Variable(tf.zeros([layers[l],layers[l+1]], dtype=tf.float32), dtype=tf.float32, name = 'W_mu')              #  为了参数化σ 再初始化一个ρ
            b_rho = tf.Variable(tf.random_normal([1,layers[l+1]], stddev=self.prior_sigma), dtype=tf.float32, name = 'b_rho')
            b_mu = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32, name = 'b_mu')
            
            W_sigma = tf.nn.softplus(W_rho)                                      # σ = log(1+exp(ρ)) 保证数值稳定性，一直为正。   σ 是 一个核函数对角矩阵，代表每个权值之间的分布都是不相关的
            W = W_mu +W_sigma* tf.random_normal(W_mu.shape)                      # 重采样
            b_sigma = tf.nn.softplus(b_rho)
            b = b_mu + b_sigma*tf.random_normal(b_mu.shape)                      # 这里表明了网络学习到的 W_mu,W_sigma,b_mu,b_sigma四个参数，
                                                                                 # 比普通神经网络多两个参数，具体的权值是一个分布，它由标准正态分布重采样构成           
            weights.append(W)
            biases.append(b)  
            
            KL_loss =self.weight_1*(self.norm *(self.KL_divergence(W, W_mu, W_sigma) + self.KL_divergence(b, b_mu, b_sigma)))  #计算KL-divergence 这是模型复杂度损失
            tf.add_to_collection('loss', KL_loss )      
            
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def obs_batch(self, batchsize_obs):
    
        idx = np.arange(0 , len(self.x_u))
        np.random.shuffle(idx)
        idx = idx[:batchsize_obs]   
        x_batch = [self.x_u[i] for i in idx]
        y_batch = [self.y_u[i] for i in idx]
        t_batch = [self.t_u[i] for i in idx]
        u_batch = [self.u[i] for i in idx]

        return np.asarray(x_batch), np.asarray(y_batch), np.asarray(t_batch), np.asarray(u_batch)

    def f_batch(self, batchsize_f):
        
        idx = np.arange(0 , len(self.x_f))
        np.random.shuffle(idx)
        idx = idx[:batchsize_f]   
        x_batch = [self.x_f[i] for i in idx]
        y_batch = [self.y_f[i] for i in idx]
        t_batch = [self.t_f[i] for i in idx]  
        
        return np.asarray(x_batch), np.asarray(y_batch), np.asarray(t_batch)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0  #变量转化到-1，到1之间
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x,y,t):
        u = self.neural_net(tf.concat([x,y,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x,y,t):
        u = self.net_u(x,y,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0] 
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t - self.a*(u_xx+u_yy)     #二维非稳态无内热源传热偏微分方程 趋于0
        
        return f
    
    def net_b(self,x,y,t):
        u = self.net_u(x,y,t)
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_yy = tf.gradients(u_y, y)[0] 
        u_xx = tf.gradients(u_x, x)[0]
        return u_x,u_y
        
        
    def KL_divergence(self, W, mu, sigma):
        variational_dist = tf.contrib.distributions.Normal(mu, sigma)            # 变分分布θ，稍后计算log(q(w|θ))
        return tf.reduce_sum(variational_dist.log_prob(W) - self.log_prior_prob(W))  # 期望部分用蒙特卡洛逼近，就是在(q(w|θ))采样的加权平均
    
    def log_prior_prob(self, W):                                                 # 对数先验（给定高斯分布下）权值为 W 的概率 log p(w)
        comp_1_dist = tf.contrib.distributions.Normal(0.0, self.prior_sigma)

        return comp_1_dist.log_prob(W)
    
    def neg_log_likelihood(self,y_obs,y_pred):
        dist = tf.contrib.distributions.Normal(loc=y_pred, scale= self.noise)           #  计算 -log p(D|w)
        return tf.reduce_sum(-dist.log_prob(y_obs))
    
    def num_w(self, layers):
        num = 0
        for i in range(len(layers)-1):
            n = (layers[i]+1)*layers[i+1]
            num =num + n
            i = i+1
        return num
    
    def callback(self, loss):
        print('Loss:', loss)
        
#    def train(self,nIter,epoch):
#        for i in range(epoch + 1):
#            x_u, y_u, t_u, u = self.obs_batch(batchsize_obs = self.batchsize_obs)
#            x_f, y_f, t_f = self.f_batch(batchsize_f = self.batchsize_f)
#            
#            tf_dict = {self.x_u_tf: x_u, self.y_u_tf: y_u, self.t_u_tf: t_u, self.u_tf: u,
#                       self.x_f_tf: x_f, self.y_f_tf: y_f, self.t_f_tf: t_f}
#        
#            start_time = time.time()                                                                                                                  
#            for it in range(nIter):
#                self.sess.run(self.train_op_Adam, tf_dict)
#                
#
#            elapsed = time.time() - start_time
#            loss_value = self.sess.run(self.loss, tf_dict)
#            print('epoch: %d, Loss: %.3e, Time: %.2f' % 
#                  (i, loss_value, elapsed))
#            #print(self.weigh_2)
#            start_time = time.time()
#        
    def train(self,nIter):
        
        tf_dict = {self.x_u_tf: self.x_u, self.y_u_tf: self.y_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.y_f_tf: self.y_f, self.t_f_tf: self.t_f}
         
        start_time = time.time()                                                                                                                  
        for it in range(nIter+1):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it > (1/3*nIter):                
                self.weight_1 = self.weight_decay*self.weight_1                        #weight decay  如何设计保证观测数据能良好的限制模型以后再开始，这样保证稳健性也是个问题
            
            if it % 5000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                #print(self.weigh_2)
                start_time = time.time()
    
    def predict(self, X_star):
                
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.y_u_tf: X_star[:,1:2], self.t_u_tf: X_star[:,2:3]})  
               
        return u_star
    
    def predict_b(self, X_star):        
        u_x = self.sess.run(self.b_pred_x, {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], self.t_f_tf: X_star[:,2:3]})
        u_y = self.sess.run(self.b_pred_x, {self.x_f_tf: X_star[:,0:1], self.y_f_tf: X_star[:,1:2], self.t_f_tf: X_star[:,2:3]})
        return u_x,u_y
    
    def save(self, path):
        saver = self.saver
        Sess = self.sess
        saver.save(Sess, path)
        
    def load(self, path):
        Sess = self.sess
        saver = self.saver
        saver.restore(Sess, path)
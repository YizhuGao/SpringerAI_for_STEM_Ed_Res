# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:14:35 2024

@author: 欣宝
"""


from __future__ import print_function
from __future__ import division
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


N_NOTES = 7
N_KP = 7

np.random.seed(4178)
exam_weight0 = np.random.randint(1,6,size=N_KP)
exam_weight1 = exam_weight0/exam_weight0.sum() 
exam_level = np.random.uniform(0.5,1,N_KP) 
exam_weight1 = np.ones([N_KP])/N_KP        # correspond to r
#exam_weight2 = np.ones([N_KP])/N_KP        # correspond to r2, change this weight to see different result. 




weight_mat1 = np.zeros((N_NOTES, N_KP))
weight_mat1[0] = np.array([ 0.7,  0.3,  0.,  0.,  0.,  0.,  0.])
weight_mat1[1] = np.array([ 0.7,  0.,  0.3,  0.,  0.,  0.,  0.])
weight_mat1[2] = np.array([ 0.3,  0.6,  0.0,  0.,  0.1,  0.,  0.])
weight_mat1[3] = np.array([ 0.2,  0.4,  0.4,  0.,  0.,  0.,  0.])
weight_mat1[4] = np.array([  0.0,  0.,  0.,  1.0,  0.,  0.,  0.])
weight_mat1[5] = np.array([ 0.,  0.2,  0.,  0.,  0.8,   0.,  0.])
weight_mat1[6] = np.array([  0.,  0.,  0.,  0.1,  0.,  0.4,  0.5])

state0 = np.zeros(N_KP)  # initial state
state0 = state0.tolist()

class Learning():
    def __init__(self):
        np.random.seed(4178)

        self.action_space = [str(x+1) for x in range(N_NOTES)] + ['stop']
        self.n_actions = len(self.action_space)
        self.origin = state0
        self.n_states = len(self.origin)
        self.transition = weight_mat1
        self.weight = exam_weight1
        #self.weight2 = exam_weight2
        self.diff = exam_level
        self.number = 2
        self.guess = 0.25   
        self.eps = np.finfo(np.float32).eps
        self.model_name = 'continuous'

    
    def qualify(self,s):
        Y = np.zeros(N_KP)
        Y[0] = 1
        if s[0]>=0.3:
            Y[1] = 1
        if s[0]>=0.2:
            Y[2] = 1
        if s[0]>=0.9:
            Y[3] = 1
        if s[1]>=0.9:
            Y[4] = 1
        if s[3]>=0.3:
            Y[5] = 1
        if s[3]>=0.3:
            Y[6] = 1
        return Y  
    
    def step(self,s,a):
        '''
        arguments:
        s: current state in [0,1]^{N_KP}
        a: current action (scalar)
        return:
        s_: next state (list)
        r: reward (scalar)
        done: stop or not (bool)
        '''
        s = np.array(s)
        done = False
        if a != self.n_actions - 1:
            y = -np.log(1-s)
            increase = np.random.chisquare(2)*self.transition[a]
            increase = increase*self.qualify(s)
                             
            y = y + increase
            s_ = 1 - np.exp(-y)    
        else:
            s_ = s
        if sum(s_> 1.0) == N_KP or a == self.n_actions - 1:           # do not allow stop in the curiosity-driven reward.
            done = True
            s_ = s
        r = np.dot(s_,self.weight)
        #r2 = np.dot(s_,self.weight2)

        s_ = s_.tolist()
        return s_, r, done
                
    def reset(self):
        return self.origin
    
    
    def transfer(self, s):
        t = s.copy()
        for i in range(len(s)):
            if s[i] == 0:
                s[i] = self.eps
            elif s[i] == 1:
                s[i] = 1 - self.eps
    
            t[i] = -np.log(1-s[i]) + np.log(s[i])
        return t    
    
    def logistic(self,t):
        ss = t.copy() 
        for i in range(len(t)):
            ss[i] = 1/(1+np.exp(-t[i]))
        return ss
    
    def results(self,s_,a):
        w =np.zeros((self.number,self.n_states))
        for i in range(self.number):
            if np.any(weight_mat1[a] < 0) or np.any(weight_mat1[a] > 1):
                raise ValueError("weight_mat1[a] 包含无效值")
                print(weight_mat1[a])
                print(a)
            if N_KP <= 0:
                raise ValueError("N_KP 必须是正整数")
                print(N_KP)
            w[i] = np.random.binomial(N_KP,weight_mat1[a],size = self.n_states)
#            w[i] = np.random.multinomial(1,weight_mat1[a],size = 1)
        row_sum = w.sum(axis=1,dtype =float)
        
        if np.any(row_sum == 0):
            print("行和为0，无法进行归一化,因此设为等概率")
            w[i] = np.ones(self.n_states) / self.n_states
        else:
            w = w / row_sum[:, np.newaxis]
        
        
       # w = w/row_sum[:,np.newaxis]   
        #print("w values", w)
        
        if np.any(np.isnan(w)):
            print("发现 w 包含 NaN:")
            print("weight_mat1[a]:", weight_mat1[a])
            print("a:", a)
            print("N_KP:", N_KP)

        b = np.random.uniform(-5., 5.,self.number)
        level = np.dot(w,s_)
        
        #print("level values:", level)
        #print("b values:", b)
        exp = 1 + np.exp(np.clip(-level + b, a_min=None, a_max=700))
        #exp = 1 + np.exp((-level + b))
        
        #exp = np.maximum(exp, 1e-10)
        exp = np.clip(exp, a_min=1e-10, a_max=None)
        
        if self.guess < 0 or self.guess > 1:
            raise ValueError("self.guess is out of range [0, 1]")
        
        #print("Exp values:", exp)
        
        prob = self.guess + (1-self.guess)/exp
        
        #print("Prob values before clipping:", prob)
        
        prob = np.clip(prob, 0, 1)  # 将 prob 的值限制在 0 和 1 之间
        if np.any(np.isnan(prob)):
            raise ValueError("概率值包含 NaN")
        
        outcomes = np.random.binomial(1,prob,size = self.number)
        return outcomes,b,w
    
    
    def neg_loglik(self,theta,outcomes, b, w):
        level = np.dot(w,theta)
        exp = 1 + np.exp((-level + b))
        prob = self.guess + (1-self.guess)/exp
        p = np.log(prob)
        q = np.log(1 - prob)
        negloglik = -np.sum(p*outcomes) - np.sum((q)*(1-outcomes))
        return negloglik
 
    
    def estpara(self,s,a,s_):
        t = self.transfer(s.copy())
        t_ = self.transfer(s_.copy())
        outcomes, b, w = self.results(t_,a)
        initParams = t
        bnds = ()
        initial = list((np.array(t)+np.array(t_))/2)
        for i in range(self.n_states):
            bnds = bnds + ((initParams[i],1/self.eps),)
           # bnds = bnds + ((0,1),)       
  #      estimation = minimize(self.neg_loglik, initial, method='trust-constr', bounds=bnds,args=(outcomes, b, w))
        
        estimation = minimize(self.neg_loglik, initial,bounds=bnds,args=(outcomes, b, w))

        
        return self.logistic(estimation.x).tolist()
    


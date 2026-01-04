# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:15:32 2024

@author: 欣宝
"""


import torch
import sys
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
from continuous_2 import Learning

from torch.autograd import Variable
import pandas as pd
import numpy as np
from utils import v_wrap, push_and_pull, record, SharedAdam
import torch.nn.functional as F
import torch.multiprocessing as mp
from filelock import Timeout, FileLock
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')



os.environ["OMP_NUM_THREADS"] = "1" 

UPDATE_GLOBAL_ITER = 100             # Time to update the global parameter
GAMMA = 0.9999                       #  Discount rate (future reward is discounted), note that (0.9999)**40 = 0.996
MAX_EP = 40000                # Training episode: here we need several steps to fill the memory pool
scale = 100                          # rescale reward, this parameter does not influence the result.
env = Learning()
N_S = env.n_states
N_A = env.n_actions-1                


LR_1 = float(sys.argv[1])            # Learning rate of A3C network.
#LR_2 = float(sys.argv[2])            # Learning rate of Predictive network.
est_state = eval(sys.argv[2])        # True: assessment error exists. False: Perfect assessment.
number = int(sys.argv[3])            # The number of item used to estimate.
horizon = int(sys.argv[4])           # The length of learning period. In the continuous case 2, we choose horizon = 40.
cpu_number = int(sys.argv[5])        # cpu number.


env.number = number
torch.manual_seed(12345)
np.random.seed(4178)

class Net(nn.Module):                                    # Actor-Critic network
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        torch.manual_seed(12345)

        self.s_dim = s_dim #40
        self.a_dim = a_dim #22
        self.pi1 = nn.Linear(s_dim, 10)
        self.pi2 = nn.Linear(10, 10)
        self.pi3 = nn.Linear(10, 10)
        self.pi4 = nn.Linear(10, a_dim)
        
        self.v1 = nn.Linear(s_dim, 10)
        self.v2 = nn.Linear(10, 10)
        self.v3 = nn.Linear(10, 10)
        self.v4 = nn.Linear(10, 1)
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.relu(self.pi1(x))     #激活函数
        pi2 = torch.relu(self.pi2(pi1))
        pi3 = torch.relu(self.pi3(pi2))

        logits = self.pi4(pi3)    #policy #输入某一时刻的state，输出action被采纳的概率
        
        v1 = torch.relu(self.v1(x))
        v2 = torch.relu(self.v2(v1))
        v3 = torch.relu(self.v3(v2))

        values = self.v4(v3)      #Q function 
        
        return torch.exp(F.log_softmax(logits, dim= -1)), values #softmax归一化
#输出的是概率和Q

    def choose_action(self, s):
        self.eval()
        logits, values = self.forward(s)
        prob = logits 
        prob = torch.relu(prob)/(torch.sum(torch.relu(prob))) #Policy的返回结果，在状态x下各个action被执行的概率
        m = self.distribution(prob) #生成分布
        return m.sample().numpy() # 从分布中采样（根据各个action的概率）



    def loss_func(self, s, a, v_t):
        
        self.train()
        logits, values = self.forward(s)
        td = v_t - values #
        c_loss = td**2
        
        
        probs = logits
        m = self.distribution(probs) #生成分布
    
        #entropy = m.entropy()
        #r_t = torch.exp(m.log_prob(a))/old
        exp_v = m.log_prob(a) * td.detach().squeeze() #torch.min(r_t * td.detach(), torch.clamp(r_t,0.8,1.2)*td.detach())+ 0.001* entropy
        
        a_loss = -exp_v
        total_loss = torch.mean(a_loss+c_loss)
        return total_loss

       
class Worker(mp.Process):
    def __init__(self, gnet, opt, scheduler_1,global_ep, global_ep_r, res_queue, res_queue_2, name):
        super(Worker, self).__init__()
        torch.manual_seed(12345)

        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue, self.res_queue_2 = global_ep, global_ep_r, res_queue, res_queue_2
        self.gnet, self.opt = gnet, opt     # global network
        self.lnet = Net(N_S, N_A)           # local network
        self.lnet.load_state_dict(self.gnet.state_dict())
        
        self.env = env
        self.censor = 0.0
        self.horizon = horizon 
        
        self.memory_counter = 0                                         # storing memory
        self.MEMORY_CAPACITY = 6000
        self.memory = np.zeros((self.MEMORY_CAPACITY, N_S * 2 + 1))     # initialize memory
        self.BATCH_SIZE = 64

        self.scheduler_1 = scheduler_1

        
    def store_transition(self, s, a, s_):
        transition = np.hstack((s, a, s_)) # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
       
    def run(self):
        TIME = self.horizon
        total_step = 1
        while self.g_ep.value < MAX_EP:
            with self.g_ep.get_lock():                
                tmp_date = self.g_ep.value

                self.g_ep.value += 1
                
            '''
                if tmp_date % 100 == 0:
                    self.g_target.load_state_dict(self.gnet.state_dict())  
            '''
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r, buffer_s_= [], [], [], []
            #store_s,store_a,store_r = [], [], []
            ep_r= 0.  #总奖励
            step = 0           
            cs = s 
            while True:
                self.scheduler_1.step()


                step += 1
                a = self.lnet.choose_action(v_wrap(cs))
                s_, r, done = self.env.step(s,a)          ### Imporant!!! The reward (r and r2) here is used for further analysis (draw figure)
                                                              
                if done or step == TIME:
                    r= (r*scale)
                    done = True
                else:
                    r = 0.

                if est_state:                                
                    cs_ = env.estpara(cs,a,s_) 
                else:
                    cs_ = s_ 
                    
                ep_r += r 
               
                #print("step",step,"action",a,"reward",r,"emotion",emoflag)                  
                buffer_a.append(a)
                buffer_s.append(cs)
                buffer_s_.append(cs_)

                buffer_r.append(r)
                
                self.store_transition(cs, a, cs_)


                    
                if done or step == TIME:  # update global and assign to local net
                    push_and_pull(self.opt, self.lnet, self.gnet, done, cs_, buffer_s, buffer_a, buffer_r,GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done or step == TIME:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r,self.res_queue, self.res_queue_2,self.name, step)
                        break
                s = s_
                cs = cs_
                total_step += 1
        
        self.res_queue.put(None)
        self.res_queue_2.put(None)

if __name__ == "__main__":
    torch.manual_seed(12345)
    np.random.seed(4178)
    

    gnet = Net(N_S, N_A)        # global network
        
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=LR_1,betas=(0.92, 0.999))      # global optimizer
        
    scheduler_1 = lr_scheduler.StepLR(opt, step_size= 200, gamma = 0.99999)
    
    global_ep, global_ep_r, res_queue, res_queue_2= mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, scheduler_1, global_ep, global_ep_r, res_queue, res_queue_2, i) for i in range(cpu_number)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot             mp.cpu_count()
   # res_2 = []
    while True:
        r = res_queue.get()
       # emo = res_queue_2.get()
        #if (r is not None) and (emo is not None):
        if r is not None:
            res.append(r)
           # res_2.append(emo)
        else:
            break
    [w.join() for w in workers]
    
    
    smooth_res=[]
    x = res[0]
    for i in range(len(res)):
        x =  0.99*x + 0.01*res[i]
        smooth_res.append(x)
     

    RES, SMOOTH_RES = [],[]
    
   # RES_2= []

    aver = 100
    I = int(len(res)/aver)

    for i in range(I):
        a = np.mean(res[i*aver:(i+1)*aver])
        b = np.mean(smooth_res[i*aver:(i+1)*aver])
        #a2 = np.mean(res_2[i*aver:(i+1)*aver])
               
            
        RES.append(a)
        SMOOTH_RES.append(b)
       # RES_2.append(a2)
       
        
    SMOOTH_train = pd.DataFrame(SMOOTH_RES)
    train = pd.DataFrame(RES)

   # train_2 = pd.DataFrame(RES_2)
    
    SMOOTH_train = pd.DataFrame.transpose(SMOOTH_train)
    train = pd.DataFrame.transpose(train)
    
  #  train_2 = pd.DataFrame.transpose(train_2) 
    
        
    # Save the result to the .csv file (without replacing original file)
    
    lock = FileLock('smooth_results_continuous_2_%s_%s_%s_c_v2.csv.lock'%(LR_1,est_state,number))
    with lock:
        SMOOTH_train.to_csv('smooth_results_continuous_2_%s_%s_%s_c_v2.csv'%(LR_1,est_state,number), mode = 'a', index=False, header=False)
        
    lock = FileLock('results_continuous_2_%s_%s_%s_%s_c_v2.csv.lock'%(LR_1,est_state,number,horizon))
    with lock:
        train.to_csv('results_continuous_2_%s_%s_%s_%s_c_v2.csv'%(LR_1,est_state,number,horizon), mode = 'a', index=False, header=False)
        
 
   
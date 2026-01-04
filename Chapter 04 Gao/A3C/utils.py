# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:13:44 2024

@author: 欣宝
"""



from torch import nn
import torch
import numpy as np
from torch.autograd import Variable


torch.manual_seed(12345)
np.random.seed(12345)
    
def v_wrap(np_array, dtype=np.float32):
    np_array = np.array(np_array)
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return Variable(torch.from_numpy(np_array))     

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]  
    buffer_v_target = []
    # calculate target
    for r in br[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()
    
    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))  
    
    opt.zero_grad()  
    loss.backward()  
   
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()  
    lnet.load_state_dict(gnet.state_dict())

def record(global_ep, global_ep_r, ep_r, res_queue, res_queue_2, name,step):
    '''
    with global_ep.get_lock():
        global_ep.value += 1
    '''
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(ep_r)         
     
    if global_ep.value%50==0:
        print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % ep_r,
        "| step: %.0f" % step, flush=True
        )


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

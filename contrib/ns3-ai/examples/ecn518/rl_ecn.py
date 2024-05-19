# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# Copyright (c) 2020 Huazhong University of Science and Technology, Dian Group
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#         Hao Yin <haoyin@uw.edu>

from py_interface import *
from ctypes import *
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
# parser.add_argument('--output_dir', type=str,
#                     default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')
parser.add_argument('--test_act', action='store_true',
                    help='whether use content action')
parser.add_argument('--save_model', action='store_true',
                    help='whether save model')



class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 390),#   action space 1 :  980
            #output is the all slection of action
        )
    def forward(self, x):
        return self.layers(x)


class DQN(object):
    def __init__(self):
        self.eval_net = net()
        self.target_net = net()
        self.learn_step = 0
        self.batchsize = 64

        self.observer_shape = 6

        self.target_update = 50
        self.memory_counter = 0
        self.memory_capacity = 2000
        self.memory = np.zeros((2000, 2*6+2))    # s, a, r, s'
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.HuberLoss()#nn.MSELoss()
        self.gamma = 0.8
        self.dqn_type = 'DDQN'

    def choose_action(self, x):
        x = torch.Tensor(x)
        if np.random.uniform() > 0.99 ** self.memory_counter:    # choose best
            action = self.eval_net.forward(x)
            action = torch.argmax(action, 0).numpy()
        else:    # explore
            #explore from all action
            action = np.random.randint(0, 390)
        return action

    def store_transition(self, s, a, r, s_):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = np.hstack((s, [a, r], s_))
        self.memory_counter += 1

    def learn(self, ):
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        sample_list = np.random.choice(self.memory_capacity, self.batchsize)
        # choose a mini batch
        sample = self.memory[sample_list, :]
        s = torch.Tensor(sample[:, :self.observer_shape])
        a = torch.LongTensor(
            sample[:, self.observer_shape:self.observer_shape+1])
        r = torch.Tensor(
            sample[:, self.observer_shape+1:self.observer_shape+2])
        s_ = torch.Tensor(sample[:, self.observer_shape+2:])
        q_eval = self.eval_net(s).gather(1, a)
        
        if self.dqn_type == 'DDQN':
            max_action = self.eval_net(s_).max(1)[1].view(-1,1)
            q_next_values = self.target_net(s_).gather(1,max_action)
        else:
            #max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            #max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
            q_next_values = self.target_net(s_).detach().max(1, True)[0].data
        q_target = r + self.gamma * q_next_values

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save(self):
        torch.save(self.target_net.state_dict(),'dqn.pt')
        #model = TheModelClass(args, *kwargs)
        #model.load_state_dict(torch.load(PATH))


def Action():
    """
    min_v =[1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    max_v = [128, 256, 512, 1024, 2048, 5120, 10240]
    p = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.25, 0.5, 0.75, 1]
    action_list = [[0] * 3] * 980
    # print(type(action_list))
    num = 0
    for i in range(len(min_v)):
        for j in range(len(max_v)):
            if min_v[i] < max_v[j]:
                for k in range(len(p)):
                    # print(min_v[i],max_v[j],p[k])
                    action_list[num] = min_v[i], max_v[j], p[k]
                    num += 1
            else:
                pass
    return action_list
    """
    kmin =[2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    kmax = [128, 256, 512, 1024, 2048, 5120, 10240]
    pmax = [0.01, 0.1, 0.25, 0.5, 0.75, 1]
    action_space = [[0] * 3] * 390
    num = 0
    for i in range(len(kmin)):
        for j in range(len(kmax)):
            if kmin[i] <= kmax[j]:
                for k in range(len(pmax)):
                    action_space[num] = kmin[i], kmax[j], pmax[k]
                    num += 1
            else:
                pass
    return action_space
    
class sEcnRlEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ('egressqlen', c_uint32),
        ('linkrate', c_double),
        ('ecnlinkrate', c_double),
        ('ecnmin', c_uint32),
        ('ecnmax', c_uint32),
        ('ecnpmax', c_double),
        ('envType', c_uint8),
        ('simTime_us', c_int64)
    ]

class EcnRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('newecnmin', c_uint32),
        ('newecnmax', c_uint32),
        ('newecnpmax', c_double)
    ]  
act_list = Action()
last_state = [0,0,0,0,0,0]
last_a =0

time_list = []
qlen_list = []
throughout = []

Init(1234, 4096)
var = Ns3AIRL(2330, sEcnRlEnv, EcnRlAct)
args = parser.parse_args()
if args.use_rl:
    dqn = DQN()
exp = Experiment(1234, 4096, 'test_ecn414 hpccmix/test_c.txt', '../../')
exp.run(show_output=0)
exp.reset()  
try:
    while not var.isFinish():
        with var as data:
            if not data:
                break
            #print(var.GetVersion())
            #基于读到的数据得到新的控制窗口和阈值，再放入共享缓存里
            egress_qlen = data.env.egressqlen
            link_rate = data.env.linkrate
            ecn_link_rate = data.env.ecnlinkrate
            ecn_min = data.env.ecnmin
            ecn_max = data.env.ecnmax
            ecn_pmax = data.env.ecnpmax
            sim_time_us = data.env.simTime_us

            qlen_list.append(egress_qlen)
            throughout.append(link_rate)
            time_list.append(sim_time_us/1000000000)

            
            if args.use_rl:
                s = [egress_qlen, link_rate,ecn_link_rate,ecn_min,ecn_max,ecn_pmax]
                a = dqn.choose_action(s)
                # print(a)

                new_ecn_min = act_list[a][0]
                new_ecn_max = act_list[a][1]
                new_ecn_pmax = act_list[a][2]
                
                #print(new_ecn_min,new_ecn_max,new_ecn_pmax)
                if args.test_act:
                    data.act.newecnmin = 40
                    data.act.newecnmax =1600
                    data.act.newecnpmax = 0.2
                else:
                    data.act.newecnmin = new_ecn_min
                    data.act.newecnmax = new_ecn_max
                    data.act.newecnpmax = new_ecn_pmax


                if egress_qlen>0:
                    Dl = (1-math.log2(egress_qlen / 2000)/math.log2(5120))
                    #print(Dl)
                else:
                    Dl = 1

                reward = Dl*7+link_rate*3
                # print(reward)

                if(sum(last_state) == 0):
                    #print("drop data")
                    pass
                else:
                    # print("memory",last_state,last_a,reward,s)
                    dqn.store_transition(last_state, last_a, reward, s)

                last_state[0] = egress_qlen
                last_state[1] = link_rate
                last_state[2] = ecn_link_rate
                last_state[3] = ecn_min
                last_state[4] = ecn_max
                last_state[5] = ecn_pmax 
                last_a = a


                if dqn.memory_counter > dqn.memory_capacity:
                    dqn.learn()

                    """
                    if(egress_qlen>20000):
                        Dl = (1-math.floor(math.log2(egress_qlen / 20000))/10)
                    else:
                        Dl =1
                    """            
except KeyboardInterrupt:
    exp.kill()
    del exp
if args.save_model:
    dqn.save()
if args.result:
    # plt.plot(time_list, qlen_list, linewidth=1, color='r')
    # plt.xlabel('Step Number')
    # timestamp = time.strftime('%Y_%m_%d_%H%M_%S') 
    # plt.savefig('qlen{}.png'.format(timestamp))
    # 创建一个包含两个子图的figure，1代表列数，2代表行数，返回一个包含子图的数组
    fig, axs = plt.subplots(1,2)

    # 在第一个子图上绘制第一个函数
    axs[0].plot(time_list, throughout)
    axs[0].set_title('port u')
    axs[0].set_xlabel('simulator time')
    axs[0].set_ylabel('port throughpout')

    # 在第二个子图上绘制第二个函数
    axs[1].plot(time_list, qlen_list)
    axs[1].set_title('egress qlen')
    axs[1].set_xlabel('simulator time')
    axs[1].set_ylabel('qlen')
    # 调整子图间距，确保图表之间不重叠
    plt.tight_layout()

    timestamp = time.strftime('%Y_%m_%d_%H%M_%S') 
    plt.savefig('result{}.png'.format(timestamp))


# if args.result:
#     for res in res_list:
#         y = globals()[res]
#         x = range(len(y))
#         plt.clf()
#         plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
#         plt.xlabel('Step Number')
#         plt.title('Information of {}'.format(res[:-2]))
#         plt.savefig('{}.png'.format(os.path.join(args.output_dir, res[:-2])))

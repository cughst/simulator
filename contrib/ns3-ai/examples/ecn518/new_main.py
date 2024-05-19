
from py_interface import *
from ctypes import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from Buffer import ReplayBuffer
from Agent import DQN
from tqdm import tqdm
import argparse
from Action_List import Action
import os

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
# parser.add_argument('--output_dir', type=str,
#                     default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')
parser.add_argument('--use_rl_action', action='store_true',
                    help='whether use rl action')
parser.add_argument('--save_model', action='store_true',
                    help='whether save model')


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_DQN(agent,var, num_episodes, replay_buffer, minimal_size,
              batch_size,last_state,last_a,args):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(500):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                with var as data:
                    if not data:
                        break
                    # print(var.GetVersion())
                    # 基于读到的数据得到新的控制窗口和阈值，再放入共享缓存里
                    egress_qlen = data.env.egressqlen
                    link_rate = data.env.linkrate
                    ecn_link_rate = data.env.ecnlinkrate
                    ecn_min = data.env.ecnmin
                    ecn_max = data.env.ecnmax
                    ecn_pmax = data.env.ecnpmax
                    sim_time_us = data.env.simTime_us

                    qlen_list.append(egress_qlen)
                    throughout.append(link_rate)
                    time_list.append(sim_time_us / 1000000000)

                    if args.use_rl_action:
                        s = [egress_qlen, link_rate, ecn_link_rate, ecn_min, ecn_max, ecn_pmax]
                        # print(a)
                        action = agent.take_action(s)
                        max_q_value = agent.max_q_value(
                            s) * 0.005 + max_q_value * 0.995  # 平滑处理
                        max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                        new_ecn_min = act_list[action][0]
                        new_ecn_max = act_list[action][1]
                        new_ecn_pmax = act_list[action][2]
                        # print(new_ecn_min,new_ecn_max,new_ecn_pmax)

                        data.act.newecnmin = new_ecn_min
                        data.act.newecnmax = new_ecn_max
                        data.act.newecnpmax = new_ecn_pmax

                        if egress_qlen > 0:
                            Dl = (1 - math.log2(egress_qlen / 2000) / math.log2(5120))
                            # print(Dl)
                        else:
                            Dl = 1
                        reward = Dl * 50 + link_rate * 50
                        # print(reward)
                        if (sum(last_state) == 0):
                            # print("drop data")
                            pass
                        else:
                            # print(last_state,last_a,reward,s)
                            replay_buffer.add(last_state,last_a,reward,s)
                        episode_return += reward

                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns = replay_buffer.sample(
                                batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r
                            }
                            agent.update(transition_dict)
                    else:
                        s = [egress_qlen, link_rate, ecn_link_rate, ecn_min, ecn_max, ecn_pmax]
                        # print(sim_time_us,s)
                        data.act.newecnmin = 40
                        data.act.newecnmax = 1600
                        data.act.newecnpmax = 0.2
                        if egress_qlen > 0:
                            Dl = (1 - math.log2(egress_qlen / 2000) / math.log2(5120))
                            # print(Dl)
                        else:
                            Dl = 1
                        reward = Dl * 50 + link_rate * 50
                        episode_return += reward
                    return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list

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
if __name__ == "__main__":
    lr = 1e-5
    num_episodes = 1000
    hidden_dim = 64
    gamma = 0.8
    epsilon = 0.01
    target_update = 50
    buffer_size = 50000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    print(device)

    act_list = Action()
    last_state = [0, 0, 0, 0, 0, 0]
    last_a = 0
    time_list = []
    qlen_list = []
    throughout = []
    Init(1234, 4096)
    var = Ns3AIRL(2330, sEcnRlEnv, EcnRlAct)
    state_dim = 6 # txrate qlen txrate_m min max pmax
    action_dim = 390  #
    args = parser.parse_args()

    if args.use_rl:
        agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device,dqn_type="DoubleDQN")
        # if os.path.exists("518_dqn.pt"):
        #     agent.load()
            
    exp = Experiment(1234, 4096, 'test_ecn414 hpccmix/test_c.txt', '../../')
    exp.run(show_output=0)
    exp.reset()

    replay_buffer = ReplayBuffer(buffer_size)

    return_list, max_q_value_list = train_DQN(agent,var, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size,last_state,last_a,args)
    if args.save_model:
        agent.save()
    if args.result:
        episodes_list = list(range(len(return_list)))
        mv_return = moving_average(return_list, 5)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN ')
        plt.show()
        #
        # frames_list = list(range(len(max_q_value_list)))
        # plt.plot(frames_list, max_q_value_list)
        # plt.xlabel('Frames')
        # plt.ylabel('Q value')
        # plt.title('DQN on ')
        # plt.show()
from pdb import set_trace
from gobigger.agents import BotAgent
from gobigger.server import Server
from gobigger.render import EnvRender
import sys

from tensorflow.python.framework.ops import to_raw_op
sys.path.append('..')
import torch
from envs import GoBiggerEnv
from model.gb import TorchRNNModel
from config.no_spatial import env_config

import glob
import pickle
import numpy as np
from torch.optim import Adam, RMSprop
from collections import namedtuple
from torch.nn import functional as F
from functools import partial
import multiprocessing
from multiprocessing import Pipe, Process
from tensorboardX import SummaryWriter
import datetime

bot_data_one_episode = namedtuple("bot_data_one_episode", ["obs", "action", "reward"])


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

model_config = {
    "custom_model": "go_bigger", 
    "lstm_cell_size": 128 ,
    "max_seq_len" : 10,
    "custom_model_config": {
        "obs_shape" : 50,
        "entity_shape" : 31,
        "obs_embedding_size" : 128,
        "entity_embedding_size" : 128,
        "all_embedding_size" : 128,
    }
}

class PPOBot():
    def __init__(self) -> None:
        self.num_actions = 16
        self.model = TorchRNNModel(None, None, self.num_actions, model_config, "PPOBot")
        # self.state = self.initial_state()
        # self.optmizer = RMSprop(self.model.trainable_variables(), 1e-5)
        self.optmizer = Adam(self.model.trainable_variables(), 1e-5)
        self.max_seq_len = 10 
        self.tblogger = SummaryWriter('./log/{}/'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        self.learn_iter = 0
        self.policy_criterion = torch.nn.CrossEntropyLoss()


    def initial_state(self, batch_size):
        hc = [self.model.get_initial_state() for _ in range(batch_size)]
        state = [torch.cat([each[0].unsqueeze(dim=0) for each in hc], dim=0),  torch.cat([each[1].unsqueeze(dim=0) for each in hc], dim=0)]
        return state

    def learn(self, obs, action, reward, state = None):
        bs, total_seq_len, entity_shape = *obs.shape,
        action = action.float()
        reward = reward.float()

        obs = torch.cat((obs, torch.zeros(bs, 1, entity_shape)), dim=1)
        

        if state == None:
            state = self.initial_state(bs)

        if cuda:
            state = [each.to(torch.device("cuda:0")) for each in state]
            obs = obs.to(torch.device("cuda:0"))
            action = action.to(torch.device("cuda:0"))
            reward = reward.to(torch.device("cuda:0"))

        for sl in range(int(total_seq_len / self.max_seq_len)):
            state = [each.detach() for each in state]
            obs_sl = obs[:, sl*self.max_seq_len : (sl+1)*self.max_seq_len+1]
            reward_sl = reward[:, sl*self.max_seq_len : (sl+1)*self.max_seq_len].squeeze(-1)
            action_sl = action[:, sl*self.max_seq_len:(sl+1)*self.max_seq_len]

            logits, state = self.model.forward_rnn(obs_sl, state, self.max_seq_len)

            # probs = F.softmax(logits, dim=-1)[:, :-1]
            probs = logits[:, :-1]
            all_values = self.model.value_function().reshape(bs, self.max_seq_len+1)
            value_current = all_values[:, :-1]
            value_next = all_values[:, 1:]
            
            probs = probs.reshape(-1, self.num_actions)
            action_sl = torch.max(action_sl.reshape(-1, self.num_actions), 1)[1]

            policy_loss = self.policy_criterion(probs, action_sl)
            vf_loss = F.mse_loss(reward_sl + 0.99*value_next.detach(), value_current)

            total_loss = policy_loss + vf_loss

            self.optmizer.zero_grad()
            total_loss.backward()
            self.optmizer.step()

            self.tblogger.add_scalar("data/policy_loss", policy_loss, self.learn_iter)
            self.tblogger.add_scalar("data/vf_loss", vf_loss, self.learn_iter)
            self.tblogger.add_scalar("data/total_loss", total_loss, self.learn_iter)

            self.learn_iter += 1
            print(f"{sl} back propogation succeed {total_loss}")




class Worker():
    def __init__(self, start_id, num_episode) -> None:
        self.server = Server(dict(
            map_width=1000,
            map_height=1000,
            save_video=False,
            match_time=60*10,
            # save_path='./videos/'
        )) # server的默认配置就是标准的比赛setting

        self.render = EnvRender(self.server.map_width, self.server.map_height) # 引入渲染模块
        self.server.set_render(self.render) # 设置渲染模块
        self.server.reset() # 初始化游戏引擎
        self._last_team_size = None
        self.env = GoBiggerEnv(env_config)
        self.player_num_per_team = 3
        self.bot_agents = [] # 用于存放本局比赛中用到的所有bot
        self._team_num = 4
        for player in self.server.player_manager.get_players():
            self.bot_agents.append(BotAgent(player.name)) # 初始化每个bot，注意要给每个bot提供队伍名称和玩家名称 
        
        self.worker_id = start_id
        self.num_episode = num_episode
        self.start_id = start_id * self.num_episode


    def collect(self):
        exp = {}
        batch_obs = []
        batch_act = []
        batch_rew = []
        for episode in range(self.start_id, self.start_id+self.num_episode):
            experience = bot_data_one_episode([], [], [])
            self.server.reset() # 初始化游戏引擎
            for i in range(10000):
                # 获取到返回的环境状态信息
                obs = self.server.obs()
                # 动作是一个字典，包含每个玩家的动作

                actions = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in self.bot_agents}
                ma_obs = self.extract_ma_obs(obs)
                ma_a = self.extract_ma_actions(actions)
                team_reward = self._get_reward(obs)
                team_reward = np.array([team_reward[i//3] for i in range(12)])

                experience.obs.append(ma_obs)
                experience.action.append(ma_a)
                experience.reward.append(team_reward)

                finish_flag = self.server.step(actions=actions) # 环境执行动作
                # print('{} {:.4f} leaderboard={}'.format(i, self.server.last_time, obs[0]['leaderboard']))
                if finish_flag:
                    print('{episode}th Game Over')
                    break

            trained_obs = torch.from_numpy(np.concatenate(experience.obs, axis=1))
            trained_actions = torch.from_numpy(np.stack(experience.action, axis=1))
            trained_rewards = torch.from_numpy(np.stack(experience.reward, axis=1))
            batch_obs.append(trained_obs)
            batch_act.append(trained_actions)
            batch_rew.append(trained_rewards)

        batch_dat = [ torch.cat(each, 0) for each in [batch_obs, batch_act, batch_rew] ]
        print(f"{episode} is dumped!")
        pickle.dump(batch_dat, open(f"data/bexp-{self.worker_id}.pkl", 'ab+'), protocol=4)


    def extract_ma_obs(self, obs, teams=[0,1,2,3]):
        obs = self.env._obs_transform(obs)
        ma_obs = []
        for team in teams:
            # Only extract team 0
            for i in range(self.player_num_per_team):
                ma_obs.append(np.concatenate((obs[team][i]["scalar_obs"].reshape(1, -1), obs[team][i]["unit_obs"].reshape(1, -1)), axis=-1))
        return np.stack(ma_obs, axis=0)

    def extract_ma_actions(self, actions):
        action = []
        discret_action = [ (round(v[0]) if v[0] else 1, round(v[1]) if v[1] else 0, round(v[2])) for k, v in actions.items() ]
        for each in discret_action:
            x,y,action_type = each
            if x == 0 and y == 1:
                direction = 0
            elif x == 0 and y == -1:
                direction = 1
            elif x == -1 and y == 0:
                direction = 2
            else:
                direction = 3
            action.append((action_type+1)*4+direction)
        return np.eye(16)[action]

    def _get_reward(self, obs: tuple) -> list:
        global_state, _ = obs

        # reward shaping:
        # 1. difference incremental reward -> cur_size - last_size -> 0.01 * (-40, 40) clip (-1, 1)
        # 2. team rank reward -> cur_size - max_size -> * 0.01 clip(-1, 1)
        # 3. team final win/loss reward -> 5 for 1st; 2 for 2nd; -2 for 3rd; -5 for 4th.

        if self._last_team_size is None:
            team_reward = [np.array([0.]) for __ in range(self._team_num)]
        else:
            team_reward = []
            for i in range(self._team_num):
                team_name = str(i)
                last_size = self._last_team_size[team_name]
                cur_size = global_state['leaderboard'][team_name]
                diff_incremental_reawrd = np.clip(np.array([cur_size - last_size]) * 0.01, -1, 1)
                max_size = max(list(global_state['leaderboard'].values()))
                team_rank_reward = np.clip(np.array([cur_size - max_size]) * 0.001, -1, 0) + 0.5

                team_reward_item = 0.5 * diff_incremental_reawrd + team_rank_reward
                
                team_reward.append(team_reward_item)

            # if global_state['last_time'] >= global_state['total_time']:
            if abs(global_state['last_time'] % 15 - 0) < 0.01 or global_state['last_time'] >= global_state['total_time']:
                rank = np.array(list(global_state['leaderboard'].values()))
                rank = np.argsort(rank)[::-1]
                final_reward = [5, 2, -2, -5]
                for i in range(len(rank)):
                    team_reward[rank[i]] += final_reward[i]
            
        self._last_team_size = global_state['leaderboard']
        return team_reward


def env_func(env, **kwargs):
    return env(**kwargs)

def remote_worker(conn, worker):
    worker = worker.x()
    print(f" {worker.start_id} begin collect! ")
    worker.collect()

cuda = True

if __name__ == '__main__':
    collect_data = False
    
    if collect_data:
        multiprocessing.set_start_method('spawn')
        parent_conns, worker_conns = zip(*[Pipe() for _ in range(4)])
        ps = [Process(target=remote_worker, args=(worker_conn, CloudpickleWrapper(partial(env_func, env=Worker, start_id=i+2, num_episode=4)))) for i, worker_conn in enumerate(worker_conns)]
        
        for p in ps:
            p.daemon = True
            p.start()
        
        for p in ps:
            p.join()

    else:
        learner = PPOBot()
        if cuda:
            learner.mdoel = learner.model.to(torch.device("cuda:0"))

        # files = glob.glob("/home/xyx/git/ballball/my_submission/entry/data/*.pkl")
        files = glob.glob("./data/*.pkl")
        for f in files:
            f_handler = open(f, 'rb')
            data = pickle.load(f_handler)
            for i in range(50000):
                learner.learn(*data)

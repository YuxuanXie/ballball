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

import pickle
import numpy as np
from torch.optim import Adam
from collections import namedtuple
from torch.nn import functional as F

bot_data_one_episode = namedtuple("bot_data_one_episode", ["obs", "action", "reward"])


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
        self.model = TorchRNNModel(None, None, 16, model_config, "PPOBot")
        # self.state = self.initial_state()
        self.optmizer = Adam(self.model.trainable_variables(), 1e-5)
        self.max_seq_len = 50


    def initial_state(self, batch_size):
        hc = [self.model.get_initial_state() for _ in range(batch_size)]
        state = [torch.cat([each[0].unsqueeze(dim=0) for each in hc], dim=0),  torch.cat([each[1].unsqueeze(dim=0) for each in hc], dim=0)]
        return state

    # input -> {id -> action}
    # def get_actions(self, obs):
    #     bs_transform = self.env._obs_transform(obs)[0]

    #     inputs = []
    #     for i in range(len(self.player_names)):
    #         inputs.append(np.concatenate((bs_transform[0]["scalar_obs"].reshape(1, -1), bs_transform[0]["unit_obs"].reshape(1, -1)), axis=-1))
    #     inputs = torch.from_numpy(np.stack(inputs, axis=0))
        
    #     logits, self.state = self.model.forward_rnn(inputs, self.state, [1, 1, 1])
    #     logits = torch.squeeze(logits, dim=1)
    #     actions = torch.argmax(logits, dim=1, keepdim=False).tolist()
    #     a = {f'{i}' : self.env._to_raw_action(action) for i, action in zip(self.player_names, actions)}
    #     return a

    def learn(self, obs, action, reward, state = None):
        bs, total_seq_len, entity_shape = *obs.shape,
        action = action.float()
        reward = reward.float()

        obs = torch.cat((obs, torch.zeros(bs, 1, entity_shape)), dim=1)

        if state == None:
            state = self.initial_state(bs)

        for sl in range(int(total_seq_len / self.max_seq_len)):
            state = [each.detach() for each in state]
            obs_sl = obs[:, sl*self.max_seq_len : (sl+1)*self.max_seq_len+1]
            reward_sl = reward[:, sl*self.max_seq_len : (sl+1)*self.max_seq_len].squeeze(-1)
            action_sl = action[:, sl*self.max_seq_len:(sl+1)*self.max_seq_len]

            logits, state = self.model.forward_rnn(obs_sl, state, self.max_seq_len)

            probs = F.softmax(logits, dim=-1)[:, :-1]
            all_values = self.model.value_function().reshape(bs, self.max_seq_len+1)
            value_current = all_values[:, :-1]
            value_next = all_values[:, 1:]

            policy_loss = F.cross_entropy(probs, action_sl)
            vf_loss = F.mse_loss(reward_sl + 0.99*value_next.detach(), value_current)

            total_loss = policy_loss + vf_loss

            self.optmizer.zero_grad()
            total_loss.backward()
            self.optmizer.step()
            print(f"{sl} back propogation succeed {total_loss}")




class Worker():
    def __init__(self) -> None:
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


    def collect(self):
        exp = {}
        for episode in range(100):
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
                    print('Game Over')
                    break

            trained_obs = torch.from_numpy(np.concatenate(experience.obs, axis=1))
            trained_actions = torch.from_numpy(np.stack(experience.action, axis=1))
            trained_rewards = torch.from_numpy(np.stack(experience.reward, axis=1))

            exp.update({ episode : [trained_obs, trained_actions, trained_rewards]})

        pickle.dump(exp, open("exp.pkl", 'ab+'), protocol=pickle.HIGHEST_PROTOCOL)


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
        discret_action = [ (round(v[0]), round(v[1]), round(v[2])) for k, v in actions.items() ]
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








if __name__ == '__main__':
    worker = Worker()
    worker.collect()
    # learner = PPOBot()
    # data = pickle.load(open('exp.pkl', 'rb'))
    # learner.learn(*data[1])
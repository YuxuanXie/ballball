from gobigger.agents import BotAgent
from gobigger.server import Server
from gobigger.render import EnvRender
import sys

from tensorflow.python.framework.ops import to_raw_op
sys.path.append('..')
import os
import copy
import torch
from envs import GoBiggerEnv
from model.gb import TorchRNNModel
from config.no_spatial import env_config

import pickle
import numpy as np
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


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
    def __init__(self, checkpoint_path, player_names) -> None:
        self.checkpoint = pickle.load(open(checkpoint_path, 'rb'))
        self.worker_info = pickle.loads(self.checkpoint['worker'])
        self.model = TorchRNNModel(self.worker_info['policy_specs']['policy-0'].observation_space, 
                                    self.worker_info['policy_specs']['policy-0'].action_space, 16, model_config, "PPOBot")
        self.model.load_state_dict(convert_to_torch_tensor(self.worker_info['state']['policy-0']['weights']))
        self.env = GoBiggerEnv(env_config)
        self.player_names = player_names
        self.state = self.initial_state()


    def initial_state(self):
        hc = [self.model.get_initial_state() for _ in range(len(self.player_names))]
        state = [torch.cat([each[0].unsqueeze(dim=0) for each in hc], dim=0),  torch.cat([each[1].unsqueeze(dim=0) for each in hc], dim=0)]
        return state

    # input -> {id -> action}
    def get_actions(self, obs):
        bs_transform = self.env._obs_transform(obs)[0]

        inputs = []
        for i in range(len(self.player_names)):
            inputs.append(np.concatenate((bs_transform[0]["scalar_obs"].reshape(1, -1), bs_transform[0]["unit_obs"].reshape(1, -1)), axis=-1))
        inputs = torch.from_numpy(np.stack(inputs, axis=0))
        
        logits, self.state = self.model.forward_rnn(inputs, self.state, [1, 1, 1])
        logits = torch.squeeze(logits, dim=1)
        actions = torch.argmax(logits, dim=1, keepdim=False).tolist()
        a = {f'{i}' : self.env._to_raw_action(action) for i, action in zip(self.player_names, actions)}
        return a



def launch_a_game():
    server = Server(dict(
        map_width=1000,
        map_height=1000,
        save_video=True,
        match_time=60*10,
        save_path='./videos/'
        )) # server的默认配置就是标准的比赛setting
    render = EnvRender(server.map_width, server.map_height) # 引入渲染模块
    server.set_render(render) # 设置渲染模块
    server.reset() # 初始化游戏引擎

    bot_agents = [] # 用于存放本局比赛中用到的所有bot
    
    for player in server.player_manager.get_players():
        bot_agents.append(BotAgent(player.name)) # 初始化每个bot，注意要给每个bot提供队伍名称和玩家名称 

    # ppo_agent = PPOBot("/Users/yuxuan/git/goBigger/my_submission/entry/results/checkpoint_002000/checkpoint-2000")
    ppo_agent = PPOBot("/Users/yuxuan/git/goBigger/my_submission/entry/results/checkpoint_000800/checkpoint-800", ['0', '1', '2'])

    for i in range(100000):
        # 获取到返回的环境状态信息
        obs = server.obs()
        # 动作是一个字典，包含每个玩家的动作

        actions_bot = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in bot_agents}
        actions = actions_bot

        actions_ppo = ppo_agent.get_actions(obs)
        actions.update(actions_ppo)
        # actions = DQNAgent.get_actions(obs)

        # for id in actions.keys():
        #     if id > '3':
        #         actions[id] = actions_bot[id]

        finish_flag = server.step(actions=actions) # 环境执行动作
        print('{} {:.4f} leaderboard={}'.format(i, server.last_time, obs[0]['leaderboard']))
        if finish_flag:
            print('Game Over')
            break
    server.close()


if __name__ == '__main__' and __package__ is None:
    launch_a_game()
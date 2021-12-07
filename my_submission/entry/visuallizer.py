from gobigger.agents import BotAgent
from gobigger.server import Server
from gobigger.render import EnvRender
import sys
sys.path.append('..')
import os
import copy
import torch
from envs import GoBiggerEnv
from ding.policy import DQNPolicy
from ding.config import compile_config
from model import GoBiggerStructedNetwork
from config.no_spatial import main_config

class DQNBot():
    def __init__(self, player_name) -> None:
        self.name = player_name

        self.cfg = copy.deepcopy(main_config)
        self.cfg = compile_config(
            self.cfg,
            policy=DQNPolicy,
            save_cfg=False,
        )
        self.cfg.env.all_vision = True
        # print(self.cfg)
        self.root_path = os.path.abspath(os.path.dirname(__file__))
        self.model = GoBiggerStructedNetwork(**self.cfg.policy.model)
        self.model.load_state_dict(torch.load(os.path.join(self.root_path, '../supplements', 'ckpt_best.pth.tar'), map_location='cpu')['model'])
        self.policy = DQNPolicy(self.cfg.policy, model=self.model).collect_mode
        self.env = GoBiggerEnv(self.cfg.env)
    
    def get_actions(self, obs):
        obs_transform = self.env._obs_transform(obs)
        obs_transform = {index: each_team for index, each_team in enumerate(obs_transform)}
        raw_actions = []
        forward_result = self.policy.forward(obs_transform, eps=1.0)
        # print(forward_result)
        for key in obs_transform.keys():
            raw_actions += forward_result[key]['action'].tolist()
        actions = {n: GoBiggerEnv._to_raw_action(a) for n, a in zip(obs[1].keys(), raw_actions)}
        return actions

def launch_a_game():
    server = Server(dict(
        map_width=300,
        map_height=300,
        save_video=True,
        match_time=20,
        save_path='./videos/'
        )) # server的默认配置就是标准的比赛setting
    render = EnvRender(server.map_width, server.map_height) # 引入渲染模块
    server.set_render(render) # 设置渲染模块
    server.reset() # 初始化游戏引擎

    bot_agents = [] # 用于存放本局比赛中用到的所有bot

    for player in server.player_manager.get_players():
        bot_agents.append(BotAgent(player.name)) # 初始化每个bot，注意要给每个bot提供队伍名称和玩家名称 

    # DQNAgent = DQNBot('xyx')
    

    for i in range(100000):
        # 获取到返回的环境状态信息
        obs = server.obs()
        # 动作是一个字典，包含每个玩家的动作

        actions_bot = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in bot_agents}
        actions = actions_bot
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
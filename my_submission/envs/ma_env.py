from typing import List

# from torch import random
# import pathlib
# sys.path.append(pathlib.Path(__file__).parent.resolve())
from envs.gobigger_env import GoBiggerEnv
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from easydict import EasyDict
import numpy as np
from gobigger.agents import BotAgent
from gym.spaces import Discrete, Dict, Box, MultiDiscrete
from pygame.math import Vector2
import copy
import random



class RandomPolicy:

    def __init__(self, action_type_shape: int, player_num: int):
        self.collect_data = False  # necessary
        self.action_type_shape = action_type_shape
        self.player_num = player_num

    def forward(self, **kwargs) -> List:
        actions = [np.random.randint(0, self.action_type_shape, size=(1)) for _ in range(self.player_num)]

        return actions

    def reset(self, data_id: list = []) -> None:
        pass


class RulePolicy:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = {str(i): BotAgent(str(i)) for i in range(start, end)}

    def forward(self, data, **kwargs) -> List:

        action = []
        for o in data:   # len(data[env_id]) = player_num_per_team
            raw_obs = o['collate_ignore_raw_obs']
            raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])]  for x in raw_obs['overlap']['clone']]
            key = str(int(o['player_name']))
            bot = self.bot[key]
            action.append(bot.step(raw_obs))

        return action

    def reset(self, data_id: list = []) -> None:
        pass


class MAGoBigger(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        self._env = GoBiggerEnv(env_config)
        self.team_num = env_config.team_num
        self.player_num_per_team = env_config.player_num_per_team
        self.action_type_shape = 16
        
        # team 0
        self.team0 = [str(i) for i in range(self.player_num_per_team)]
        # self.team2_3 = RandomPolicy(16, 2*self.player_num_per_team)
        self.team1 = [str(3 + i) for i in range(self.player_num_per_team)]
        self.team2 = [str(6 + i) for i in range(self.player_num_per_team)]

        self.team3 = RulePolicy(3, self.player_num_per_team)

        self.positive_target_pos = { str(i) : None for i in range(self.team_num * self.player_num_per_team)}
        self.negative_target_pos = { str(i) : None for i in range(self.team_num * self.player_num_per_team)}
        self.prev_pos = { str(i) : None for i in range(self.team_num * self.player_num_per_team)}

        self.cur_obs = None



    def reset(self) -> MultiAgentDict:
        obs = self._env.reset()
        self.original_obs = obs
        ma_obs = self.extract_ma_obs(obs)
        self.cur_obs = copy.deepcopy(ma_obs)
        self.cur_obs.update(self.extract_ma_obs(obs, teams=[1,2]))
        return ma_obs

    def seed(self, seed: int, dynamic_seed: bool = True):
        self._env.seed(seed)
    
    # actions : dict(agent_id -> a)
    def step(self, actions):
        
        actions_copy = copy.deepcopy(actions)
        actions_copy.update({str(i) : np.array([random.randint(0, 2), random.randint(0, 2), random.randint(0,3)]) for i in range(3, 9)})
        for k, v in actions_copy.items():
            # overlap = self._env._env.obs()[1][k]['overlap']
            # overlap = self._env.bot.preprocess(overlap)
            # clone_balls = overlap['clone']
            # my_clone_balls, _ = self.process_clone_balls(clone_balls, k)
            # selected_pos = self.cur_obs[k]['unit_obs'][v[0],0:2].tolist()
            if v[0] == 1 and v[1] == 1:
                actions_copy[k] = np.array([None, None, v[2]]) 
            else:
                direction = Vector2([v[0]-1, v[1]-1]).normalize()
                actions_copy[k] = np.array([direction.x, direction.y, v[2]]) 

        gb_actions = []
        for i in range(3):
            gb_actions.append([actions_copy[str(i*3)], actions_copy[str(i*3+1)], actions_copy[str(i*3+2)]])

        team3_obs = self.original_obs[3]
        actions3 = self.team3.forward(team3_obs)
        # actions2_3 = self.team2_3.forward()
        gb_actions += [np.array(actions3)]
        # gb_actions += actions2_3

        feedback = self._env.step([np.array(i) for i in gb_actions])
        self.original_obs = feedback.obs
        observations = self.extract_ma_obs(feedback.obs)
        self.cur_obs = copy.deepcopy(observations)
        self.cur_obs.update(self.extract_ma_obs(feedback.obs, teams=[1,2]))

        rewards = {i: feedback.reward[0][0] for i in self.team0}
        # rewards.update({i: feedback.reward[1][0] for i in self.team1})
        # rewards.update({i: feedback.reward[2][0] for i in self.team2})

        # for agent_id in rewards.keys():
        #     # print(f"{agent_id} : team_reward = {rewards[agent_id]} intrinsic_reward = {self.get_intrinsic_reward(self._env._env.obs()[1][agent_id], agent_id)}")
        #     intrinsic_reward = np.clip(self.get_intrinsic_reward(self._env._env.obs()[1][agent_id], agent_id)*0.1, -1, 1)
        #     # self._env.info[int(agent_id) // 3]["final_eval_reward"] += np.clip(self.get_intrinsic_reward(self._env._env.obs()[1][agent_id], agent_id)*0.1, -1, 1)
        #     rewards[agent_id] = 0.8*rewards[agent_id] + 0.2*intrinsic_reward

        # for agent_id in range(9,12):
        #     self.get_intrinsic_reward(self._env._env.obs()[1][str(agent_id)], str(agent_id))
        # print(f"{feedback.reward}")
        dones = {i: feedback.done for i in self.team0} 
        # dones.update({i : feedback.done for i in self.team1})
        # dones.update({i : feedback.done for i in self.team2})
        dones['__all__'] = feedback.done 

        info = {}
        info['0'] = {}
        if dones["__all__"]:
            info['0']['final_reward'] = [feedback.info[i]['final_eval_reward'] for i in range(self.team_num)]
            info['0']['total_size'] = [feedback.info[i]['final_size'] for i in range(self.team_num)]
            info['0']['size'] = self._env._env.obs()[0]["leaderboard"]
            info['0']['rank'] = np.argsort(np.array(list(self._env._env.obs()[0]['leaderboard'].values())))[::-1]

        return observations, rewards, dones, info

    @property
    def action_space(self):
        # return Discrete(self.action_type_shape)
        return MultiDiscrete([3,3,4])

    @property
    def observation_space(self):
        return Dict({
            'scalar_obs' : Box(low=-10, high=10, shape=(50,)),
            'unit_obs' : Box(low=-10, high=10, shape=(200,31))
        })

    def extract_ma_obs(self, obs, teams=[0]):
        ma_obs = dict()
        for team in teams:
            # Only extract team 0
            for i in range(self.player_num_per_team):

                ma_obs[str(i + team * self.player_num_per_team)] = {}
                ma_obs[str(i + team * self.player_num_per_team)]['scalar_obs'] = obs[team][i]['scalar_obs']
                ma_obs[str(i + team * self.player_num_per_team)]['unit_obs'] = obs[team][i]['unit_obs']

        return ma_obs


    def process_clone_balls(self, clone_balls, name):
        my_clone_balls = []
        others_clone_balls = []
        for clone_ball in clone_balls:
            if clone_ball['player'] == name:
                my_clone_balls.append(copy.deepcopy(clone_ball))
        my_clone_balls.sort(key=lambda a: a['radius'], reverse=True)

        for clone_ball in clone_balls:
            if clone_ball['player'] != name:
                others_clone_balls.append(copy.deepcopy(clone_ball))
        others_clone_balls.sort(key=lambda a: a['radius'], reverse=True)
        return my_clone_balls, others_clone_balls


    def get_intrinsic_reward(self, raw_obs, agent_id):

        # 4. target position diff incremental reward 
        diff_target_pos_reward = 0

        overlap = raw_obs['overlap']
        overlap = self._env.bot.preprocess(overlap)
        food_balls = overlap['food']
        thorns_balls = overlap['thorns']
        spore_balls = overlap['spore']
        clone_balls = overlap['clone']

        my_clone_balls, others_clone_balls = self.process_clone_balls(clone_balls, agent_id)

        diff_target_pos_reward = 0

        if self.prev_pos[agent_id] and self.negative_target_pos[agent_id]:
            diff_target_pos_reward += (my_clone_balls[0]['position'] - self.negative_target_pos[agent_id]).length() - (self.prev_pos[agent_id] - self.negative_target_pos[agent_id]).length()
        if self.prev_pos[agent_id] and self.positive_target_pos[agent_id]:
            diff_target_pos_reward += (self.prev_pos[agent_id] - self.positive_target_pos[agent_id]).length() - (my_clone_balls[0]['position'] - self.positive_target_pos[agent_id]).length()

        self.prev_pos[agent_id] = my_clone_balls[0]['position']

        if len(others_clone_balls) > 0 and my_clone_balls[0]['radius'] < others_clone_balls[0]['radius']:
            self.negative_target_pos[agent_id] = others_clone_balls[0]['position']
        else:
            min_distance, min_thorns_ball = self._env.bot.process_thorns_balls(thorns_balls, my_clone_balls[0])
            if min_thorns_ball is not None:
                self.positive_target_pos[agent_id] = min_thorns_ball['position']
            else:
                min_distance, min_food_ball = self._env.bot.process_food_balls(food_balls, my_clone_balls[0])
                if min_food_ball is not None:
                    self.positive_target_pos[agent_id] = min_food_ball['position'] 
                else:
                    self.positive_target_pos[agent_id] = Vector2(0, 0)

        # print(f"agent = {agent_id} prev position = {self.prev_pos[agent_id]} neg_pos =  {self.negative_target_pos[agent_id]} pos_pos = {self.positive_target_pos[agent_id]} reward = {diff_target_pos_reward}")

        return diff_target_pos_reward   
    



if __name__ == "__main__":

    env = dict(
        collector_env_num=32,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=1e10,
        player_num_per_team=3,
        team_num=4,
        match_time=60,
        map_height=300,
        map_width=300,
        # team_num=4,
        # match_time=200,
        # map_height=1000,
        # map_width=1000,
        resize_height=160,
        resize_width=160,
        spatial=False,
        speed = False,
        all_vision = False,
        train=True,
        manager=dict(shared_memory=False, ),
    )
    env = MAGoBigger(EasyDict(env))
    obs = env.reset()
    for i in range(1000):
        actions = { str(i) : np.array([random.randint(0,2), random.randint(0,2), random.randint(0,3)]) for i in range(9)}
        env.step(actions)

from typing import List
# import pathlib
# sys.path.append(pathlib.Path(__file__).parent.resolve())
from envs.gobigger_env import GoBiggerEnv
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, EnvType, MultiAgentDict
from easydict import EasyDict
import numpy as np
from gobigger.agents import BotAgent
from gym.spaces import Discrete, Dict, Box 



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
        self.trained_agents = [str(i) for i in range(self.player_num_per_team)]
        self.team1 = RulePolicy(1, self.player_num_per_team)
        self.team2_3 = RandomPolicy(16, 2*self.player_num_per_team)



    def reset(self) -> MultiAgentDict:
        obs = self._env.reset()
        self.original_obs = obs
        ma_obs = self.extract_ma_obs(obs)
        return ma_obs

    def seed(self, seed: int, dynamic_seed: bool = True):
        self._env.seed(seed)
    
    # actions : dict(agent_id -> a)
    def step(self, actions):
        
        for k, v in actions.items():
            actions[k] = np.array([v]) 

        gb_actions = []
        for i in range(self.player_num_per_team):
            gb_actions.append(actions[str(i)])

        team1_obs = self.original_obs[1]
        actions1 = self.team1.forward(team1_obs)
        actions2_3 = self.team2_3.forward()
        gb_actions += [np.array(actions1)]
        gb_actions += actions2_3

        feedback = self._env.step([np.array(i) for i in gb_actions])
        self.original_obs = feedback.obs
        observations = self.extract_ma_obs(feedback.obs)
        rewards = {i: feedback.reward[0][0] for i in self.trained_agents}
        dones = {i: feedback.done for i in self.trained_agents} 
        dones['__all__'] = feedback.done 
        info = {}
        info['0'] = {}
        if dones["__all__"]:
            info['0']['final_reward'] = [feedback.info[i]['final_eval_reward'] for i in range(self.team_num)]
            info['0']['size'] = self._env._env.obs()[0]["leaderboard"]
            info['0']['rank'] = np.argsort(np.array(list(self._env._env.obs()[0]['leaderboard'].values())))[::-1]

        return observations, rewards, dones, info

    @property
    def action_space(self):
        return Discrete(self.action_type_shape)

    @property
    def observation_space(self):
        return Dict({
            'scalar_obs' : Box(low=-1000, high=1000, shape=(50,)),
            'unit_obs' : Box(low=-1000, high=1000, shape=(200,31))
        })

    def extract_ma_obs(self, obs):
        ma_obs = dict()
        # Only extract team 0
        for i in range(self.player_num_per_team):
            ma_obs[str(i)] = {}
            ma_obs[str(i)]['scalar_obs'] = obs[0][i]['scalar_obs']
            ma_obs[str(i)]['unit_obs'] = obs[0][i]['unit_obs']
        return ma_obs
    



if __name__ == "__main__":

    env=dict(
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
    env = MAGoBigger(env)
    obs = env.reset()
    actions = { str(i) : np.array([1]) for i in range(3)}
    env.step(actions)
import random
import os
import gym
import numpy as np
import copy
import torch
import time

from .envs import GoBiggerEnv
from .config.no_spatial import env_config
from .model.gb import TorchRNNModel

import pickle
import numpy as np
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from pygame.math import Vector2


model_config = {
    "custom_model": "go_bigger", 
    "lstm_cell_size": 128 ,
    "max_seq_len" : 8,
    "vf_share_layers": True,
    "custom_model_config": {
        "obs_shape" : 50,
        "entity_shape" : 31,
        "obs_embedding_size" : 32,
        "entity_embedding_size" : 64,
        "all_embedding_size" : 128,
    }
}

class PPOBot():
    def __init__(self, checkpoint_path, player_names) -> None:
        self.checkpoint = pickle.load(open(checkpoint_path, 'rb'))
        self.worker_info = pickle.loads(self.checkpoint['worker'])
        self.model = TorchRNNModel(self.worker_info['policy_specs']['policy-0'].observation_space, self.worker_info['policy_specs']['policy-0'].action_space, 16, model_config, "PPOBot")
        self.model.load_state_dict(convert_to_torch_tensor(self.worker_info['state']['policy-0']['weights']))
        # self.model = TorchRNNModel(None, None, 16, model_config, "PPOBot")
        # self.model.load_state_dict(torch.load("1.pkl", map_location='cpu'))
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
        logits_split = torch.split(logits, [3,3,4], dim=-1)
        a = {}
        agent_id = 0
        for x, y, type in zip(*logits_split):
            x = torch.argmax(x, dim=0, keepdim=False).tolist()
            y = torch.argmax(y, dim=0, keepdim=False).tolist()
            type = torch.argmax(type, dim=0, keepdim=False).tolist()

            if x == 1 and y == 1:
                a[agent_id] = np.array([None, None, type]) 
            else:
                direction = Vector2([x-1, y-1]).normalize()
                a[agent_id] = np.array([direction.x, direction.y, type]) 

            agent_id+=1
        return a



class BaseSubmission:

    def __init__(self, team_name, player_names):
        self.team_name = team_name
        self.player_names = player_names

    def get_actions(self, obs):
        '''
        Overview:
            You must implement this function.
        '''
        raise NotImplementedError


class MySubmission(BaseSubmission):

    def __init__(self, team_name, player_names):
        super(MySubmission, self).__init__(team_name, player_names)
        self.cfg = env_config
        print(self.cfg)
        self.root_path = os.path.abspath(os.path.dirname(__file__))
        # self.model = GoBiggerStructedNetwork(**self.cfg.policy.model)
        # self.model.load_state_dict(torch.load(os.path.join(self.root_path, 'supplements', 'ckpt_best.pth.tar'), map_location='cpu')['model'])
        # self.policy = DQNPolicy(self.cfg.policy, model=self.model).eval_mode
        self.env = GoBiggerEnv(self.cfg)
        # self.botAgents = [BotAgent(i) for i in self.player_names]
        import pdb; pdb.set_trace()
        self.ppo_agent = PPOBot(os.path.join(self.root_path, 'supplements', 'checkpoint'), player_names)

    def get_actions(self, obs):
        # obs_transform = self.env._obs_transform(obs)[0]
        # obs_transform = {0: obs_transform}
        # raw_actions = self.policy.forward(obs_transform)[0]['action']
        # raw_actions = raw_actions.tolist()
        # actions = {n: GoBiggerEnv._to_raw_action(a) for n, a in zip(obs[1].keys(), raw_actions)}
        # ------------------------------------------------------------------------------------------------
        # actions = {bot_agent.name: bot_agent.step(obs[1][bot_agent.name]) for bot_agent in self.botAgents}
        # ------------------------------------------------------------------------------------------------

        actions=self.ppo_agent.get_actions(obs)

        return actions

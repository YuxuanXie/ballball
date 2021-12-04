
import os
import numpy as np
import copy
from tensorboardX import SummaryWriter
import torch
import sys
import datetime
sys.path.append('..')

from ding.config import compile_config
from ding.worker import BattleSampleSerialCollector, BattleInteractionSerialEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from gobigger.agents import BotAgent

from envs import GoBiggerEnv
from model import GoBiggerStructedNetwork
from config.no_spatial import main_config
import pickle


class RulePolicy:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = {str(i): BotAgent(str(i)) for i in range(start, end)}

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for o in data[env_id]:   # len(data[env_id]) = player_num_per_team
                raw_obs = o['collate_ignore_raw_obs']
                raw_obs['overlap']['clone'] = [[x[0], x[1], x[2], int(x[3]), int(x[4])]  for x in raw_obs['overlap']['clone']]
                key = str(int(o['player_name']))
                bot = self.bot[key]
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


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


class CollectorManager():
    def __init__(self, conn, cfg) -> None:
        self.cfg = cfg
        self.conn = conn
        self.seed = 0
        self.cfg.policy.cuda = False

        self.collector_env_num = self.cfg.env.collector_env_num
        self.collector_env_cfg = copy.deepcopy(self.cfg.env)
        self.collector_env_cfg.train = True
        self.collector_env = SyncSubprocessEnvManager(env_fn=[lambda: GoBiggerEnv(self.collector_env_cfg) for _ in range(self.collector_env_num)], cfg=self.cfg.env.manager)
        self.collector_env.seed(self.seed)
        self.rule_collect_policy = [RulePolicy(team_id, self.cfg.env.player_num_per_team) for team_id in range(1, self.cfg.env.team_num)]
        self.model = GoBiggerStructedNetwork(**self.cfg.policy.model)
        self.policy = DQNPolicy(self.cfg.policy, model=self.model)

        self.eps_cfg = self.cfg.policy.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(self.eps_cfg.start, self.eps_cfg.end, self.eps_cfg.decay, self.eps_cfg.type)


        # self.collector = BattleSampleSerialCollector(cfg.policy.collect.collector, self.collector_env, [policy.collect_mode] + rule_collect_policy, tb_logger, exp_name=cfg.exp_name)
        self.collector = BattleSampleSerialCollector(self.cfg.policy.collect.collector, self.collector_env, [self.policy.collect_mode] + self.rule_collect_policy, exp_name=self.cfg.exp_name)

    def update_model(self, state_dict):
        print("Update model!")
        # print(state_dict)
        self.policy.collect_mode.load_state_dict(state_dict)


    def send_exp(self, new_data):
        self.conn.send({
            "new_data": new_data, 
            "env_step": self.collector.envstep
            })
    
    def start(self, ):

        print("waiting for first model")
        learner_info = self.conn.recv()
        print("Get mdoel")
        print(learner_info.keys())
        # print(pickle.loads(learner_info['state_dict']))
        # state_dict = pickle.loads(learner_info['state_dict'])
        self.train_iter = learner_info["train_iter"]
        # self.update_model(state_dict)
        print("Get mdoel")

        while True:
            if self.conn.poll():
                learner_info = self.conn.recv()
                state_dict = learner_info["state_dict"]
                self.train_iter = learner_info["train_iter"]
                # self.update_model(state_dict)
            
            print("Start collecting")
            eps = self.epsilon_greedy(self.collector.envstep)
            new_data, _ = self.collector.collect(train_iter=self.train_iter, policy_kwargs={'eps': eps})
            self.conn.send_exp(new_data)
            print("Send data")



def remote_worker(conn, cfg):
    collector = CollectorManager(conn, cfg.x)
    print("start remote worker!")
    collector.start()

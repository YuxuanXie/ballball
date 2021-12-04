import os
from ding import worker
import numpy as np
import copy
from tensorboardX import SummaryWriter
import torch
import sys
import datetime
sys.path.append('..')

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import SyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from gobigger.agents import BotAgent

from envs import GoBiggerEnv
from model import GoBiggerStructedNetwork
from config.no_spatial import main_config

from worker import remote_worker, CloudpickleWrapper
from multiprocessing import Pipe, Process
import time
import pickle


class RandomPolicy:

    def __init__(self, action_type_shape: int, player_num: int):
        self.action_type_shape = action_type_shape
        self.player_num = player_num

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action': np.random.randint(0, self.action_type_shape, size=(self.player_num))
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass


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


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg.exp_name = cfg.exp_name + '-' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_envstep = 0
    # collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env_num = cfg.env.evaluator_env_num
    # collector_env_cfg = copy.deepcopy(cfg.env)
    # collector_env_cfg.train = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.train = False
    # collector_env = SyncSubprocessEnvManager(
    #     env_fn=[lambda: GoBiggerEnv(collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    # )
    random_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    rule_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    # collector_env.seed(seed)
    random_evaluator_env.seed(seed, dynamic_seed=False)
    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = GoBiggerStructedNetwork(**cfg.policy.model)
    # load_path='/home/xyx/git/GoBigger-Challenge-2021/di_baseline/my_submission/entry/gobigger_no_spatial_baseline_dqn/ckpt/ckpt_best.pth.tar'
    # model.load_state_dict(torch.load(load_path , map_location='cpu')['model'])
    policy = DQNPolicy(cfg.policy, model=model)
    team_num = cfg.env.team_num
    # rule_collect_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
    rule_eval_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
    # eps_cfg = cfg.policy.other.eps
    # epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner'
    )

    # collector = BattleSampleSerialCollector(
    #     cfg.policy.collect.collector, collector_env, [policy.collect_mode] + rule_collect_policy, tb_logger, exp_name=cfg.exp_name
    # )
    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator, rule_evaluator_env, [policy.eval_mode] + rule_eval_policy, tb_logger, exp_name=cfg.exp_name, instance_name='rule_evaluator'
    )

    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    parent, child = Pipe()
    p = Process(target=remote_worker, args=(child, CloudpickleWrapper(cfg)))
    # p.daemon=True
    p.start()
    
    # child.close()
    
    # parent.send({"state_dict": policy.learn_mode.state_dict(), "train_iter": learner.train_iter})
    parent.send({"state_dict": [1], "train_iter": learner.train_iter})

    for _ in range(max_iterations):
        
        if parent.poll() or learner.policy.get_attribute('batch_size') > replay_buffer.count():
            print("waiting!")
            worker_info = parent.recv()
            collector_envstep = worker_info["env_step"]
            new_data = worker_info["new_data"]

            replay_buffer.push(new_data[0], cur_collector_envstep=collector_envstep)
            replay_buffer.push(new_data[1], cur_collector_envstep=collector_envstep)


        for i in range(cfg.policy.learn.update_per_collect):
            print("begin training!")
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector_envstep)
        
        parent.send({"state_dict":policy.learn_mode.state_dict(), "train_iter": learner.train_iter})




if __name__ == "__main__":
    main(main_config)

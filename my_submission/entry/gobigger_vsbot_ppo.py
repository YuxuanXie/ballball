import os
import numpy as np
import copy
from tensorboardX import SummaryWriter
import sys
sys.path.append('..')

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from gobigger.agents import BotAgent

from envs import GoBiggerEnv
from model import GoBiggerPPO
from config.gobigger_no_spatial_ppo import main_config

import datetime


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
    cfg.exp_name = cfg.exp_name + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        PPOPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.train = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.train = False
    collector_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    random_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    rule_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    random_evaluator_env.seed(seed, dynamic_seed=False)
    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = GoBiggerPPO(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    team_num = cfg.env.team_num
    rule_collect_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]
    random_eval_policy = RandomPolicy(
        cfg.policy.model.action_type_shape, cfg.env.player_num_per_team
    )
    rule_eval_policy = [RulePolicy(team_id, cfg.env.player_num_per_team) for team_id in range(1, team_num)]

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner'
    )
    collector = BattleSampleSerialCollector(
        cfg.policy.collect.collector,
        collector_env, [policy.collect_mode] + rule_collect_policy,
        tb_logger,
        exp_name=cfg.exp_name
    )
    random_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        random_evaluator_env, [policy.eval_mode] + [random_eval_policy for _ in range(team_num - 1)],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='random_evaluator'
    )
    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        rule_evaluator_env, [policy.eval_mode] + rule_eval_policy,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='rule_evaluator'
    )

    for _ in range(max_iterations):
        # Sampling data from environments
        new_data, _ = collector.collect(train_iter=learner.train_iter)
        for i in range(cfg.policy.learn.update_per_collect):
            learner.train(new_data, collector.envstep)

        if random_evaluator.should_eval(learner.train_iter):
            random_stop_flag, random_reward, _ = random_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep)
            rule_stop_flag, rule_reward, _ = rule_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep)
            if random_stop_flag and rule_stop_flag:
                break


if __name__ == "__main__":
    main(main_config)

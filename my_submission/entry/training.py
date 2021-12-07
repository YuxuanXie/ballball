import ray
import sys
sys.path.append('..')

from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy as PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
import tensorflow.compat.v1 as tf
from ray.tune.result import DEFAULT_RESULTS_DIR

from envs.ma_env import MAGoBigger
from config.no_spatial import env_config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'exp_name', None,
    'Name of the ray_results experiment directory where results are stored.')
tf.app.flags.DEFINE_string(
    'env', 'gb',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_string(
    'algorithm', 'PPO',
    'Name of the rllib algorithm to use.')
tf.app.flags.DEFINE_integer(
    'num_agents', 5,
    'Number of agent policies')
tf.app.flags.DEFINE_integer(
    'train_batch_size', 8,
    'Size of the total dataset over which one epoch is computed.')
tf.app.flags.DEFINE_integer(
    'checkpoint_frequency', 20,
    'Number of steps before a checkpoint is saved.')
tf.app.flags.DEFINE_integer(
    'training_iterations', 2000,
    'Total number of steps to train for')
tf.app.flags.DEFINE_integer(
    'num_cpus', 32,
    'Number of available CPUs')
tf.app.flags.DEFINE_integer(
    'num_gpus', 1,
    'Number of available GPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpus_for_workers', False,
    'Set to true to run workers on GPUs rather than CPUs')
tf.app.flags.DEFINE_boolean(
    'use_gpu_for_driver', True,
    'Set to true to run driver on GPU rather than CPU.')
# num_workers_per_device increases and the sample time also increases. 2 is better here 
tf.app.flags.DEFINE_float(
    'num_workers_per_device', 1,
    'Number of workers to place on a single device (CPU or GPU)')
tf.app.flags.DEFINE_float(
    'num_cpus_for_driver', 1,
    'Number of workers to place on a single device (CPU or GPU)')

tf.app.flags.DEFINE_float('entropy_coeff', 0.00, 'The entropy')
# tf.app.flags.DEFINE_float('strong_proportion', 0.5, 'The proportion of strong agents')
tf.app.flags.DEFINE_float('gamma', 0.99, 'gamma')
tf.app.flags.DEFINE_float('lam', 0.99, 'lambda')


gc_default_params = {
    'lr_init': 1e-4,
    'lr_final': 1e-4,
}
ppo_params = {
    'entropy_coeff': 0.001,
    #'entropy_coeff_schedule': [[0, FLAGS.entropy_coeff],[2000000, 0.0]],
    'use_gae': True,
    'kl_coeff': 0.1,
    "lambda" : FLAGS.lam,
    "gamma" : FLAGS.gamma,
    "clip_param" : 0.3,
    "sgd_minibatch_size" : 128,
    "train_batch_size" : 256,
    "num_sgd_iter" : 4,
    # "rollout_fragment_length" : 200,
    # "sgd_minibatch_size" : 128*5,
    # "train_batch_size" : 5000,
    # "num_sgd_iter" : 8,
    # "evaluation_interval" : 100,
    # "evaluation_num_episodes" : 50,
    # "evaluation_config" : {"explore": False},
    # "batch_mode" : "complete_episodes",
}



def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):


    def env_creator(_):
        return MAGoBigger(env_config)

    single_env = MAGoBigger(env_config)
    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (PPOPolicyGraph, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policy graphs
    policy_graphs = {}
    policy_graphs['policy-0'] = gen_policy()
    
    def policy_mapping_fn(agent_id):
        return 'policy-0'


    # # register the custom model
    # model_name = "conv_to_fc_net"
    # ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
                "train_batch_size": train_batch_size,
                # "horizon": 3 * one_layer_length, # it dosnot make done in step function true
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [2000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "framework" : "torch",
                "multiagent": {
                    "policies": policy_graphs,
                    "policy_mapping_fn": tune.function(policy_mapping_fn),
                },
                # "model": {"custom_model": "conv_to_fc_net", "lstm_cell_size": 128 },
    })

    config.update(ppo_params)
    config.update({"callbacks": {
        "on_episode_end": tune.function(on_episode_end),
    }})
    return algorithm, env_name, config


def on_episode_end(info):
    episode = info["episode"]
    info = episode._agent_to_last_info
    if info["0"]:
        for i in range(4):
            episode.custom_metrics[f"reward{i}"] = info["0"]["final_reward"][i]
            episode.custom_metrics[f"size{i}"] = info["0"]["size"][str(i)]
            episode.custom_metrics[f"rank{i}"] = info["0"]["rank"][i]

def main(unused_argv):
    ray.init(num_cpus=FLAGS.num_cpus)
    hparams = gc_default_params
    alg_run, env_name, config = setup(FLAGS.env, hparams, FLAGS.algorithm,
                                      FLAGS.train_batch_size,
                                      FLAGS.num_cpus,
                                      FLAGS.num_gpus, FLAGS.num_agents,
                                      FLAGS.use_gpus_for_workers,
                                      FLAGS.use_gpu_for_driver,
                                      FLAGS.num_workers_per_device)

    if FLAGS.exp_name is None:
        exp_name = FLAGS.env + '_' + FLAGS.algorithm
    else:
        exp_name = FLAGS.exp_name + '/'

    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": FLAGS.training_iterations
            },
            'checkpoint_freq': FLAGS.checkpoint_frequency,
            "config": config,
        }},
    )



if __name__ == '__main__':
    tf.app.run(main)



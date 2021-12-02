from easydict import EasyDict

gobigger_ppo_config = dict(
    exp_name='gobigger_no_spatial_ppo',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        stop_value=1e10,
        player_num_per_team=3,
        team_num=4,
        match_time=200,
        map_height=1000,
        map_width=1000,
        resize_height=160,
        resize_width=160,
        spatial=False,
        speed = False,
        all_vision = False,
        train=True,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=False,
        continuous=False,
        model=dict(
            obs_shape=50,
            action_shape=16,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=4,
            batch_size=128,
            learning_rate=0.0001,
            value_weight=1.0,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=100,
            ),
        ),
    ),
)
gobigger_ppo_config = EasyDict(gobigger_ppo_config)
main_config = gobigger_ppo_config
gobigger_ppo_create_config = dict(
    env=dict(
        type='gobigger',
        import_names=['dizoo.gobigger.envs.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
gobigger_ppo_create_config = EasyDict(gobigger_ppo_create_config)
create_config = gobigger_ppo_create_config


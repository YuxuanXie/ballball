from easydict import EasyDict

gobigger_dqn_config = dict(
    exp_name='results/vsbot',
    env=dict(
        collector_env_num=2,
        evaluator_env_num=2,
        n_evaluator_episode=2,
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
    ),
    policy=dict(
        cuda=True,
        model=dict(
            scalar_shape=50,
            per_unit_shape=31,
            # scalar_shape=36,
            # per_unit_shape=23,
            action_type_shape=16,
            rnn=False,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=2,
            batch_size=512,
            learning_rate=0.0005,
            ignore_done=True,
            learner=dict(
                log_show_after_iter =10, 
                save_ckpt_after_iter=100,
            )
        ),
        collect=dict(
            n_sample=512, 
            unroll_len=1, 
            collector=dict(
                collect_print_freq = 10,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.5,
                end=0.1,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=50000, ),
        ),
    ),
)
gobigger_dqn_config = EasyDict(gobigger_dqn_config)
main_config = gobigger_dqn_config
gobigger_dqn_create_config = dict(
    env=dict(
        type='gobigger',
        import_names=['dizoo.gobigger.envs.gobigger_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
gobigger_dqn_create_config = EasyDict(gobigger_dqn_create_config)
create_config = gobigger_dqn_create_config

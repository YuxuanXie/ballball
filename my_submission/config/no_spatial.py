from easydict import EasyDict

env_config=dict(
    collector_env_num=1,
    evaluator_env_num=8,
    n_evaluator_episode=8,
    stop_value=1e10,
    player_num_per_team=3,
    team_num=4,
    match_time=300,
    map_height=1000,
    map_width=1000,
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

env_config = EasyDict(env_config)
